import math

import pytest
import torch

from posttrain.ops import token_logprobs
from posttrain.rl import PolicyLossConfig, aggregate_token_losses, policy_gradient_loss

MAX_TOKENS = 12

CONFIGS = {
    "grpo": PolicyLossConfig.grpo(),
    "dapo": PolicyLossConfig.dapo(),
    "dr_grpo": PolicyLossConfig.dr_grpo(),
    # k1 is not a useful penalty (zero expected gradient on-policy, see
    # posttrain.rl.kl); this configuration only exercises the k1 code path.
    "seq_k1_code_path": PolicyLossConfig(normalization="sequence", beta=0.1, kl_estimator="k1"),
    "token_k2": PolicyLossConfig(normalization="token", beta=0.3, kl_estimator="k2"),
}


def _batch(seed: int, batch: int = 4, seq_len: int = 10, requires_grad: bool = False):
    """Random responses of different lengths, with left prompt and right padding."""
    gen = torch.Generator().manual_seed(seed)
    old_logp = -torch.rand(batch, seq_len, generator=gen, dtype=torch.float64) * 3
    # Ratios in roughly [0.6, 1.65] so that some tokens fall outside the clip range.
    logp = old_logp + (torch.rand(batch, seq_len, generator=gen, dtype=torch.float64) - 0.5)
    ref_logp = old_logp + 0.3 * torch.randn(batch, seq_len, generator=gen, dtype=torch.float64)
    advantages = torch.randn(batch, generator=gen, dtype=torch.float64)
    mask = torch.zeros(batch, seq_len, dtype=torch.bool)
    for i in range(batch):
        start = 1 + i % 2  # prompt tokens are masked out
        mask[i, start : start + 2 + 2 * i] = True
    logp.requires_grad_(requires_grad)
    return logp, old_logp, ref_logp, advantages, mask


def _reference_loss(logp, old_logp, ref_logp, advantages, mask, cfg: PolicyLossConfig) -> float:
    """Loop-based transcription of the paper equations, independent of the library."""
    lp, old, ref, m = logp.tolist(), old_logp.tolist(), ref_logp.tolist(), mask.tolist()
    per_sequence: list[list[float]] = []
    for i, row in enumerate(m):
        losses = []
        for t, valid in enumerate(row):
            if not valid:
                continue
            ratio = math.exp(lp[i][t] - old[i][t])
            clipped = min(max(ratio, 1 - cfg.eps_low), 1 + cfg.eps_high)
            a = float(advantages[i])
            loss = -min(ratio * a, clipped * a)
            if cfg.beta:
                log_r = ref[i][t] - lp[i][t]
                kl = {
                    "k1": -log_r,
                    "k2": 0.5 * log_r**2,
                    "k3": math.exp(log_r) - 1 - log_r,
                }[cfg.kl_estimator]
                loss += cfg.beta * kl
            losses.append(loss)
        per_sequence.append(losses)
    if cfg.normalization == "sequence":
        return sum(sum(s) / len(s) for s in per_sequence) / len(per_sequence)
    total = sum(sum(s) for s in per_sequence)
    if cfg.normalization == "token":
        return total / sum(len(s) for s in per_sequence)
    return total / (len(per_sequence) * MAX_TOKENS)


@pytest.mark.parametrize("name", CONFIGS)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_loop_reference(name, seed):
    cfg = CONFIGS[name]
    logp, old_logp, ref_logp, adv, mask = _batch(seed)

    out = policy_gradient_loss(
        logp, old_logp, adv, mask, cfg, ref_logp=ref_logp, max_tokens=MAX_TOKENS
    )

    expected = _reference_loss(logp, old_logp, ref_logp, adv, mask, cfg)
    assert out.loss.item() == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("name", CONFIGS)
def test_gradcheck(name):
    cfg = CONFIGS[name]
    _, old_logp, ref_logp, adv, mask = _batch(seed=5)

    def loss_fn(logp):
        out = policy_gradient_loss(
            logp, old_logp, adv, mask, cfg, ref_logp=ref_logp, max_tokens=MAX_TOKENS
        )
        return out.loss

    logp = _batch(seed=5, requires_grad=True)[0]
    assert torch.autograd.gradcheck(loss_fn, (logp,), eps=1e-6, atol=1e-8)


@pytest.mark.parametrize("name", CONFIGS)
def test_prompt_and_pad_tokens_have_no_effect(name):
    cfg = CONFIGS[name]
    logp, old_logp, ref_logp, adv, mask = _batch(seed=7, requires_grad=True)
    base = policy_gradient_loss(
        logp, old_logp, adv, mask, cfg, ref_logp=ref_logp, max_tokens=MAX_TOKENS
    )
    (grad,) = torch.autograd.grad(base.loss, logp)

    # Garbage, including -inf, at every masked position.
    junk = torch.full_like(old_logp, -math.inf)
    junk_logp = torch.where(mask, logp.detach(), junk).requires_grad_(True)
    perturbed = policy_gradient_loss(
        junk_logp,
        torch.where(mask, old_logp, junk),
        adv,
        mask,
        cfg,
        ref_logp=torch.where(mask, ref_logp, torch.full_like(ref_logp, 3.0)),
        max_tokens=MAX_TOKENS,
    )

    (junk_grad,) = torch.autograd.grad(perturbed.loss, junk_logp)

    assert perturbed.loss.item() == base.loss.item()
    assert torch.equal(grad[~mask], torch.zeros_like(grad[~mask]))
    assert torch.equal(junk_grad, grad)
    assert torch.isfinite(perturbed.per_token_loss).all()


@pytest.mark.parametrize("name", CONFIGS)
def test_non_finite_masked_logits_give_finite_gradients_end_to_end(name):
    """Logits -> token_logprobs -> loss with -inf rows at prompt and pad positions."""
    cfg = CONFIGS[name]
    gen = torch.Generator().manual_seed(9)
    _, _, _, adv, mask = _batch(seed=9)
    logits = torch.randn(*mask.shape, 7, generator=gen, dtype=torch.float64)
    targets = torch.randint(0, 7, mask.shape, generator=gen)
    targets = torch.where(mask, targets, torch.full_like(targets, -100))
    with torch.no_grad():
        old_logp = token_logprobs(logits + 0.1, targets, mask)
        ref_logp = token_logprobs(logits - 0.1, targets, mask)

    def loss_and_grad(x):
        x = x.clone().requires_grad_(True)
        logp = token_logprobs(x, targets, mask)
        out = policy_gradient_loss(
            logp, old_logp, adv, mask, cfg, ref_logp=ref_logp, max_tokens=MAX_TOKENS
        )
        (grad,) = torch.autograd.grad(out.loss, x)
        return out.loss, grad

    base_loss, base_grad = loss_and_grad(logits)
    dirty = torch.where(mask.unsqueeze(-1), logits, torch.full_like(logits, -math.inf))
    loss, grad = loss_and_grad(dirty)

    assert loss.item() == base_loss.item()
    assert torch.isfinite(grad).all()
    assert torch.equal(grad, base_grad)


def _on_policy_token_weights(cfg: PolicyLossConfig, advantages, lengths, seq_len=8):
    """d loss / d logp at ratio 1: the weight each token receives in the update."""
    mask = torch.zeros(len(lengths), seq_len, dtype=torch.bool)
    for i, n in enumerate(lengths):
        mask[i, :n] = True
    logp = torch.full((len(lengths), seq_len), -1.0, dtype=torch.float64, requires_grad=True)
    out = policy_gradient_loss(logp, logp.detach(), advantages, mask, cfg, max_tokens=seq_len)
    (grad,) = torch.autograd.grad(out.loss, logp)
    return grad, mask


def test_sequence_normalization_has_length_bias_that_dr_grpo_removes():
    """Two equally good answers of length 2 and 6, both with advantage +1.

    With the original GRPO normalization each token of response i gets the
    weight -A_i / (G |o_i|): the short answer's tokens are pushed three times
    harder. The constant normalizer of Dr.GRPO gives every token the same
    weight -A_i / (G L_max).
    """
    advantages = torch.tensor([1.0, 1.0], dtype=torch.float64)
    lengths = [2, 6]

    grad, mask = _on_policy_token_weights(
        PolicyLossConfig(normalization="sequence"), advantages, lengths
    )
    torch.testing.assert_close(grad[0, :2], torch.full((2,), -1 / (2 * 2), dtype=torch.float64))
    torch.testing.assert_close(grad[1, :6], torch.full((6,), -1 / (2 * 6), dtype=torch.float64))
    assert (grad[0, 0] / grad[1, 0]).item() == pytest.approx(3.0)

    grad, mask = _on_policy_token_weights(PolicyLossConfig.dr_grpo(), advantages, lengths)
    torch.testing.assert_close(grad[mask], torch.full((8,), -1 / (2 * 8), dtype=torch.float64))

    grad, mask = _on_policy_token_weights(PolicyLossConfig.dapo(), advantages, lengths)
    torch.testing.assert_close(grad[mask], torch.full((8,), -1 / 8, dtype=torch.float64))


def test_sequence_normalization_penalizes_long_wrong_answers_less_per_token():
    advantages = torch.tensor([-1.0, -1.0], dtype=torch.float64)

    grad, _ = _on_policy_token_weights(
        PolicyLossConfig(normalization="sequence"), advantages, [2, 6]
    )

    # Gradient descent lowers log-probs proportionally to these weights; the long
    # wrong answer is discouraged three times less per token.
    assert grad[0, 0].item() == pytest.approx(3 * grad[1, 0].item())
    assert grad[1, 0] > 0


def _single_token_grad(cfg: PolicyLossConfig, ratio: float, advantage: float) -> float:
    logp = torch.tensor([[math.log(ratio)]], dtype=torch.float64, requires_grad=True)
    old_logp = torch.zeros(1, 1, dtype=torch.float64)
    mask = torch.ones(1, 1, dtype=torch.bool)
    adv = torch.tensor([advantage], dtype=torch.float64)
    out = policy_gradient_loss(logp, old_logp, adv, mask, cfg)
    (grad,) = torch.autograd.grad(out.loss, logp)
    return grad.item()


def test_clip_higher_keeps_gradient_for_positive_advantage_tokens():
    symmetric = PolicyLossConfig(normalization="token", eps_low=0.2, eps_high=0.2)
    clip_higher = PolicyLossConfig(normalization="token", eps_low=0.2, eps_high=0.28)

    # ratio 1.25 with A > 0: outside the symmetric range, inside clip-higher.
    assert _single_token_grad(symmetric, 1.25, 1.0) == 0.0
    assert _single_token_grad(clip_higher, 1.25, 1.0) == pytest.approx(-1.25)
    # Beyond 1.28 both clip.
    assert _single_token_grad(clip_higher, 1.3, 1.0) == 0.0
    # The lower bound is untouched: ratio 0.75 with A < 0 is clipped by both.
    assert _single_token_grad(symmetric, 0.75, -1.0) == 0.0
    assert _single_token_grad(clip_higher, 0.75, -1.0) == 0.0


def test_clipping_never_removes_the_corrective_gradient():
    """PPO only clips moves *in* the advantage direction.

    A token whose ratio already went the wrong way (ratio > 1 with A < 0, or
    ratio < 1 with A > 0) keeps its full gradient so the update can undo it.
    """
    cfg = PolicyLossConfig(normalization="token")
    assert _single_token_grad(cfg, 1.5, -1.0) == pytest.approx(1.5)
    assert _single_token_grad(cfg, 0.5, 1.0) == pytest.approx(-0.5)


def test_clip_fraction_counts_tokens_with_zeroed_gradient():
    logp = torch.log(torch.tensor([[1.25, 1.0, 0.7, 1.5]], dtype=torch.float64))
    old_logp = torch.zeros_like(logp)
    mask = torch.ones_like(logp, dtype=torch.bool)
    out = policy_gradient_loss(
        logp, old_logp, torch.tensor([1.0], dtype=torch.float64), mask, PolicyLossConfig()
    )
    # 1.25 and 1.5 are clipped for A > 0; 0.7 is below the range but A > 0 keeps it.
    assert out.clip_fraction.item() == pytest.approx(0.5)


def test_kl_penalty_requires_reference_and_constant_requires_budget():
    logp, old_logp, _, adv, mask = _batch(seed=0)
    with pytest.raises(ValueError, match="ref_logp"):
        policy_gradient_loss(logp, old_logp, adv, mask, PolicyLossConfig.grpo())
    with pytest.raises(ValueError, match="max_tokens"):
        policy_gradient_loss(logp, old_logp, adv, mask, PolicyLossConfig.dr_grpo())


MICRO_BATCHES = [[0], [1, 2], [3]]  # uneven split of the 4 responses in _batch


def _accumulated(cfg, logp, old_logp, ref_logp, adv, mask, **counts):
    total = torch.zeros((), dtype=logp.dtype)
    for rows in MICRO_BATCHES:
        out = policy_gradient_loss(
            logp[rows],
            old_logp[rows],
            adv[rows],
            mask[rows],
            cfg,
            ref_logp=ref_logp[rows],
            max_tokens=MAX_TOKENS,
            **counts,
        )
        total = total + out.loss
    return total


@pytest.mark.parametrize("name", CONFIGS)
def test_gradient_accumulation_with_global_counts_equals_full_batch(name):
    cfg = CONFIGS[name]
    logp, old_logp, ref_logp, adv, mask = _batch(seed=11, requires_grad=True)
    full = policy_gradient_loss(
        logp, old_logp, adv, mask, cfg, ref_logp=ref_logp, max_tokens=MAX_TOKENS
    ).loss
    (full_grad,) = torch.autograd.grad(full, logp)

    counts = {"num_sequences": mask.shape[0], "num_tokens": int(mask.sum())}
    accumulated = _accumulated(cfg, logp, old_logp, ref_logp, adv, mask, **counts)
    (acc_grad,) = torch.autograd.grad(accumulated, logp)

    assert accumulated.item() == pytest.approx(full.item(), abs=1e-12)
    torch.testing.assert_close(acc_grad, full_grad, atol=1e-12, rtol=0)


@pytest.mark.parametrize("normalization", ["sequence", "token", "constant"])
def test_gradient_accumulation_with_local_counts_is_wrong(normalization):
    """Averaging per-micro-batch losses is a mean of means, as in SFT."""
    cfg = PolicyLossConfig(normalization=normalization)
    logp, old_logp, ref_logp, adv, mask = _batch(seed=11, requires_grad=True)
    full = policy_gradient_loss(
        logp, old_logp, adv, mask, cfg, ref_logp=ref_logp, max_tokens=MAX_TOKENS
    ).loss

    mean_of_means = _accumulated(cfg, logp, old_logp, ref_logp, adv, mask) / len(MICRO_BATCHES)

    assert abs(mean_of_means.item() - full.item()) > 1e-3


def test_empty_response_counts_in_default_divisor_unless_num_sequences_is_given():
    per_token_loss = torch.tensor([[1.0, 3.0], [5.0, 7.0]], dtype=torch.float64)
    mask = torch.tensor([[True, True], [False, False]])

    assert aggregate_token_losses(per_token_loss, mask, "sequence").item() == 1.0
    assert aggregate_token_losses(per_token_loss, mask, "sequence", num_sequences=1).item() == 2.0
    with pytest.raises(ValueError, match="num_tokens must be positive"):
        aggregate_token_losses(per_token_loss, mask, "token", num_tokens=0)
