"""Toy RL with verifiable rewards: SFT, then GRPO-family RL on 2-digit addition.

A tiny GPT is first trained briefly with supervised fine-tuning on
``"ab+cd=" -> str(ab + cd)``, deliberately stopped early so it is only partly
correct. It is then trained with a group-relative policy-gradient objective
from ``posttrain.rl`` against an exact-match reward, with no access to the
reference answers except through that 0/1 reward.

The script writes a JSON report (``--out``, default ``results/toy_rlvr.json``)
with the per-step training reward, held-out greedy accuracy, seed, wall time
and hardware. Run::

    python examples/toy_rlvr.py --seed 0

One run took 23 to 28 s on the CPU described in the README, longer when the
machine is busy. This is a sanity check that the objectives produce a
learning signal end to end, not a benchmark.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import random
import sys
import time
import warnings
from pathlib import Path

# torch probes for NumPy at import time and warns if it is missing; nothing
# here needs NumPy, so keep the first line of output about the run itself.
warnings.filterwarnings("ignore", message="Failed to initialize NumPy", category=UserWarning)

import torch  # noqa: E402

from posttrain.models import TinyGPT, TinyGPTConfig  # noqa: E402
from posttrain.ops import token_logprobs  # noqa: E402
from posttrain.rl import (  # noqa: E402
    PolicyLossConfig,
    group_relative_advantages,
    policy_gradient_loss,
)
from posttrain.sft import causal_lm_loss  # noqa: E402

DIGITS = "0123456789"
PLUS, EQUALS, EOS, PAD = 10, 11, 12, 13
VOCAB = 14
PROMPT_LEN = 6  # "ab+cd=" with zero-padded operands
MAX_NEW_TOKENS = 4  # up to three answer digits plus EOS

RECIPES = {
    "grpo": PolicyLossConfig.grpo,
    "dapo": PolicyLossConfig.dapo,
    "dr_grpo": PolicyLossConfig.dr_grpo,
}


def encode_prompt(a: int, b: int) -> list[int]:
    return [*map(int, f"{a:02d}"), PLUS, *map(int, f"{b:02d}"), EQUALS]


def encode_answer(a: int, b: int) -> list[int]:
    return [*map(int, str(a + b)), EOS]


def decode_answer(tokens: list[int]) -> str | None:
    """Digits before the first EOS, or None if the response never terminates."""
    if EOS not in tokens:
        return None
    body = tokens[: tokens.index(EOS)]
    if not body or any(t > 9 for t in body):
        return None
    return "".join(DIGITS[t] for t in body)


def split_problems(seed: int, num_eval: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    problems = [(a, b) for a in range(100) for b in range(100)]
    random.Random(seed).shuffle(problems)
    return problems[num_eval:], problems[:num_eval]


def sft_batch(problems: list[tuple[int, int]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-padded prompt+answer rows with loss only on answer tokens."""
    width = PROMPT_LEN + MAX_NEW_TOKENS
    ids = torch.full((len(problems), width), PAD)
    labels = torch.full((len(problems), width), -100)
    for row, (a, b) in enumerate(problems):
        seq = encode_prompt(a, b) + encode_answer(a, b)
        ids[row, : len(seq)] = torch.tensor(seq)
        # Position t predicts token t+1; targets are the answer tokens only.
        for t in range(PROMPT_LEN - 1, len(seq) - 1):
            labels[row, t] = seq[t + 1]
    return ids, labels


@torch.no_grad()
def generate(
    model: TinyGPT, prompts: torch.Tensor, *, greedy: bool, gen: torch.Generator | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample responses; returns (tokens, log-probs, mask) of shape [B, MAX_NEW_TOKENS].

    The mask is true up to and including the first EOS, so tokens generated
    after the model has finished never enter the loss.
    """
    seq = prompts
    logps = []
    for _ in range(MAX_NEW_TOKENS):
        logits = model(seq)[:, -1].float()
        if greedy:
            nxt = logits.argmax(-1)
        else:
            nxt = torch.multinomial(torch.softmax(logits, -1), 1, generator=gen).squeeze(-1)
        logps.append(torch.log_softmax(logits, -1).gather(-1, nxt[:, None]).squeeze(-1))
        seq = torch.cat([seq, nxt[:, None]], dim=1)
    tokens = seq[:, PROMPT_LEN:]
    finished_before = torch.cumsum((tokens == EOS).int(), dim=1) - (tokens == EOS).int()
    mask = finished_before == 0
    return tokens, torch.stack(logps, dim=1), mask


def rewards_for(tokens: torch.Tensor, problems: list[tuple[int, int]]) -> torch.Tensor:
    out = [
        float(decode_answer(row) == str(a + b))
        for row, (a, b) in zip(tokens.tolist(), problems, strict=True)
    ]
    return torch.tensor(out)


def response_logprobs(model: TinyGPT, prompts: torch.Tensor, tokens: torch.Tensor, mask):
    seq = torch.cat([prompts, tokens], dim=1)
    logits = model(seq)[:, PROMPT_LEN - 1 : -1]
    return token_logprobs(logits, tokens, mask)


@torch.no_grad()
def greedy_accuracy(model: TinyGPT, problems: list[tuple[int, int]]) -> float:
    prompts = torch.tensor([encode_prompt(a, b) for a, b in problems])
    tokens, _, _ = generate(model, prompts, greedy=True)
    return rewards_for(tokens, problems).mean().item()


def hardware() -> dict[str, object]:
    return {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "logical_cpus": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": "cpu",
    }


def main(argv: list[str] | None = None) -> dict[str, object]:
    """Run SFT then RL, write the JSON report to ``--out`` and return it."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--recipe", choices=sorted(RECIPES), default="dr_grpo")
    parser.add_argument("--sft-steps", type=int, default=400)
    parser.add_argument("--sft-batch", type=int, default=64)
    parser.add_argument("--sft-lr", type=float, default=3e-3)
    parser.add_argument("--rl-steps", type=int, default=150)
    parser.add_argument("--prompts-per-step", type=int, default=32)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--updates-per-batch", type=int, default=2)
    parser.add_argument("--rl-lr", type=float, default=3e-4)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--num-eval", type=int, default=500)
    parser.add_argument("--out", type=Path, default=Path("results/toy_rlvr.json"))
    args = parser.parse_args(argv)
    command = sys.argv if argv is None else ["examples/toy_rlvr.py", *argv]

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    train, held_out = split_problems(args.seed, args.num_eval)
    cfg = TinyGPTConfig(vocab_size=VOCAB, d_model=64, n_layers=2, n_heads=4, max_positions=16)
    model = TinyGPT(cfg)
    start = time.perf_counter()

    # Stage 1: brief SFT, stopped early on purpose.
    opt = torch.optim.AdamW(model.parameters(), lr=args.sft_lr, weight_decay=0.0)
    sft_losses = []
    for _ in range(args.sft_steps):
        ids, labels = sft_batch(rng.sample(train, args.sft_batch))
        loss = causal_lm_loss(model(ids), labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sft_losses.append(loss.item())
    sft_time = time.perf_counter() - start
    sft_accuracy = greedy_accuracy(model, held_out)
    print(
        f"SFT: {args.sft_steps} steps, loss {sft_losses[-1]:.3f}, held-out acc {sft_accuracy:.3f}"
    )

    # Stage 2: RL against the exact-match reward.
    loss_cfg = RECIPES[args.recipe]()
    reference = copy.deepcopy(model).requires_grad_(False) if loss_cfg.beta > 0 else None
    opt = torch.optim.AdamW(model.parameters(), lr=args.rl_lr, weight_decay=0.0)
    steps, evals = [], [{"step": 0, "held_out_accuracy": sft_accuracy}]
    rl_start = time.perf_counter()
    for step in range(1, args.rl_steps + 1):
        problems = rng.sample(train, args.prompts_per_step)
        group = [p for p in problems for _ in range(args.group_size)]
        prompts = torch.tensor([encode_prompt(a, b) for a, b in group])
        tokens, old_logp, mask = generate(model, prompts, greedy=False, gen=gen)
        rewards = rewards_for(tokens, group)
        advantages = group_relative_advantages(
            rewards, args.group_size, scale_by_std=loss_cfg.scale_advantages_by_std
        )
        ref_logp = None
        if reference is not None:
            with torch.no_grad():
                ref_logp = response_logprobs(reference, prompts, tokens, mask)
        for _ in range(args.updates_per_batch):
            out = policy_gradient_loss(
                response_logprobs(model, prompts, tokens, mask),
                old_logp,
                advantages,
                mask,
                loss_cfg,
                ref_logp=ref_logp,
                max_tokens=MAX_NEW_TOKENS,
            )
            opt.zero_grad()
            out.loss.backward()
            opt.step()
        steps.append(
            {
                "step": step,
                "mean_reward": rewards.mean().item(),
                "clip_fraction": out.clip_fraction.item(),
                "mean_response_tokens": mask.sum(-1).float().mean().item(),
            }
        )
        if step % args.eval_every == 0 or step == args.rl_steps:
            acc = greedy_accuracy(model, held_out)
            evals.append({"step": step, "held_out_accuracy": acc})
            print(f"RL step {step}: train reward {rewards.mean():.3f}, held-out acc {acc:.3f}")
    rl_time = time.perf_counter() - rl_start

    result = {
        "command": " ".join(["python", *command]),
        "seed": args.seed,
        "args": {k: v.as_posix() if isinstance(v, Path) else v for k, v in vars(args).items()},
        "model": vars(cfg),
        "loss_config": vars(loss_cfg),
        "sft": {
            "steps": args.sft_steps,
            "final_loss": sft_losses[-1],
            "held_out_accuracy": sft_accuracy,
            "wall_time_s": round(sft_time, 2),
        },
        "rl": {"steps": steps, "eval": evals, "wall_time_s": round(rl_time, 2)},
        "total_wall_time_s": round(time.perf_counter() - start, 2),
        "hardware": hardware(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out} in {result['total_wall_time_s']} s")
    return result


if __name__ == "__main__":
    main()
