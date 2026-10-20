# post-training-from-scratch

[![ci](https://github.com/GonzaloFuentes1/post-training-from-scratch/actions/workflows/ci.yml/badge.svg)](https://github.com/GonzaloFuentes1/post-training-from-scratch/actions/workflows/ci.yml)

From-scratch PyTorch implementations of the objectives used to post-train
large language models: SFT with sequence packing and loss masking, offline
preference optimization (DPO, conservative DPO, IPO, SimPO), and
group-relative RL (GRPO, DAPO, Dr.GRPO). Each objective is a small function
on plain tensors with a docstring that states it and explains the design
choices, and each is pinned down by invariant tests: parity against
independent loop-based references, float64 gradchecks, analytic gradients,
Monte Carlo bias checks, and end-to-end equivalences on a tiny transformer.
The aim is a reference that is easy to read and hard to get subtly wrong.

## What it is and what it is not

It is:

- a set of reference implementations of loss functions and the data
  transformations they depend on (packing, attention masks, loss masks,
  advantages, KL estimators);
- a test suite that states the properties those objectives must satisfy and
  checks them on CPU in under a minute;
- a toy end-to-end example (SFT warm start, then RL with a verifiable reward)
  that shows the RL losses produce a learning signal; CI runs it with tiny
  settings so it cannot silently break.

It is not a trainer. There is no distributed training, no inference engine
for rollouts, no checkpointing and no model zoo. For production post-training
use [TRL](https://github.com/huggingface/trl),
[OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) or
[verl](https://github.com/volcengine/verl). DAPO's dynamic sampling and
overlong reward shaping are data-pipeline techniques and are not implemented
here; only its loss (token-level normalization and clip-higher) is.

## Quickstart

Requires Python 3.11+.

```bash
git clone https://github.com/GonzaloFuentes1/post-training-from-scratch.git
cd post-training-from-scratch
uv venv --python 3.12
source .venv/bin/activate   # Windows (Git Bash): source .venv/Scripts/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -e ".[dev]"
pytest
```

```python
import torch
from posttrain.rl import PolicyLossConfig, group_relative_advantages, policy_gradient_loss

rewards = torch.tensor([1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0])  # 2 prompts x 4 samples
cfg = PolicyLossConfig.dr_grpo()  # or .dapo(); .grpo() also needs ref_logp=...
adv = group_relative_advantages(rewards, group_size=4, scale_by_std=cfg.scale_advantages_by_std)

# Stand-ins for model outputs: [8, T] log-probs of the sampled tokens under the
# sampling policy (old_logp) and the current policy (logp), and a [8, T] mask
# that is True on response tokens.
T = 6
old_logp = -torch.rand(8, T)
logp = (old_logp + 0.1 * torch.randn(8, T)).requires_grad_()
mask = torch.ones(8, T, dtype=torch.bool)
mask[:4, 4:] = False  # the first prompt's responses are shorter

out = policy_gradient_loss(logp, old_logp, adv, mask, cfg, max_tokens=T)
out.loss.backward()
print(out.loss.item(), out.clip_fraction.item())
```

`tests/test_readme.py` executes this snippet, so it cannot drift from the API.

## Layout

```
src/posttrain/
  ops.py               masked sum / mean, per-token log-probs
  sft/
    packing.py         first-fit-decreasing packing, labels that never cross documents
    attention.py       block-diagonal causal mask, per-document position ids
    chat.py            ChatML-like template with assistant-only loss masks
    loss.py            token-weighted cross-entropy, exact under gradient accumulation
  preference/
    logprobs.py        summed / averaged response log-likelihoods
    losses.py          DPO, conservative DPO, IPO, SimPO
  rl/
    advantages.py      group-relative advantages
    kl.py              Schulman's k1, k2, k3 estimators
    losses.py          clipped policy-gradient loss: GRPO, DAPO, Dr.GRPO
  models/tiny_gpt.py   small transformer used by tests and the example
examples/toy_rlvr.py   SFT warm start + RL on 2-digit addition
```

## Objectives

### SFT

Next-token cross-entropy over target tokens only. With gradient accumulation
over micro-batches $b$ the loss is normalized by the number of target tokens
in the whole batch, not per micro-batch:

$$\mathcal{L}_{\text{SFT}} = \frac{1}{N_{\text{global}}} \sum_{b} \sum_{t \in b} -\log \pi_\theta(x_t \mid x_{<t}).$$

Averaging per-micro-batch means instead weights a token by the inverse of
the size of the micro-batch it happened to land in. Packed rows use a
block-diagonal causal mask and position ids that restart at 0 for every
document, which makes a packed forward pass identical to running the
documents separately. In chat data only the assistant content and its
`<|im_end|>` marker are targets; the end marker is how the model learns to
stop.

### Preference optimization

With chosen $y_w$, rejected $y_l$ and the margin
$m = \log\frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)} - \log\frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)}$:

| objective | loss | note |
|---|---|---|
| DPO | $-\log\sigma(\beta m)$ | $\partial\mathcal{L}/\partial m = -\beta\,\sigma(-\beta m)$, never zero |
| conservative DPO (cDPO) | $-(1-\epsilon)\log\sigma(\beta m) - \epsilon\log\sigma(-\beta m)$ | stationary at $m^* = \beta^{-1}\log\frac{1-\epsilon}{\epsilon}$ |
| IPO | $\left(m - \frac{1}{2\tau}\right)^2$ | regresses the margin to a finite target |
| SimPO | $-\log\sigma\left(\frac{\beta}{\lvert y_w\rvert}\log\pi_\theta(y_w) - \frac{\beta}{\lvert y_l\rvert}\log\pi_\theta(y_l) - \gamma\right)$ | no reference model; length-normalized |

DPO is the Bradley-Terry likelihood under the implicit reward
$r(y) = \beta \log \pi_\theta(y)/\pi_{\text{ref}}(y)$. Because its gradient
never vanishes, on deterministic preferences it keeps pushing the margin
up; conservative DPO and IPO give the margin a finite optimum.

### Group-relative RL

For $G$ responses $o_i$ to a prompt with rewards $r_i$:

$$A_i = \frac{r_i - \operatorname{mean}(r)}{\operatorname{std}(r) + \varepsilon}
\quad\text{(GRPO)}, \qquad A_i = r_i - \operatorname{mean}(r) \quad\text{(Dr.GRPO)},$$

$$\ell_{i,t} = -\min\left(\rho_{i,t} A_i,\ \operatorname{clip}(\rho_{i,t}, 1-\epsilon_{\text{low}}, 1+\epsilon_{\text{high}})\, A_i\right) + \beta\, \widehat{\mathrm{KL}}_{i,t},
\qquad \rho_{i,t} = \frac{\pi_\theta(o_{i,t}\mid q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}\mid q, o_{i,<t})}.$$

The recipes differ in how $\ell_{i,t}$ is reduced, how advantages are
scaled and how ratios are clipped:

| | GRPO | DAPO | Dr.GRPO |
|---|---|---|---|
| reduction | $\frac{1}{G}\sum_i \frac{1}{\lvert o_i\rvert}\sum_t \ell_{i,t}$ | $\frac{1}{\sum_i \lvert o_i\rvert}\sum_i\sum_t \ell_{i,t}$ | $\frac{1}{G\,L_{\max}}\sum_i\sum_t \ell_{i,t}$ |
| weight on $\nabla\log\pi_\theta(o_{i,t})$ in the objective at $\rho = 1$ | $A_i / (G\lvert o_i\rvert)$, depends on the response length | $A_i / \sum_j \lvert o_j\rvert$, length-independent; one normalizer shared by all tokens in the batch | $A_i / (G L_{\max})$, length-independent; fixed normalizer |
| advantage | mean-centered, divided by std | mean-centered, divided by std | mean-centered only |
| clipping $(\epsilon_{\text{low}}, \epsilon_{\text{high}})$ | symmetric (preset 0.2, 0.2) | asymmetric "clip-higher" (0.2, 0.28) | symmetric (preset 0.2, 0.2) |
| KL penalty | $\beta = 0.04$, k3 estimator | none | none |
| preset | `PolicyLossConfig.grpo()` | `PolicyLossConfig.dapo()` | `PolicyLossConfig.dr_grpo()` |

The weights are those of the objective that is maximized. The loss is its
negative, so the gradient of the loss with respect to a token's log-prob is
minus that weight; the verification table below quotes the loss gradient.

Under the original sequence-level reduction a short correct answer is
reinforced more per token than a long one, and a long wrong answer is
penalized less per token than a short one, which biases the policy towards
longer incorrect responses. Dividing by the group std additionally
up-weights prompts whose rewards barely vary, relative to mean-centering
alone. Dr.GRPO removes both terms.
Clip-higher lets low-probability tokens with positive advantage grow further
before clipping, which DAPO uses against entropy collapse.

The KL penalty uses Schulman's estimators with $x \sim \pi_\theta$ and
$r = \pi_{\text{ref}}(x)/\pi_\theta(x)$: $k_1 = -\log r$ (unbiased, high
variance, often negative), $k_2 = \frac{1}{2}(\log r)^2$ (biased, low
variance) and $k_3 = (r - 1) - \log r$ (unbiased, non-negative, low
variance). Those are properties of the *value*. In the loss the estimate is
differentiated with the samples held fixed, and on-policy the expected
gradients rank the estimators differently:

| estimator | value, as an estimate of $\mathrm{KL}(\pi_\theta \Vert \pi_{\text{ref}})$ | expected gradient as a loss term |
|---|---|---|
| $k_1$ | unbiased | $0$: zero-mean noise, no regularization |
| $k_2$ | biased | $\nabla\mathrm{KL}(\pi_\theta \Vert \pi_{\text{ref}})$, unbiased |
| $k_3$ | unbiased | $\nabla\mathrm{KL}(\pi_{\text{ref}} \Vert \pi_\theta)$, the other direction |

GRPO's $k_3$ penalty therefore still pulls the policy towards the reference,
but through the gradient of $\mathrm{KL}(\pi_{\text{ref}} \Vert \pi_\theta)$,
not of the KL its value estimates. `PolicyLossConfig.grpo()` keeps $k_3$ to
stay faithful to DeepSeekMath. The derivation is in the docstring of
`posttrain/rl/kl.py`; Tang and Munos (2025) analyze these pitfalls in
detail, including the sequence-level case.

## Design notes

- **Plain tensors in, unreduced or scalar tensors out.** No function knows
  about models, tokenizers or optimizers, so each one can be tested on
  hand-built inputs. Preference losses return one value per pair and leave
  the batch reduction to the caller.
- **The reduction is part of the objective.** Normalization is an explicit
  argument (`normalization`, `max_tokens`, and the global counts
  `num_target_tokens` for SFT and `num_tokens` / `num_sequences` for RL)
  rather than a hidden `.mean()`, because that denominator is where GRPO,
  DAPO and Dr.GRPO differ and where gradient accumulation goes wrong. Under
  accumulation every micro-batch is divided by the count of the whole
  optimizer batch and the micro-batch losses are summed.
- **Masks select, they do not multiply.** Masked positions are removed with
  `torch.where`, so `-inf` or NaN padding in a tensor that is passed in
  cannot leak into the value or into that tensor's gradient. `where` does
  not cut the graph, though: if the op that *produced* the padding has a
  non-finite derivative there (`log(0)`, `logsumexp` over a row of `-inf`),
  its backward still computes `0 * inf = NaN`. Such values have to be
  replaced before that op, which `token_logprobs` does for masked logit
  rows. Tests put `-inf` into masked log-probs of the RL loss; `-inf`, `inf`
  and NaN into masked logit rows of `token_logprobs`; `-inf` and NaN into
  those of `sequence_logprobs`; and `-inf` into masked logit rows on the
  logits-to-RL-loss path. Each requires finite gradients equal to the clean
  ones.
- **Every docstring explains why.** For example why the last token of a
  packed document has no target, why padded queries attend to themselves,
  and why `logsumexp` is used to gather log-probs.
- **A model only where the invariant needs one.** The tiny transformer in
  `posttrain.models` exists so that packing and gradient-accumulation
  equivalences are tested through real attention, in float64.

## How it is verified

`pytest` runs 152 tests on CPU: 12 to 53 s across local runs on the machine
described under Results, depending on other load. CI runs ruff and the suite
on Python 3.11 and 3.12. Highlights:

| property | test |
|---|---|
| GRPO/DAPO/Dr.GRPO losses equal a loop-based transcription of the paper equations, 5 configurations x 3 seeds | `tests/rl/test_losses.py` |
| float64 `gradcheck` of every RL and preference loss | `tests/rl/test_losses.py`, `tests/preference/test_losses.py` |
| per-token loss gradient $\partial\mathcal{L}/\partial\log\pi_\theta(o_{i,t})$ at $\rho = 1$ is $-A_i/(G\lvert o_i\rvert)$ under sequence normalization (3x larger for a 2-token than a 6-token answer) and length-independent under Dr.GRPO and DAPO | `tests/rl/test_losses.py` |
| clip-higher keeps the gradient at ratio 1.25 with $A>0$; clipping never removes a corrective gradient | `tests/rl/test_losses.py` |
| prompt and pad tokens, even when set to `-inf`, change neither loss nor gradient | `tests/rl/test_losses.py` |
| advantages sum to zero per group, are invariant to per-group reward offsets, and constant groups give 0, not NaN | `tests/rl/test_advantages.py` |
| k1 and k3 are unbiased and k2 is biased, by Monte Carlo (400k samples) against the exact KL between categorical distributions | `tests/rl/test_kl.py` |
| used as loss terms, the exact expected gradients of k1, k2 and k3 are $0$, $\nabla\mathrm{KL}(\pi_\theta \Vert \pi_{\text{ref}})$ and $\nabla\mathrm{KL}(\pi_{\text{ref}} \Vert \pi_\theta)$ on categorical distributions | `tests/rl/test_kl.py` |
| DPO gradient w.r.t. the policy log-ratio equals $-\beta\sigma(-\beta m)$ with the correct sign for chosen, rejected and reference terms | `tests/preference/test_losses.py` |
| conservative DPO is stationary at $m^*$; IPO penalizes overshooting its target; SimPO equals DPO on average log-probs for any reference that cannot tell the pair apart, and is invariant to repeating a response | `tests/preference/test_losses.py` |
| packed loss and all parameter gradients of a float64 transformer equal the unpacked sum (atol 1e-10); a plain causal mask or no position reset breaks this | `tests/sft/test_packed_equivalence.py` |
| token-weighted gradient accumulation equals the full batch (atol 1e-12); the naive mean of micro-batch means is off by more than 10% | `tests/sft/test_grad_accumulation.py` |
| RL losses accumulated over uneven micro-batches with global `num_tokens` / `num_sequences` equal the full-batch loss and gradient (atol 1e-12) for every normalization; averaging micro-batch losses does not | `tests/rl/test_losses.py` |
| first-fit-decreasing stays within $\frac{11}{9}\mathrm{OPT} + \frac{6}{9}$ bins against an exhaustive optimum | `tests/sft/test_packing.py` |
| chat loss masks select exactly the assistant content and end markers | `tests/sft/test_chat.py` |

## Results: toy RL with a verifiable reward

`examples/toy_rlvr.py` trains a 2-layer, 64-wide GPT on 2-digit addition
(`"ab+cd="` to the sum). A short SFT warm start (400 steps) leaves the model
only partially correct, then 150 Dr.GRPO steps (32 prompts x 8 samples, 2
updates per batch) optimize an exact-match 0/1 reward. Accuracy is greedy
decoding on 500 held-out problems. The seed sets the initialization, the
train/held-out split and the sampling. Five seeds, each also compared with
plain SFT for 800 steps:

```bash
for seed in 0 1 2 3 4; do
  python examples/toy_rlvr.py --seed $seed --out results/toy_rlvr_seed$seed.json
  python examples/toy_rlvr.py --seed $seed --sft-steps 800 --rl-steps 0 \
    --out results/toy_sft_only_800_seed$seed.json
done
```

| seed | SFT 400 steps | SFT 400 + RL 150 steps | change from RL | SFT 800 steps |
|---|---|---|---|---|
| 0 | 0.476 | 0.584 | +0.108 | 0.934 |
| 1 | 0.770 | 0.912 | +0.142 | 0.924 |
| 2 | 0.764 | 0.948 | +0.184 | 0.998 |
| 3 | 0.938 | 0.994 | +0.056 | 0.994 |
| 4 | 0.810 | 0.946 | +0.136 | 0.998 |
| mean (min to max) | 0.752 (0.476 to 0.938) | 0.877 (0.584 to 0.994) | +0.125 (+0.056 to +0.184) | 0.970 (0.924 to 0.998) |

RL improved held-out accuracy on every seed. Mean training reward (sampled
at temperature 1) rose from 0.625 over the first 25 RL steps to 0.783 over
the last 25, averaged over seeds (seed 0: 0.310 to 0.476). The warm start
varies a lot between seeds (0.476 to 0.938), so a single seed says little
about the size of the effect. Per-step data, commands and hardware are in
`results/toy_rlvr_seed*.json` and `results/toy_sft_only_800_seed*.json`.

Hardware: Windows 11, Intel CPU with 16 logical cores, CPU only, torch
2.14.1+cpu with 10 threads, Python 3.12. One SFT + RL run took 22.7 to
27.6 s and one 800-step SFT run 18.0 to 22.0 s. Wall time depends on other
load on the machine: an earlier seed-0 run, made while other jobs were
running, took 153 s with identical accuracies. Three runs of seed 0 on this
machine gave identical accuracies and rewards.

For perspective, 400 more SFT steps (25,600 labeled examples) matched or
beat 150 RL steps (38,400 sampled responses) on every seed. When labels are
available, SFT is the more sample-efficient signal on this task. The example
demonstrates that the RL objectives produce a learning signal end to end; it
is not evidence about RL at scale.

## References

- Rafailov et al., *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*, 2023. [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)
- Mitchell, *A note on DPO with noisy preferences & relationship to IPO* (conservative DPO), 2023. [ericmitchell.ai/cdpo.pdf](https://ericmitchell.ai/cdpo.pdf)
- Azar et al., *A General Theoretical Paradigm to Understand Learning from Human Preferences* (IPO), 2023. [arXiv:2310.12036](https://arxiv.org/abs/2310.12036)
- Meng, Xia and Chen, *SimPO: Simple Preference Optimization with a Reference-Free Reward*, 2024. [arXiv:2405.14734](https://arxiv.org/abs/2405.14734)
- Shao et al., *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models* (GRPO), 2024. [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)
- Yu et al., *DAPO: An Open-Source LLM Reinforcement Learning System at Scale*, 2025. [arXiv:2503.14476](https://arxiv.org/abs/2503.14476)
- Liu et al., *Understanding R1-Zero-Like Training: A Critical Perspective* (Dr.GRPO), 2025. [arXiv:2503.20783](https://arxiv.org/abs/2503.20783)
- Schulman et al., *Proximal Policy Optimization Algorithms*, 2017. [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
- Schulman, *Approximating KL Divergence*, 2020. [joschu.net/blog/kl-approx.html](http://joschu.net/blog/kl-approx.html)
- Tang and Munos, *On a few pitfalls in KL divergence gradient estimation for RL*, 2025. [arXiv:2506.09477](https://arxiv.org/abs/2506.09477)
- Krell et al., *Efficient Sequence Packing without Cross-contamination: Accelerating Large Language Models without Impacting Performance*, 2021. [arXiv:2107.02027](https://arxiv.org/abs/2107.02027)
- Dósa, *The Tight Bound of First Fit Decreasing Bin-Packing Algorithm Is FFD(I) <= 11/9 OPT(I) + 6/9*, 2007. [doi:10.1007/978-3-540-74450-4_1](https://doi.org/10.1007/978-3-540-74450-4_1)

## License and citation

MIT, see [LICENSE](LICENSE). Citation metadata is in
[CITATION.cff](CITATION.cff).
