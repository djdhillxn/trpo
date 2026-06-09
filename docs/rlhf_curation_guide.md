# RLHF Curation Guide

The final aggregate PPO win rate is mixed, so the portfolio should not rely on a single headline number. Instead, use the final policy-suite outputs to show both wins and failures.

Main file:

```text
outputs/rlhf/qwen25_05b_helpsteer3_eval_suite_4096_ep2_u400/policy_suite_samples.csv
```

This file contains the full prompt, Base response, SFT response, PPO response, reward scores, response lengths, cap-hit flags, and pairwise winners for all 2017 validation prompts.

## Recommended example categories

Pick 8-12 examples across these buckets:

1. **Clear PPO wins**: PPO reward higher than both Base and SFT, and the response is visibly more direct/useful.
2. **SFT wins**: shows that supervised alignment did meaningful work.
3. **Base wins**: honest failure cases where the original instruction model remains stronger.
4. **Ties / near ties**: cases where all policies produce nearly identical outputs.
5. **Bad PPO examples**: one or two examples where PPO rambles, repeats, or worsens the answer.

This makes the portfolio more credible than showing only cherry-picked successes.

## Useful filters

```python
import pandas as pd

df = pd.read_csv("outputs/rlhf/qwen25_05b_helpsteer3_eval_suite_4096_ep2_u400/policy_suite_samples.csv")

# Strong PPO wins over both Base and SFT
ppo_wins = df[
    (df["winner"] == "ppo_4096_ep2_u400") &
    (df["delta_ppo_4096_ep2_u400_minus_base"] > 2.0) &
    (df["delta_ppo_4096_ep2_u400_minus_sft_4096"] > 1.0)
].sort_values("delta_ppo_4096_ep2_u400_minus_base", ascending=False)

# PPO loses badly to Base
ppo_losses = df[
    (df["winner_base_vs_ppo_4096_ep2_u400"] == "base") &
    (df["delta_ppo_4096_ep2_u400_minus_base"] < -5.0)
].sort_values("delta_ppo_4096_ep2_u400_minus_base")

# Cases where SFT improves over Base
sft_wins = df[
    (df["winner_base_vs_sft_4096"] == "sft_4096") &
    (df["delta_sft_4096_minus_base"] > 2.0)
].sort_values("delta_sft_4096_minus_base", ascending=False)
```

## Initial candidate indices

These are not final portfolio selections; they are starting points for manual review.

| Index | Why inspect it |
|---:|---|
| 0 | code prompt; PPO/SFT produce cleaner React answer than truncated Base |
| 9 | Japanese menu prompt; PPO is more directly responsive than Base/SFT |
| 799 | strong PPO reward-model win; TypeScript/code example |
| 966 | Python/code example where PPO beats both Base and SFT by reward model |
| 1040 | multilingual example where PPO beats both Base and SFT |
| 1496 | general example where PPO gives much longer answer than short/refusal-like alternatives |
| 1485 | strong PPO failure; useful as honest negative example |
| 11 | SFT/PPO degenerate repetition; useful to discuss failure modes |
| 353 | PPO produces unsafe/strange continuation; useful as a failure case |

Use the curation notebook to inspect full text before publishing any example. Some high-reward examples may contain factual errors or may only win because the reward model prefers length/formatting.

## Portfolio framing

Recommended wording:

> The final PPO policy did not dominate the base instruction model in aggregate, but the project produced a real RLHF pipeline and a set of instructive qualitative examples. The most valuable part of the work was debugging the alignment stack: long-context SFT/RM coverage, reward-model accuracy, KL-controlled PPO, checkpoint validation, and full-suite evaluation.

This is more credible than claiming universal improvement.
