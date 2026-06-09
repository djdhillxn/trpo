# RLHF Evaluation History

This note preserves the two full-validation policy-suite evaluations and explains why the 1024-token run is the primary result while the 512-token run remains a useful baseline.

Both evaluations use the same 2017 HelpSteer3 validation prompts and compare:

- Base: `Qwen/Qwen2.5-0.5B-Instruct`
- SFT: the 4096-token supervised LoRA checkpoint
- PPO: update 400 from the 4096-token, epoch-2 reward-model run

All responses are scored by the same learned reward model. These scores are proxy judgments, not human preference labels.

## Primary evaluation: 1024 generated tokens

The current evaluation allows a 3072-token prompt and up to 1024 generated tokens. The maximum prompt-plus-response sequence is therefore 4096 tokens, matching the sequence budget used for SFT and reward-model training.

Output directory:

```text
outputs/rlhf/qwen25_05b_helpsteer3_eval_suite_4096_ep2_u400_eval1024/
```

### Three-way results

| Policy | Wins | Win rate | Mean reward | Median response tokens | Cap-hit rate | Empty rate |
|---|---:|---:|---:|---:|---:|---:|
| Base | 978 | 48.49% | -3.3634 | 334 | 8.08% | 0.00% |
| SFT | 475 | 23.55% | -3.6114 | 360 | 10.16% | 0.00% |
| PPO | 467 | 23.15% | -3.5771 | 363 | 11.60% | 0.00% |
| Tie | 97 | 4.81% | — | — | — | — |

### Pairwise results

| Comparison | Left wins | Right wins | Ties | Right win rate | Mean right-left delta |
|---|---:|---:|---:|---:|---:|
| Base vs SFT | 1215 | 763 | 39 | 37.83% | -0.2480 |
| Base vs PPO | 1190 | 785 | 42 | 38.92% | -0.2137 |
| SFT vs PPO | 963 | 892 | 162 | 44.22% | +0.0343 |

PPO loses to Base more often than it wins, but its winning margins are somewhat larger on average. Across all prompts, the mean positive PPO-minus-Base margin is `+1.6210`, while the mean negative margin is `-1.4315`. The larger number of losses, 1190 versus 785 wins, keeps the aggregate mean negative.

Against SFT, PPO has a slightly positive mean delta of `+0.0343` even though it wins fewer prompts. Its average winning margin is `+1.3503`, compared with an average losing margin of `-1.1789`. This is evidence of asymmetric margins, not evidence that PPO is generally superior to SFT.

### Domain-level PPO win rates against Base

| Domain | PPO wins | Base wins | Ties | PPO win rate |
|---|---:|---:|---:|---:|
| Code | 115 | 321 | 2 | 26.26% |
| General | 414 | 487 | 30 | 44.47% |
| STEM | 97 | 146 | 2 | 39.59% |
| Multilingual | 159 | 236 | 8 | 39.45% |

The largest weakness is code. General prompts are the closest comparison, although Base still wins more often.

## Archived baseline: 512 generated tokens

The earlier suite used the same 3072-token prompt cap but allowed only 512 generated tokens.

Output directory:

```text
outputs/rlhf/qwen25_05b_helpsteer3_eval_suite_4096_ep2_u400/
```

### Three-way results

| Policy | Wins | Win rate | Mean reward | Median response tokens | Cap-hit rate |
|---|---:|---:|---:|---:|---:|
| Base | 827 | 41.00% | -3.5339 | 332 | 29.90% |
| SFT | 556 | 27.57% | -3.4280 | 363 | 29.95% |
| PPO | 525 | 26.03% | -3.6666 | 356 | 29.70% |
| Tie | 109 | 5.40% | — | — | — |

### Pairwise results

| Comparison | Left wins | Right wins | Ties | Right win rate | Mean right-left delta |
|---|---:|---:|---:|---:|---:|
| Base vs SFT | 1044 | 927 | 46 | 45.96% | +0.1059 |
| Base vs PPO | 1068 | 904 | 45 | 44.82% | -0.1327 |
| SFT vs PPO | 965 | 898 | 154 | 44.52% | -0.2386 |

## What changed

The 1024-token run substantially reduced cap hits:

| Policy | 512-token cap hits | 1024-token cap hits | Absolute reduction |
|---|---:|---:|---:|
| Base | 603 | 163 | 440 |
| SFT | 604 | 205 | 399 |
| PPO | 599 | 234 | 365 |

This is a clear operational improvement: fewer responses end only because the evaluator exhausted its generation allowance.

The longer run also exposed more repetition. Using the fraction of repeated word-level 4-grams as a simple diagnostic, PPO produced 325 responses above a 25% repetition threshold in the 1024-token run, compared with 224 in the 512-token run. Severe repetition above 50% rose from 74 to 156 PPO responses. Longer inference therefore reveals both useful continuations and failures that the shorter cap previously cut off.

## Comparison caveat

The two runs are not a perfectly controlled token-budget ablation. The 512-token run used evaluation batch size 8, while the 1024-token run used batch size 128. Most responses changed, including many that had not reached the old cap. Batched bfloat16 generation can cross close logit boundaries, and evaluator changes or environment differences may also affect exact greedy continuations.

The 1024-token suite is still the primary evaluation because it is the latest complete run and has much lower truncation. However, differences between the two suites should be described as run-to-run changes associated with the longer evaluation configuration, not attributed exclusively to `max_new_tokens`.

## Associated artifacts

- [`rlhf_qualitative_audit.md`](rlhf_qualitative_audit.md): manual interpretation of wins, failures, repetition, and reward-model mismatches.
- [`rlhf_future_work.md`](rlhf_future_work.md): research directions motivated by these results.
- `outputs/rlhf/qwen25_05b_helpsteer3_eval_suite_4096_ep2_u400_eval1024/qualitative_audit_auto.md`: automated full-suite audit.
- `outputs/rlhf/qwen25_05b_helpsteer3_eval_suite_4096_ep2_u400_eval1024/selected_qualitative_examples.md`: selected prompts and full Base/SFT/PPO responses.
