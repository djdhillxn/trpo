# RLHF Experiment Log

This file records the main RLHF experiments, including the failures that shaped the final configuration.

## Phase 0: Goal

The original repository implemented TRPO, NPG, and PPO for classical RL benchmarks. The RLHF extension was meant to translate that PPO experience into LLM post-training:

```text
Qwen2.5-0.5B-Instruct
        ↓ SFT on HelpSteer3 preferred responses
SFT policy / frozen reference
        ↓ reward model from HelpSteer3 preference pairs
scalar reward model
        ↓ token-level PPO with KL penalty
PPO-aligned policy adapter
        ↓ suite evaluation
Base vs SFT vs PPO responses
```

## Phase 1: Short-context smoke runs

The early runs used short response caps such as 96 or 128 generated tokens. This made debugging faster but created two problems:

1. many generated answers were visibly truncated;
2. code and explanatory responses often ended halfway, making qualitative comparison difficult.

The early PPO experiments also exposed several RLHF failure modes:

- gibberish/multilingual drift,
- vulgar/debug-pattern generations,
- EOS or blank-response collapse,
- overzealous reward shaping,
- evaluation bugs caused by wrong checkpoint paths.

These failures led to stronger prompt sanitation, KL anchoring, empty-response checks, anti-repetition checks, and checkpoint validation.

## Phase 2: Token-length diagnostic

A length diagnostic over HelpSteer3 showed that the original 1024-token SFT/RM limit discarded too much training signal.

| Limit | Train SFT truncation | Train RM truncation | Validation SFT truncation | Validation RM truncation |
|---:|---:|---:|---:|---:|
| 1024 | 38.47% | 40.82% | 36.78% | 39.49% |
| 2048 | 15.48% | 16.47% | 13.51% | 14.87% |
| 3072 | 5.28% | 5.83% | 4.69% | 5.32% |
| 4096 | 0.83% | 1.00% | 0.68% | 0.89% |

The project therefore moved to 4096-token SFT/RM training, reducing truncation on the training data to about 1%.

## Phase 3: SFT-4096

Final SFT settings:

```yaml
model: Qwen/Qwen2.5-0.5B-Instruct
max_length: 4096
epochs: 2
batch_size: 6
gradient_accumulation_steps: 3
learning_rate: 5.0e-6
lora_rank: 16
```

Output directory:

```text
outputs/rlhf/qwen25_05b_helpsteer3_sft_4096/
```

This stage produced the supervised policy used both as a comparison policy and as the PPO initialization/reference.

## Phase 4: Reward model 4096, two epochs

The first 4096 reward-model epoch already reached about 72% validation accuracy. A second resumed epoch produced:

| Metric | Value |
|---|---:|
| validation pairs | 1917 |
| validation accuracy | 71.62% |
| validation loss | 0.9734 |
| average margin | 0.9094 |
| code accuracy | 74.88% |
| general accuracy | 71.01% |
| stem accuracy | 63.37% |
| multilingual accuracy | 75.15% |

Output directory:

```text
outputs/rlhf/qwen25_05b_helpsteer3_reward_4096_epoch2/
```

The second epoch did not produce a dramatic jump, but the model remained substantially better than the earlier 1024-token reward model. The lower STEM accuracy is an important limitation.

## Phase 5: PPO 4096 epoch-2 long-512

Final PPO settings:

```yaml
policy_init_checkpoint_dir: outputs/rlhf/qwen25_05b_helpsteer3_sft_4096/checkpoint_final
ref_checkpoint_dir: outputs/rlhf/qwen25_05b_helpsteer3_sft_4096/checkpoint_final
reward_model.checkpoint_dir: outputs/rlhf/qwen25_05b_helpsteer3_reward_4096_epoch2/checkpoint_best
max_prompt_length: 3072
max_new_tokens: 512
learning_rate: 3.0e-7
clip_range: 0.06
kl.init_kl_coef: 0.18
kl.min_kl_coef: 0.14
ppo_epochs: 1
rollout_batch_size: 2
total_updates_requested: 400
total_updates_completed: 397
```

Output directory:

```text
outputs/rlhf/qwen25_05b_helpsteer3_ppo_4096_epoch2_long512/
```

The long PPO run was intentionally aggressive. It did not collapse: empty response rate stayed at zero and the final evaluation produced coherent long responses. But the policy remained close to the SFT reference and did not outperform the base model overall.

## Phase 6: 512-token policy-suite baseline

The first complete suite generated Base, SFT, and PPO responses once per validation prompt, scored them with the same epoch-2 reward model, and derived every pairwise comparison from the same table.

Output directory:

```text
outputs/rlhf/qwen25_05b_helpsteer3_eval_suite_4096_ep2_u400/
```

### Three-way winner counts

| Policy | Wins | Win rate | Mean reward | Median response tokens | Cap-hit rate |
|---|---:|---:|---:|---:|---:|
| Base | 827 | 41.00% | -3.5339 | 332 | 29.90% |
| SFT | 556 | 27.57% | -3.4280 | 363 | 29.95% |
| PPO | 525 | 26.03% | -3.6666 | 356 | 29.70% |
| Tie | 109 | 5.40% | — | — | — |

### Pairwise

| Comparison | Left wins | Right wins | Ties | Right win rate | Mean right-left delta |
|---|---:|---:|---:|---:|---:|
| Base vs SFT | 1044 | 927 | 46 | 45.96% | +0.1059 |
| Base vs PPO | 1068 | 904 | 45 | 44.82% | -0.1327 |
| SFT vs PPO | 965 | 898 | 154 | 44.52% | -0.2386 |

This run was a major improvement over the 96- and 128-token smoke tests, but roughly 30% of responses still reached the cap. It is retained as a baseline rather than the primary report.

## Phase 7: Primary 1024-token evaluation and audit

The evaluation cap was doubled without changing the 3072-token prompt allowance, so prompt plus response still fits within the 4096-token SFT and reward-model sequence budget.

Output directory:

```text
outputs/rlhf/qwen25_05b_helpsteer3_eval_suite_4096_ep2_u400_eval1024/
```

### Three-way winner counts

| Policy | Wins | Win rate | Mean reward | Median response tokens | Cap-hit rate |
|---|---:|---:|---:|---:|---:|
| Base | 978 | 48.49% | -3.3634 | 334 | 8.08% |
| SFT | 475 | 23.55% | -3.6114 | 360 | 10.16% |
| PPO | 467 | 23.15% | -3.5771 | 363 | 11.60% |
| Tie | 97 | 4.81% | — | — | — |

### Pairwise

| Comparison | Left wins | Right wins | Ties | Right win rate | Mean right-left delta |
|---|---:|---:|---:|---:|---:|
| Base vs SFT | 1215 | 763 | 39 | 37.83% | -0.2480 |
| Base vs PPO | 1190 | 785 | 42 | 38.92% | -0.2137 |
| SFT vs PPO | 963 | 892 | 162 | 44.22% | +0.0343 |

Cap hits fell substantially, but the longer generations exposed more repetition. PPO exceeded a 25% repeated word-level 4-gram fraction on 325 responses, 16.11% of the suite, and exceeded 50% on 156 responses. Manual review also found fabricated citations, incorrect chemistry, irrelevant continuations, and high-reward repetition loops. These failures demonstrate that reward-model score is not a substitute for qualitative or domain-specific evaluation.

The 512- and 1024-token suites are not a perfectly controlled token-limit ablation because batch size also changed. The latest run remains the primary report because it is more complete and less truncated, but metric differences should not be attributed solely to `max_new_tokens`.

## Conclusion

The experiments produced an end-to-end RLHF pipeline for Qwen2.5-0.5B-Instruct using HelpSteer3. The final long-context reward model reached 71.62% pairwise validation accuracy, PPO training remained stable, and the evaluation system completed and audited all 2017 validation prompts. PPO changed behavior and produced useful local improvements, but it did not globally outperform the base instruction model.

The 1024-token suite is a more revealing endpoint than the shorter evaluation. It reduces accidental truncation while exposing weaknesses in stopping, factuality, and reward-model judgment. The main result is therefore the complete implementation and the evidence it produces about both successful alignment behavior and failure modes. Detailed run history is in [`rlhf_evaluation_history.md`](rlhf_evaluation_history.md), manually reviewed examples are in [`rlhf_qualitative_audit.md`](rlhf_qualitative_audit.md), and the resulting research program is in [`rlhf_future_work.md`](rlhf_future_work.md).
