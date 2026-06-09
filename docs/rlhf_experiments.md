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

## Phase 6: Final policy-suite evaluation

Final evaluation generated Base, SFT, and PPO responses once per validation prompt, scored them with the same epoch-2 reward model, and derived every pairwise comparison from the same table.

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

### Domain-level observations

- PPO was most competitive on **code** and **general** prompts, but still did not beat base overall.
- SFT had slightly better mean reward than base but did not win pairwise more often.
- PPO and SFT both produced many long complete responses, unlike the early 128-token runs.
- The full run is therefore more useful qualitatively than the early runs, even though aggregate PPO win rate is mixed.

## Conclusion

The experiments produced an end-to-end RLHF pipeline for Qwen2.5-0.5B-Instruct using HelpSteer3. The final long-context reward model reached 71.62% pairwise validation accuracy. PPO training was stable and produced complete long-form outputs, but did not globally outperform the base instruction model under the learned reward model. The main result is the implementation and its diagnostics, rather than evidence that a small PPO adapter can beat a production instruction model.
