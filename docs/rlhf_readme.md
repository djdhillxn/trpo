# RLHF Post-Training with Qwen2.5, HelpSteer3, and Token-Level PPO

This document is the main entry point for the RLHF extension of the original TRPO / NPG / PPO repository. The original project studied policy optimization on MuJoCo and Atari. This extension applies the same ideas to language-model post-training:

> Can the same PPO trust-region idea be adapted from environment rollouts to language-model post-training?

The implementation trains an RLHF pipeline around **Qwen2.5-0.5B-Instruct** and **NVIDIA HelpSteer3**:

1. supervised fine-tuning (SFT) on preferred HelpSteer3 responses,
2. reward-model training from HelpSteer3 chosen/rejected preference pairs,
3. KL-controlled token-level PPO on sampled LLM responses,
4. policy-suite evaluation comparing Base, SFT, and PPO responses on the same prompts.

This student-scale run does not beat a modern instruction-tuned model. It does provide an end-to-end, debuggable RLHF pipeline with documented failure modes, long-context data handling, reward-model diagnostics, PPO checkpoints, resumable evaluation, and qualitative example curation.

## Why this belongs in a TRPO/PPO repository

The original repository was about conservative policy improvement: vanilla policy gradients can move the policy too far, while NPG, TRPO, and PPO measure or constrain policy movement in policy space. The RLHF extension keeps that same idea, but changes the environment:

| Earlier project | RLHF extension |
|---|---|
| MuJoCo / Atari state | chat prompt tokens |
| action | generated token |
| rollout trajectory | prompt + generated response |
| environment reward | learned scalar reward model |
| old-policy KL / trust region | KL to frozen SFT reference model |
| PPO update on action log-probs | PPO update on response-token log-probs |

This is why the RLHF implementation lives under `trpo_repro/rlhf/` instead of directly reusing the Gym-oriented PPO code. The math is still PPO-Clip with a KL anchor, but the tensors, masking, sampling, and scoring pipeline are different.

## Model and data

### Base model

We use `Qwen/Qwen2.5-0.5B-Instruct`, a 0.49B parameter instruction-tuned model. The Qwen2.5-0.5B-Instruct model card lists a **32,768-token context length** and **8,192-token generation length**. The model itself can therefore support far longer generations than we used, but full-length RLHF training is much more expensive than inference.

### Dataset

We use HelpSteer3 preference data. Each training example contains a conversation context, two candidate responses, domain/language metadata, and a preference score. The final run used the full train/validation splits after filtering invalid or tied preference rows.

### Chat formatting

HelpSteer3 stores messages in a chat-style format. Before training/evaluation, we render those messages with the Qwen tokenizer chat template:

```text
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
...
<|im_end|>
<|im_start|>assistant
```

Qwen chat/instruct models expect this format, which is produced through `tokenizer.apply_chat_template(...)`.

## Final long-context configuration

Earlier experiments used short output budgets such as 128 new tokens. Those runs were useful for debugging, but they clipped many responses and were not suitable for qualitative examples. We therefore ran a final long-context version.

| Stage | Final setting |
|---|---:|
| SFT max sequence length | 4096 total tokens |
| Reward-model max sequence length | 4096 total tokens |
| PPO max prompt length | 3072 prompt tokens |
| PPO max generated response length | 512 new tokens |
| Evaluation max prompt length | 3072 prompt tokens |
| Evaluation max generated response length | 1024 new tokens |

The evaluation budget is larger than the PPO rollout budget: inference is not required to use the same response cap as training. A 3072-token prompt plus 1024 generated tokens remains within the 4096-token sequence length used for SFT and reward-model training.

## Why 4096 mattered

A token-length diagnostic showed that the earlier 1024-token SFT/RM cap was too small for HelpSteer3:

| Limit | Train SFT truncation | Train RM truncation | Validation SFT truncation | Validation RM truncation |
|---:|---:|---:|---:|---:|
| 1024 | 38.47% | 40.82% | 36.78% | 39.49% |
| 2048 | 15.48% | 16.47% | 13.51% | 14.87% |
| 3072 | 5.28% | 5.83% | 4.69% | 5.32% |
| 4096 | 0.83% | 1.00% | 0.68% | 0.89% |

At 4096 tokens, the training stages retain substantially more of each example.

## Training stages

### 1. SFT policy

```bash
python scripts/rlhf_train_sft_policy.py \
  --config configs/rlhf/qwen25_05b_helpsteer3_sft.yaml
```

Final SFT configuration:

- base model: `Qwen/Qwen2.5-0.5B-Instruct`
- LoRA rank: 16
- max length: 4096
- epochs: 2
- batch size: 6
- gradient accumulation: 3
- learning rate: `5e-6`
- output: `outputs/rlhf/qwen25_05b_helpsteer3_sft_4096/`

The SFT stage teaches the policy to imitate the preferred HelpSteer3 response. It is the supervised alignment baseline and also the starting/reference policy for PPO.

### 2. Reward model

```bash
python scripts/rlhf_train_reward_model.py \
  --config configs/rlhf/qwen25_05b_helpsteer3_reward.yaml
```

The final reward model was trained in two one-epoch runs: first from Qwen, then resumed from the best checkpoint for a second epoch.

Final reward-model result:

| Metric | Value |
|---|---:|
| validation pairs | 1917 |
| validation accuracy | 71.62% |
| validation loss | 0.9734 |
| average reward margin | 0.9094 |
| code accuracy | 74.88% |
| general accuracy | 71.01% |
| STEM accuracy | 63.37% |
| multilingual accuracy | 75.15% |

The model is useful as a PPO training signal, but it is not a perfect judge. Reward-model-based win rates are proxy metrics rather than ground-truth human preferences.

### 3. PPO post-training

```bash
python scripts/rlhf_train_ppo.py \
  --config configs/rlhf/qwen25_05b_helpsteer3_ppo.yaml
```

Final PPO configuration:

- initial policy: `outputs/rlhf/qwen25_05b_helpsteer3_sft_4096/checkpoint_final`
- frozen reference: same SFT checkpoint
- reward model: `outputs/rlhf/qwen25_05b_helpsteer3_reward_4096_epoch2/checkpoint_best`
- max prompt length: 3072
- max new tokens: 512
- LoRA rank: 16
- total updates: 400 requested; 397 completed
- PPO epochs per rollout batch: 1
- learning rate: `3e-7`
- clip range: 0.06
- KL coefficient: initialized at 0.18 with minimum 0.14
- output: `outputs/rlhf/qwen25_05b_helpsteer3_ppo_4096_epoch2_long512/`

The PPO run did not collapse: empty-rate stayed at zero, response lengths remained long, and the checkpoint loaded correctly in the final suite evaluation. However, it did not outperform the base or SFT policies overall.

## Primary 1024-token policy-suite evaluation

Instead of running three separate pairwise evaluations, the final evaluator generates Base, SFT, and PPO responses once per prompt, scores all three with the same reward model, and derives all pairwise comparisons from the same table.

```bash
python scripts/rlhf_evaluate_policy_suite.py \
  --config configs/rlhf/qwen25_05b_helpsteer3_eval_suite.yaml
```

Final evaluation:

- split: HelpSteer3 validation
- examples: 2017
- prompt budget: 3072 tokens
- generation budget: 1024 tokens
- policies: Base Qwen, SFT-4096, PPO-4096-epoch2-update400
- output: `outputs/rlhf/qwen25_05b_helpsteer3_eval_suite_4096_ep2_u400_eval1024/`

The earlier 512-token evaluation is preserved in [`rlhf_evaluation_history.md`](rlhf_evaluation_history.md). The 1024-token suite is the primary result because it reduces cap hits from roughly 30% to 8-12%, although the two runs are not a perfectly controlled ablation because evaluation batch size also changed.

### Overall three-way winner counts

| Policy | Wins | Win rate | Mean reward | Median response tokens | Cap-hit rate | Empty rate |
|---|---:|---:|---:|---:|---:|---:|
| Base | 978 | 48.49% | -3.3634 | 334 | 8.08% | 0.00% |
| SFT-4096 | 475 | 23.55% | -3.6114 | 360 | 10.16% | 0.00% |
| PPO-4096 | 467 | 23.15% | -3.5771 | 363 | 11.60% | 0.00% |
| Tie | 97 | 4.81% | — | — | — | — |

### Pairwise comparisons

| Comparison | Left wins | Right wins | Ties | Right win rate | Mean right-left reward delta |
|---|---:|---:|---:|---:|---:|
| Base vs SFT | 1215 | 763 | 39 | 37.83% | -0.2480 |
| Base vs PPO | 1190 | 785 | 42 | 38.92% | -0.2137 |
| SFT vs PPO | 963 | 892 | 162 | 44.22% | +0.0343 |

PPO's reward margins are asymmetric. Its 785 wins over Base average `+1.6210`, while its 1190 losses average `-1.4315`. Against SFT, PPO's winning margins also exceed its losing margins on average, producing the slightly positive aggregate delta despite fewer wins. This does not overturn the win-rate result: Base remains the strongest policy under the learned reward model.

The qualitative audit adds an important limitation. At 1024 tokens, more than 25% of word-level 4-grams are repeated in 7.49% of Base, 13.78% of SFT, and 16.11% of PPO responses. Several high-reward PPO outputs are visibly broken loops, fabricated citations, or irrelevant continuations. The reward model also occasionally assigns very low scores to comparatively useful responses. See [`rlhf_qualitative_audit.md`](rlhf_qualitative_audit.md) for the evidence and full selected responses.

## How to interpret the negative reward values

The reward model outputs scalar scores on an arbitrary learned scale. The absolute value is not inherently meaningful: a score of `-3` does not mean the response is “bad” in any universal sense. What matters is the relative score between candidate responses to the same prompt.

It is also normal for reward distributions to look roughly bell-shaped. The reward head is a learned scalar regressor/ranker on top of transformer representations. After training, many examples cluster near the model's typical score range, while outliers appear for unusually preferred or dispreferred responses. In PPO training we clipped rewards to control optimization, but the evaluation plots show raw reward-model outputs.

## Scope and findings

The implementation includes:

- an end-to-end RLHF pipeline;
- long-context SFT and reward-model training;
- pairwise reward modeling with domain-level diagnostics;
- token-level PPO with KL anchoring to a frozen SFT reference;
- observed RLHF failure modes and fixes: gibberish drift, vulgar output, EOS/blank collapse, wrong checkpoint loading, and non-resumable long evaluation;
- full-validation policy-suite evaluation;
- curation tooling to inspect both wins and failures.

The results do **not** show that a 0.5B PPO adapter beats Qwen2.5-Instruct at scale. The reward model provides a usable training signal and PPO remains stable, but Base wins most comparisons and the longer audit exposes repetition and judge failures. The implementation, reproducible diagnostics, and concrete failure analysis are the main outcomes.

## Recommended reading order

- [`rlhf_experiments.md`](rlhf_experiments.md): experiment timeline, failed runs, and final metrics.
- [`rlhf_evaluation_history.md`](rlhf_evaluation_history.md): the primary 1024-token suite and archived 512-token baseline.
- [`rlhf_qualitative_audit.md`](rlhf_qualitative_audit.md): manual analysis of useful responses, failures, and reward-model mismatches.
- [`rlhf_technical_notes.md`](rlhf_technical_notes.md): SFT/RM/PPO mechanics and why the hyperparameters matter.
- [`rlhf_curation_guide.md`](rlhf_curation_guide.md): how to reproduce and extend the qualitative review.
- [`rlhf_future_work.md`](rlhf_future_work.md): a prioritized research program based on the observed limitations.
- `notebooks/analyzing_full_eval_results.ipynb`: summary analysis notebook for the final policy-suite outputs.
- `notebooks/rlhf_full_eval_and_curation.ipynb`: interactive example browser and curation notebook.
