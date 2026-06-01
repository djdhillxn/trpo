# RLHF Post-Training with Token-Level PPO

This extension adapts the repository's PPO policy-optimization work from Gym/MuJoCo/Atari rollouts to language-model post-training.

The main experiment trains **Qwen2.5-0.5B-Instruct** with PPO on prompts derived from **HelpSteer3**. A reward model is first trained from HelpSteer3 chosen/rejected preference pairs. PPO then updates a LoRA policy adapter against reward-model scores while applying a KL penalty to a frozen reference copy of the starting model.

## Why this is separate from `trpo_repro.algos.ppo`

The original PPO implementation assumes fixed-shape observations and Gym actions. RLHF uses variable-length token sequences, response masks, token log-probs, a frozen reference model, and a reward model. The math is still PPO-Clip, but the data path is different, so the RLHF implementation lives under:

```text
trpo_repro/rlhf/
```

## Pipeline

1. Prepare HelpSteer3 preference pairs.
2. Train a scalar reward model with pairwise ranking loss.
3. Generate responses from the current policy.
4. Score responses with the reward model.
5. Add KL penalty against a frozen reference model.
6. Compute token-level GAE and returns.
7. Run PPO-Clip updates over response tokens only.
8. Evaluate base-vs-PPO responses.

## Install

```bash
pip install -r requirements-rlhf.txt
pip install -e .
```

On Colab, use a GPU runtime. The configs default to 4-bit loading for the trainable policy, frozen reference model, and frozen reward model to reduce memory use.

## Run

### 1. Reward model

```bash
python scripts/rlhf_train_reward_model.py \
  --config configs/rlhf/qwen25_05b_helpsteer3_reward.yaml
```

Outputs:

```text
outputs/rlhf/qwen25_05b_helpsteer3_reward/
  checkpoint_final/
  train_metrics.jsonl
  eval_metrics.jsonl
  final_eval_metrics.json
```

### 2. PPO post-training

```bash
python scripts/rlhf_train_ppo.py \
  --config configs/rlhf/qwen25_05b_helpsteer3_ppo.yaml
```

Outputs:

```text
outputs/rlhf/qwen25_05b_helpsteer3_ppo/
  checkpoint_final/
  ppo_metrics.jsonl
  samples/
```

### 3. Before/after evaluation

```bash
python scripts/rlhf_evaluate_before_after.py \
  --config configs/rlhf/qwen25_05b_helpsteer3_eval.yaml
```

Outputs:

```text
outputs/rlhf/qwen25_05b_helpsteer3_eval/
  before_after_samples.jsonl
  before_after_samples.csv
  before_after_demo.md
```

## Key metrics

- `reward_model_score`: learned reward model score for generated responses.
- `objective_kl`: token-level KL estimate between policy and frozen reference.
- `non_score_reward`: KL penalty contribution.
- `total_reward`: final reward signal used by PPO.
- `approx_kl`: PPO old-policy vs updated-policy KL estimate.
- `clip_fraction`: fraction of tokens where PPO ratio was clipped.
- `value_explained_variance`: critic fit quality.

## Colab tips

Start with the provided small settings:

- reward model `max_train_samples: 8000`
- PPO `total_updates: 300`
- PPO `rollout_batch_size: 4`
- generation `max_new_tokens: 128`

If the run is stable, increase samples and PPO updates. If memory is tight, reduce `rollout_batch_size`, `max_prompt_length`, or `max_new_tokens`.

## Notes

This implementation intentionally does not use TRL's PPOTrainer as the main engine. It uses Hugging Face Transformers, Datasets, PEFT/LoRA, and bitsandbytes for model infrastructure, but the PPO loop is implemented in this repository so the project remains a direct extension of the original policy-optimization codebase.
