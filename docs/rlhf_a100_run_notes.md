# A100 40GB RLHF run profile

This profile is meant for Google Colab A100 40GB with Qwen2.5-0.5B-Instruct.
It favors speed and reasonable utilization over the earlier low-memory debug defaults.

## Reward model

Use the default config:

```bash
python3 scripts/rlhf_train_reward_model.py \
  --config configs/rlhf/qwen25_05b_helpsteer3_reward.yaml
```

Important settings:

- full BF16 backbone (`load_in_4bit: false`) because a 0.5B model fits comfortably on A100 and is faster than bitsandbytes 4-bit training;
- `batch_size: 8`, `gradient_accumulation_steps: 2`, effective batch size 16;
- full HelpSteer3 preference data after filtering ties/malformed pairs;
- token/log throughput and CUDA memory are written to `train_metrics.jsonl`/CSV;
- CSV files and plots refresh every `artifact_every` optimizer steps.

If peak GPU memory stays below roughly 28-30 GB and the run is stable, try `batch_size: 12` with `gradient_accumulation_steps: 2`.
If memory exceeds roughly 36 GB or the run OOMs, use `batch_size: 6`.

## PPO

Use:

```bash
python3 scripts/rlhf_train_ppo.py \
  --config configs/rlhf/qwen25_05b_helpsteer3_ppo.yaml
```

Important settings:

- policy/reference/reward models are full BF16;
- rollout batch size is 16;
- PPO minibatch size is 4;
- max prompt length is 768 and max generation length is 160;
- checkpoints are saved every 50 updates;
- CSV metrics refresh every 25 updates.

If peak GPU memory is comfortably below 30-32 GB, increase `rollout_batch_size` to 24 first.
Only increase `minibatch_size` to 6 or 8 after confirming that the PPO update step is not near OOM.

## Evaluation

Use:

```bash
python3 scripts/rlhf_evaluate_before_after.py \
  --config configs/rlhf/qwen25_05b_helpsteer3_eval.yaml
```

The default evaluates 200 validation prompts and writes JSONL, CSV, and Markdown before/after examples.
