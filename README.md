# TRPO paper reproduction

## Most important commands

### Train a run
```bash
python3 scripts/train.py --config configs/mujoco/swimmer_single_path.yaml --overwrite
python3 scripts/train.py --config configs/atari/seaquest_single_path.yaml --overwrite
python3 scripts/train.py --config configs/mujoco/swimmer_ppo_clip.yaml --overwrite
python3 scripts/train.py --config configs/atari/seaquest_ppo_clip.yaml --overwrite
python3 scripts/train.py --config configs/atari/seaquest_ppo_clip.yaml --method ppo --method-variant kl_penalty --overwrite
```

### Override runtime behavior
```bash
python3 scripts/train.py --config configs/atari/seaquest_single_path.yaml --memory-mode safe --progress-mode terminal --overwrite
python3 scripts/train.py --config configs/atari/seaquest_single_path.yaml --memory-mode standard --progress-mode notebook --overwrite
```

### Aggregate one method over seeds
```bash
python3 scripts/aggregate_results.py --runs-root outputs/swimmer_single_path --metric train_return_mean --x-axis iteration
python3 scripts/aggregate_results.py --runs-root outputs/swimmer_single_path --metric train_return_mean --x-axis iteration --summary --smooth-window 5
```

### Compare several methods on the same environment
```bash
python3 scripts/aggregate_results.py \
  --runs-root outputs/swimmer_single_path \
  --runs-root outputs/swimmer_natural_pg \
  --runs-root outputs/swimmer_random \
  --compare \
  --metric train_return_mean \
  --x-axis iteration
```

## How to use this repo

This repo is organized around paper-faithful **single-path TRPO** (`paper_mc`) first, and also includes **PPO** as a separate method family. The normal workflow is:

1. install the repo and dependencies
2. run `train.py` for a config
3. inspect `outputs/<run_name>/seed_<k>/`
4. run `aggregate_results.py` for one method or for a same-environment comparison

For MacBook Atari runs, use `memory_mode: safe` to avoid blowing up RAM. For Colab or large-RAM machines, switch back to `memory_mode: standard` if you want the original full-RAM / full-batch path.

## Installation

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

## Atari setup

```bash
python3 -m pip install --upgrade ale-py "autorom[accept-rom-license]"
AutoROM --accept-license
```

Quick verification:

```bash
python3 - <<'PY'
import gymnasium as gym
import ale_py
gym.register_envs(ale_py)
env = gym.make('ALE/Seaquest-v5')
env.reset(seed=0)
env.close()
print('OK')
PY
```

## Runtime modes

### `train.memory_mode`
- `standard`: keep rollout observations in RAM and do the original full-batch update path
- `safe`: use disk-backed rollout storage plus chunked TRPO batch computations

### `train.progress_mode`
- `auto`: detect terminal vs notebook/Colab
- `terminal`: shell-style tqdm
- `notebook`: notebook-safe progress display
- `off`: no progress bars

## Output layout

Each run writes to:

```text
outputs/<run_name>/seed_<seed>/
├── checkpoints/
├── config_resolved.yaml
├── metrics.csv
├── metrics.jsonl
└── run_metadata.json
```

Comparison plots are written under:

```text
outputs/comparisons/<env_slug>/
```

## Notes

- `paper_mc` remains the default paper-faithful estimator.
- The Atari memory-safe path changes **storage and execution order**, not the TRPO objective.
- For same-environment method comparison, `aggregate_results.py` now checks that all runs share the same `env_id`.
- Aggregation supports optional smoothing (`--smooth-window`) and an optional final/best summary table (`--summary`).

## Repo layout

```text
trpo_repro/
├── algos/
├── data/
├── envs/
├── methods/
├── models/
└── utils/
```
