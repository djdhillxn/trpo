# CLI reference for `scripts/train.py`

This file lists the high-value command line overrides supported by `scripts/train.py`.

## Core flags

| Flag | YAML target | Meaning |
|---|---|---|
| `--config PATH` | n/a | config file to load |
| `--seed INT` | `train.seed` | random seed |
| `--device cpu|cuda|...` | n/a | learner device |
| `--output-dir PATH` | n/a | output directory |
| `--overwrite` | n/a | clear and recreate output directory |
| `--resume-from PATH` | n/a | load a saved checkpoint and continue from its next epoch |

## Method selection

| Flag | YAML target | Meaning |
|---|---|---|
| `--method NAME` | `method.name` | `trpo`, `natural_pg`, `npg`, `trpo_max_kl`, `empirical_fim`, `ppo`, `random` |
| `--method-variant NAME` | `method.variant` | mainly used for PPO: `clip`, `kl_penalty` |
| `--estimator NAME` | `algo.estimator` | `trpo_paper`, `value_baseline`, `gae` (old aliases still work) |

## Training-budget overrides

| Flag | YAML target | Meaning |
|---|---|---|
| `--epochs INT` | `train.epochs` | number of training epochs / policy iterations |
| `--steps-per-epoch INT` | `train.steps_per_epoch` | target environment steps per epoch |
| `--save-interval INT` | `train.save_interval` | checkpoint frequency |

## Parallel rollout controls

| Flag | YAML target | Meaning |
|---|---|---|
| `--num-workers INT` | `train.num_workers` | number of rollout workers |
| `--num-cores INT` | `train.num_workers` | alias for `--num-workers` |

## Runtime / memory controls

| Flag | YAML target | Meaning |
|---|---|---|
| `--memory-mode standard|safe` | `train.memory_mode` | choose original vs memory-safe path |
| `--progress-mode auto|terminal|notebook|off` | `train.progress_mode` | progress-bar mode |
| `--obs-storage auto|ram|memmap` | `train.obs_storage` | where rollout observations live |
| `--full-batch-chunk-size INT` | `algo.full_batch_chunk_size` | safe-mode TRPO chunk size |

## Fisher / Hessian controls

| Flag | YAML target | Meaning |
|---|---|---|
| `--fvp-subsample-fraction FLOAT` | `algo.fvp_subsample_fraction` | fraction of the batch used for the FVP metric |
| `--fvp-estimator analytic|empirical` | `algo.fvp_estimator` | choose analytic KL-Hessian or empirical FIM |

## Example recipes

### Full 500-epoch Atari TRPO run

```bash
python3 scripts/train.py \
  --config configs/atari/seaquest_single_path.yaml \
  --epochs 500 \
  --num-workers 4 \
  --memory-mode safe \
  --obs-storage ram \
  --full-batch-chunk-size 8192 \
  --device cuda \
  --progress-mode off \
  --overwrite
```

### MuJoCo empirical-FIM ablation

Use `memory_mode standard` for empirical-FIM runs.


```bash
python3 scripts/train.py \
  --config configs/mujoco/swimmer_single_path.yaml \
  --method empirical_fim \
  --fvp-estimator empirical \
  --fvp-subsample-fraction 0.1 \
  --device cuda \
  --overwrite
```

### Atari PPO with parallel rollouts

```bash
python3 scripts/train.py \
  --config configs/atari/seaquest_ppo_clip.yaml \
  --num-workers 4 \
  --memory-mode standard \
  --device cuda \
  --progress-mode notebook \
  --overwrite
```

### Resume from a checkpoint

`--epochs` is still the final target epoch. For example, resuming an Atari run
from epoch 250 to finish at epoch 300 runs epochs 251 through 300.

```bash
python3 scripts/train.py \
  --config outputs/pong_single_path/seed_0/config_runtime.yaml \
  --resume-from outputs/pong_single_path/seed_0/checkpoints/epoch_0250.pt \
  --epochs 300 \
  --device cuda
```
