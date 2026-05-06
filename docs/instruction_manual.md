# Instruction Manual

This file collects the detailed usage notes for the TRPO / PPO policy-optimization repository. It absorbs the former CLI reference content and the detailed operational sections that used to live in the top-level README. The README is intentionally shorter now; this manual keeps the gory details close at hand.

## CLI reference for `scripts/train.py`

This file lists the high-value command line overrides supported by `scripts/train.py`.

### Core flags

| Flag | YAML target | Meaning |
|---|---|---|
| `--config PATH` | n/a | config file to load |
| `--seed INT` | `train.seed` | random seed |
| `--device cpu|cuda|...` | n/a | learner device |
| `--output-dir PATH` | n/a | output directory |
| `--overwrite` | n/a | clear and recreate output directory |
| `--resume-from PATH` | n/a | load a saved checkpoint and continue from its next epoch |

### Method selection

| Flag | YAML target | Meaning |
|---|---|---|
| `--method NAME` | `method.name` | `trpo`, `natural_pg`, `npg`, `trpo_max_kl`, `empirical_fim`, `ppo`, `random` |
| `--method-variant NAME` | `method.variant` | mainly used for PPO: `clip`, `kl_penalty` |
| `--estimator NAME` | `algo.estimator` | `trpo_paper`, `value_baseline`, `gae` (old aliases still work) |

### Training-budget overrides

| Flag | YAML target | Meaning |
|---|---|---|
| `--epochs INT` | `train.epochs` | number of training epochs / policy iterations |
| `--steps-per-epoch INT` | `train.steps_per_epoch` | target environment steps per epoch |
| `--save-interval INT` | `train.save_interval` | checkpoint frequency |

### Parallel rollout controls

| Flag | YAML target | Meaning |
|---|---|---|
| `--num-workers INT` | `train.num_workers` | number of rollout workers |
| `--num-cores INT` | `train.num_workers` | alias for `--num-workers` |

### Runtime / memory controls

| Flag | YAML target | Meaning |
|---|---|---|
| `--memory-mode standard|safe` | `train.memory_mode` | choose original vs memory-safe path |
| `--progress-mode auto|terminal|notebook|off` | `train.progress_mode` | progress-bar mode |
| `--obs-storage auto|ram|memmap` | `train.obs_storage` | where rollout observations live |
| `--full-batch-chunk-size INT` | `algo.full_batch_chunk_size` | safe-mode TRPO chunk size |

### Fisher / Hessian controls

| Flag | YAML target | Meaning |
|---|---|---|
| `--fvp-subsample-fraction FLOAT` | `algo.fvp_subsample_fraction` | fraction of the batch used for the FVP metric |
| `--fvp-estimator analytic|empirical` | `algo.fvp_estimator` | choose analytic KL-Hessian or empirical FIM |

### Example recipes

#### Full 500-epoch Atari TRPO run

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

#### MuJoCo empirical-FIM ablation

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

#### Atari PPO with parallel rollouts

```bash
python3 scripts/train.py \
  --config configs/atari/seaquest_ppo_clip.yaml \
  --num-workers 4 \
  --memory-mode standard \
  --device cuda \
  --progress-mode notebook \
  --overwrite
```

#### Resume from a checkpoint

`--epochs` is still the final target epoch. For example, resuming an Atari run
from epoch 250 to finish at epoch 300 runs epochs 251 through 300.

```bash
python3 scripts/train.py \
  --config outputs/pong_single_path/seed_0/config_runtime.yaml \
  --resume-from outputs/pong_single_path/seed_0/checkpoints/epoch_0250.pt \
  --epochs 300 \
  --device cuda
```

---

## Detailed Repository Guide

The following material was moved from the former long README so that no command, inventory item, runtime note, output-schema detail, or paper-alignment note is lost.

### TRPO / PPO policy optimization experiments

A research-oriented reinforcement learning repository for **Trust Region Policy Optimization (TRPO)** and related policy-optimization methods. The codebase now supports:

- **TRPO** (single-path / trust-region constrained)
- **Natural Policy Gradient (NPG)**
- **Empirical-FIM TRPO** (paper ablation / comparison path)
- **PPO-Clip**
- **PPO-KL-Penalty**
- **Random policy baseline**
- **Single-process or parallel rollout collection**
- **Laptop-safe memory mode** for large Atari TRPO runs
- **Colab-friendly workflows** and an included notebook

This repo is designed for **paper-faithful TRPO study**, but it also includes the practical machinery needed to run larger comparative experiments on MuJoCo and Atari.

---

### Most important commands

#### Train a run

```bash
# TRPO single-path / paper-faithful
python3 scripts/train.py \
  --config configs/mujoco/swimmer_single_path.yaml \
  --overwrite

python3 scripts/train.py \
  --config configs/atari/seaquest_single_path.yaml \
  --overwrite

# PPO clipped surrogate
python3 scripts/train.py \
  --config configs/mujoco/swimmer_ppo_clip.yaml \
  --overwrite

python3 scripts/train.py \
  --config configs/atari/seaquest_ppo_clip.yaml \
  --overwrite

# Empirical-FIM TRPO ablation
python3 scripts/train.py \
  --config configs/mujoco/swimmer_empirical_fim.yaml \
  --overwrite
```

#### Common overrides

```bash
# Use CUDA, set workers, and change the training budget from the CLI
python3 scripts/train.py \
  --config configs/atari/seaquest_single_path.yaml \
  --device cuda \
  --num-workers 4 \
  --epochs 500 \
  --steps-per-epoch 100000 \
  --save-interval 25 \
  --overwrite

# Override the estimator mode directly
python3 scripts/train.py \
  --config configs/mujoco/cartpole_linear.yaml \
  --estimator value_baseline \
  --overwrite

# Switch the Fisher / Hessian metric implementation
python3 scripts/train.py \
  --config configs/mujoco/swimmer_single_path.yaml \
  --method empirical_fim \
  --fvp-estimator empirical \
  --fvp-subsample-fraction 0.1 \
  --overwrite
```

#### Aggregate one method over seeds

```bash
python3 scripts/aggregate_results.py \
  --runs-root outputs/swimmer_single_path \
  --metric train_return_mean \
  --x-axis iteration \
  --summary
```

#### Compare methods on the same environment

```bash
python3 scripts/aggregate_results.py \
  --runs-root outputs/swimmer_single_path \
  --runs-root outputs/swimmer_natural_pg \
  --runs-root outputs/swimmer_empirical_fim \
  --runs-root outputs/swimmer_ppo_clip \
  --compare \
  --metric train_return_mean \
  --x-axis iteration \
  --smooth-window 5 \
  --summary
```

#### Evaluate a saved checkpoint

```bash
python3 scripts/evaluate.py \
  --config configs/mujoco/swimmer_single_path.yaml \
  --checkpoint outputs/swimmer_single_path/seed_0/checkpoints/epoch_0050.pt \
  --episodes 10 \
  --device cpu
```

---

### Quick start mental model

Think about the repo in four layers:

1. **Config**: choose a YAML file in `configs/`
2. **Method**: TRPO / NPG / empirical-FIM / PPO / random
3. **Runtime mode**: workers, device, safe-vs-standard memory, progress bars, FVP settings
4. **Analysis**: aggregate, smooth, summarize, compare

Typical workflow:

1. install dependencies
2. pick a config
3. run one or more seeds into `outputs/<run_name>/seed_<k>/`
4. aggregate one method over seeds
5. compare methods on the same environment
6. optionally evaluate checkpoints

---

### Colab quickstart

A reference notebook is included here:

- `docs/colab_runmanager.ipynb`

It shows how to:

- mount Google Drive
- copy the repo from Drive to `/content/trpo`
- install dependencies inside the Colab runtime
- verify CUDA
- run MuJoCo and Atari jobs
- store intermediate outputs on Colab SSD instead of Drive
- copy finished outputs back to Drive

#### Minimal Colab SSD workflow

```bash
rm -rf /content/trpo
cp -r "/content/drive/MyDrive/Colab Notebooks/839/trpo" /content/trpo
cd /content/trpo
```

Then run jobs from `/content/trpo` and store outputs on `/content/trpo_runs/...`.

#### Example: Atari TRPO on Colab with parallel rollouts and CUDA

```bash
python3 scripts/train.py \
  --config configs/atari/seaquest_single_path.yaml \
  --num-workers 4 \
  --memory-mode safe \
  --obs-storage ram \
  --full-batch-chunk-size 8192 \
  --fvp-subsample-fraction 0.1 \
  --device cuda \
  --progress-mode off \
  --output-dir /content/trpo_runs/seaquest_single_path/seed_0 \
  --overwrite
```

#### Example: Atari PPO on Colab with parallel rollouts and CUDA

```bash
python3 scripts/train.py \
  --config configs/atari/seaquest_ppo_clip.yaml \
  --num-workers 4 \
  --memory-mode standard \
  --device cuda \
  --progress-mode notebook \
  --output-dir /content/trpo_runs/seaquest_ppo_clip/seed_0 \
  --overwrite
```

---

### Installation

```bash
python3 -m pip install -r requirements.txt
```

When you run the entry-point scripts from this checkout, for example `python3 scripts/train.py ...`,
they bootstrap the repository root ahead of site-packages. That means they import the local code
you are standing in, even if an older `trpo_repro` package is installed elsewhere.

Editable install is optional and only needed if you want to import `trpo_repro` from outside the
repository root.

---

### Atari setup

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

---

### Methods and variants

#### Method names

| Method name | Meaning | Typical use |
|---|---|---|
| `trpo` | Trust Region Policy Optimization with analytic KL-Hessian / Fisher-vector products | main theory-heavy method |
| `natural_pg` / `npg` | natural policy gradient with the same surrogate geometry but no line search | locomotion comparison baseline |
| `trpo_max_kl` | TRPO variant using max-KL acceptance logic | small-scale ablation / sanity check |
| `empirical_fim` | TRPO-style update using the empirical covariance of score gradients for the metric | paper-linked empirical-FIM ablation (use `memory_mode standard`) |
| `ppo` | proximal policy optimization | practical first-order comparison |
| `random` | random policy baseline | sanity / baseline curve |

#### PPO variants

| Method | `method.variant` | Meaning |
|---|---|---|
| `ppo` | `clip` | clipped-ratio PPO surrogate |
| `ppo` | `kl_penalty` | KL-penalty PPO surrogate |

#### Estimator names (public names)

The repo now uses clearer public estimator names.

| Public name | Meaning | Old aliases still accepted |
|---|---|---|
| `trpo_paper` | paper-faithful Monte Carlo returns / trajectory Q-estimates with no learned value baseline | `paper_mc`, `paper_returns`, `paper` |
| `value_baseline` | Monte Carlo returns with a learned value baseline subtracted from the policy-training weights | `mc_baseline`, `mc_value_baseline`, `returns_value_baseline`, `mc` |
| `gae` | generalized advantage estimation | `gae` |

#### Which estimator should I use?

| Estimator | Best use | Uses learned value function? |
|---|---|---|
| `trpo_paper` | paper-faithful single-path TRPO / NPG experiments | No |
| `value_baseline` | lower-variance actor-critic style TRPO/NPG experiments | Yes |
| `gae` | PPO and modernized actor-critic training | Yes |

---

### Runtime controls

#### Memory mode

| `train.memory_mode` | Meaning | When to use |
|---|---|---|
| `standard` | original full-RAM / full-batch path | MuJoCo, PPO, or large-memory machines |
| `safe` | memory-safe path using chunked second-order computations | large Atari TRPO / NPG runs, laptops, constrained Colab runs |

#### Observation storage

| `train.obs_storage` | Meaning |
|---|---|
| `auto` | choose `memmap` in safe mode, `ram` in standard mode |
| `ram` | keep rollout observations in memory |
| `memmap` | keep rollout observations in disk-backed memory maps |

#### Progress mode

| `train.progress_mode` | Meaning |
|---|---|
| `auto` | choose terminal vs notebook tqdm automatically |
| `terminal` | shell-friendly tqdm |
| `notebook` | Colab / Jupyter-friendly tqdm |
| `off` | disable progress bars |

#### FVP / Fisher controls

| Setting | Meaning |
|---|---|
| `algo.full_batch_chunk_size` | chunk size for safe-mode TRPO full-batch computations |
| `algo.fvp_subsample_fraction` | fraction of the batch used for the Fisher-vector-product metric |
| `algo.fvp_estimator` | `analytic` or `empirical` |
| `algo.empirical_fim_batch_size` | micro-batch size used when computing empirical-FIM score-gradient products |

#### PPO scheduling controls

| Setting | Meaning |
|---|---|
| `algo.ppo_anneal_lr` | linearly anneal PPO learning rates to zero over training |
| `algo.ppo_anneal_clip_ratio` | linearly anneal PPO clip ratio to zero over training |

The Atari PPO configs enable both annealing options by default, following the spirit of the PPO paper’s Atari schedule.

---

### Parallel rollout collection

Parallel rollout collection is **optional** and only turns on when `num_workers > 1`.

#### Important notes

- default remains **single-process** collection
- intended for **Linux / Colab**
- not supported on macOS in this repo
- current parallel path expects:
  - `normalize_obs: false`
  - `obs_storage: ram`

#### Good starting points

| Scenario | Suggested settings |
|---|---|
| MuJoCo TRPO / NPG on Colab | `--num-workers 12 --device cuda` |
| Atari TRPO on Colab | `--num-workers 4 --memory-mode safe --obs-storage ram --full-batch-chunk-size 8192 --device cuda` |
| Atari PPO on Colab | `--num-workers 4 --memory-mode standard --device cuda` |

---

### High-value CLI overrides

These are the ones you will most often care about:

```bash
python3 scripts/train.py \
  --config ... \
  --method trpo \
  --method-variant clip \
  --estimator trpo_paper \
  --epochs 500 \
  --steps-per-epoch 100000 \
  --save-interval 25 \
  --num-workers 4 \
  --memory-mode safe \
  --obs-storage ram \
  --full-batch-chunk-size 8192 \
  --fvp-subsample-fraction 0.1 \
  --fvp-estimator analytic \
  --device cuda \
  --progress-mode off \
  --overwrite
```

A full CLI flag reference is at the top of this manual.

---

### Output structure and logged metrics

Each run typically writes to:

```text
outputs/<run_name>/seed_<k>/
```

Important files:

| File | Meaning |
|---|---|
| `config_resolved.yaml` | fully resolved config after inheritance and CLI overrides |
| `config_runtime.yaml` | runtime-resolved config after automatic settings such as observation storage are finalized |
| `launch_metadata.json` | raw command, argv, CLI args, CLI overrides, config path, output path, and package path captured before environment creation |
| `run_metadata.json` | run manifest with command, CLI overrides, method/env identity, runtime settings, Git info, and lifecycle status |
| `environment.json` | Python/platform/package/CUDA environment snapshot |
| `run_summary.json` | final status, duration, total steps, final epoch metrics, best return, and saved checkpoints |
| `failure.json` | exception details and traceback, written only if a run fails or is interrupted |
| `git_diff.patch` | tracked local code diff at launch, written only when there are tracked uncommitted changes |
| `metrics.jsonl` | one JSON record per epoch / iteration |
| `metrics.csv` | same epoch metrics in CSV form for aggregation |
| `checkpoints/` | saved method checkpoints |
| `_buffers/` | temporary memmap rollout storage (when used) |

#### Important metrics

| Metric | Meaning |
|---|---|
| `train_return_mean` | mean raw episodic return collected in that epoch |
| `train_return_std` | std. dev. of raw episodic return collected in that epoch |
| `ep_len_mean` | mean episode length in that epoch |
| `approx_kl` | approximate KL change after the policy update |
| `clip_fraction` | PPO fraction of ratios outside the clip band |
| `value_explained_variance_after` | PPO critic explained variance after the update |
| `collect_time_sec` | rollout-collection wall time |
| `update_time_sec` | optimization/update wall time |
| `checkpoint_time_sec` | checkpoint save wall time |
| `wall_time_sec` | total epoch wall time |

These are **raw environment returns / scores**, not extra normalized reward metrics. That is aligned with how the TRPO and PPO papers report their benchmark results.

---

### Aggregation and comparison

#### One method over seeds

```bash
python3 scripts/aggregate_results.py \
  --runs-root outputs/walker2d_single_path \
  --metric train_return_mean \
  --x-axis iteration \
  --summary
```

#### Compare multiple methods on one environment

```bash
python3 scripts/aggregate_results.py \
  --runs-root outputs/walker2d_single_path \
  --runs-root outputs/walker2d_natural_pg \
  --runs-root outputs/walker2d_empirical_fim \
  --runs-root outputs/walker2d_ppo_clip \
  --compare \
  --metric train_return_mean \
  --x-axis iteration \
  --smooth-window 5 \
  --summary
```

Comparison mode checks that all supplied runs come from the **same environment**.

---

### Config inventory

#### MuJoCo / locomotion examples

- `configs/mujoco/swimmer_single_path.yaml`
- `configs/mujoco/hopper_single_path.yaml`
- `configs/mujoco/walker2d_single_path.yaml`
- `configs/mujoco/swimmer_natural_pg.yaml`
- `configs/mujoco/hopper_natural_pg.yaml`
- `configs/mujoco/walker2d_natural_pg.yaml`
- `configs/mujoco/swimmer_empirical_fim.yaml`
- `configs/mujoco/hopper_empirical_fim.yaml`
- `configs/mujoco/walker2d_empirical_fim.yaml`
- `configs/mujoco/swimmer_ppo_clip.yaml`
- `configs/mujoco/hopper_ppo_clip.yaml`
- `configs/mujoco/walker2d_ppo_clip.yaml`

#### Atari examples

- `configs/atari/beamrider_single_path.yaml`
- `configs/atari/breakout_single_path.yaml`
- `configs/atari/enduro_single_path.yaml`
- `configs/atari/pong_single_path.yaml`
- `configs/atari/qbert_single_path.yaml`
- `configs/atari/seaquest_single_path.yaml`
- `configs/atari/spaceinvaders_single_path.yaml`
- `configs/atari/*_ppo_clip.yaml`
- `configs/atari/seaquest_ppo_kl_penalty.yaml`

---

### Code map

| Concept | File |
|---|---|
| Main training entry point | `scripts/train.py` |
| Evaluation entry point | `scripts/evaluate.py` |
| Aggregation / plotting | `scripts/aggregate_results.py` |
| Main training loop | `trpo_repro/runner.py` |
| TRPO / NPG / empirical-FIM method wrappers | `trpo_repro/methods/trpo_method.py` |
| PPO method wrapper | `trpo_repro/methods/ppo_method.py` |
| Random baseline | `trpo_repro/methods/random_policy.py` |
| TRPO optimizer logic | `trpo_repro/algos/trpo.py` |
| PPO optimizer logic | `trpo_repro/algos/ppo.py` |
| Shared returns / advantages | `trpo_repro/algos/advantages.py` |
| Parallel rollout manager | `trpo_repro/rollouts/manager.py` |
| Parallel rollout worker | `trpo_repro/rollouts/worker.py` |
| Rollout buffer | `trpo_repro/data/buffer.py` |
| Policies | `trpo_repro/models/policies.py` |
| Value functions | `trpo_repro/models/value_functions.py` |

---

### Paper alignment notes

- **TRPO single-path** is the main paper-faithful path in this repo. The original TRPO paper also included the **vine** method, but this repo focuses its mature training path on the single-path variant.
- **Natural policy gradient** is implemented because it is the most natural low-level comparison to TRPO’s trust-region step.
- **Empirical FIM** is included as a paper-linked ablation, because the TRPO paper explicitly compares the analytic KL-Hessian / Fisher construction against an empirical-FIM alternative.
- **PPO** includes both the clipped and KL-penalty variants, with the clipped version as the default practical path.

---

### Recommended freeze-and-run practice

Before launching long experiment blocks:

1. run one short smoke test for each method/environment family
2. verify `config_resolved.yaml` contains the intended overrides
3. verify `run_metadata.json` contains the expected commit hash and runtime settings
4. then freeze the repo and launch the long runs

At this point, this repository is intended to be used exactly that way.
