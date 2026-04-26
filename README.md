# TRPO / PPO policy optimization experiments

A research-oriented reinforcement learning repository built around **Trust Region Policy Optimization (TRPO)**, with additional support for **Natural Policy Gradient (NPG)**, **Proximal Policy Optimization (PPO)**, and a **random policy baseline**. The codebase is organized so you can run paper-faithful TRPO experiments, compare methods on the same environments, and aggregate results into publication/report-ready plots.

---

## Most important commands

### 1) Train a run

```bash
# TRPO single-path / paper-faithful
python3 scripts/train.py --config configs/mujoco/swimmer_single_path.yaml --overwrite
python3 scripts/train.py --config configs/atari/seaquest_single_path.yaml --overwrite

# PPO clipped surrogate
python3 scripts/train.py --config configs/mujoco/swimmer_ppo_clip.yaml --overwrite
python3 scripts/train.py --config configs/atari/seaquest_ppo_clip.yaml --overwrite

# PPO KL-penalty variant via CLI override
python3 scripts/train.py \
  --config configs/atari/seaquest_ppo_clip.yaml \
  --method ppo \
  --method-variant kl_penalty \
  --overwrite

# Natural Policy Gradient
python3 scripts/train.py --config configs/mujoco/swimmer_natural_pg.yaml --overwrite

# Random baseline
python3 scripts/train.py --config configs/mujoco/swimmer_random.yaml --overwrite
```

### 2) Override runtime behavior from the command line

```bash
# Memory-safe Atari mode (good for laptops / lower-memory machines)
python3 scripts/train.py \
  --config configs/atari/seaquest_single_path.yaml \
  --memory-mode safe \
  --progress-mode terminal \
  --overwrite

# Colab-friendly Atari run with CUDA and RAM-backed observation storage
python3 scripts/train.py \
  --config configs/atari/seaquest_single_path.yaml \
  --memory-mode safe \
  --obs_storage ram \
  --full_batch_chunk_size 8192 \
  --fvp_subsample_fraction 0.1 \
  --device cuda \
  --progress-mode notebook \
  --overwrite

# Force the original full-memory / full-batch path
python3 scripts/train.py \
  --config configs/atari/seaquest_single_path.yaml \
  --memory-mode standard \
  --progress-mode notebook \
  --overwrite
```

### 3) Aggregate one method over seeds

```bash
python3 scripts/aggregate_results.py \
  --runs-root outputs/swimmer_single_path \
  --metric train_return_mean \
  --x-axis iteration
```

### 4) Compare several methods on the same environment

```bash
python3 scripts/aggregate_results.py \
  --runs-root outputs/swimmer_single_path \
  --runs-root outputs/swimmer_natural_pg \
  --runs-root outputs/swimmer_random \
  --runs-root outputs/swimmer_ppo_clip \
  --compare \
  --metric train_return_mean \
  --x-axis iteration
```

### 5) Evaluate a saved checkpoint

```bash
python3 scripts/evaluate.py \
  --config configs/mujoco/swimmer_single_path.yaml \
  --checkpoint outputs/swimmer_single_path/seed_0/checkpoints/epoch_0050.pt \
  --episodes 10 \
  --device cpu
```

---

## Colab helper notebook

A step-by-step Colab usage notebook is included here:

- `notebooks/colab_quickstart.ipynb`

It shows how to:
- mount Google Drive
- copy the repo from Drive to `/content/trpo`
- install dependencies in the Colab runtime
- verify CUDA
- run locomotion and Atari jobs
- store outputs on local SSD during execution
- copy results back to Google Drive

If you only want the shortest Colab workflow, the notebook is the quickest reference.

---

## Quick usage model

The repository is easiest to think about in four layers:

1. **Config**: choose a YAML config under `configs/`
2. **Method**: TRPO / NPG / PPO / random
3. **Runtime mode**: memory mode, progress mode, device, chunk size, etc.
4. **Analysis**: aggregate seed runs, compare methods, export plots and CSV summaries

Typical workflow:

1. install dependencies
2. choose a config
3. train runs into `outputs/<run_name>/seed_<k>/`
4. aggregate over seeds for one method
5. compare methods on the same environment
6. optionally evaluate a saved checkpoint

---

## Installation

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

The editable install is strongly recommended so that `scripts/train.py` always imports the code from the current repo instead of an older installed package.

---

## Atari setup

### Install ALE + ROMs

```bash
python3 -m pip install --upgrade ale-py "autorom[accept-rom-license]"
AutoROM --accept-license
```

### Quick verification

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

## Methods and variants

### Methods you can pass via `--method`

| Method name | What it is | Main use in this repo | Typical config examples |
|---|---|---|---|
| `trpo` | Trust Region Policy Optimization | Main theory-heavy method, especially single-path TRPO | `configs/mujoco/swimmer_single_path.yaml`, `configs/atari/seaquest_single_path.yaml` |
| `natural_pg` / `npg` | Natural Policy Gradient | Locomotion comparison baseline | `configs/mujoco/swimmer_natural_pg.yaml` |
| `trpo_max_kl` | Max-KL TRPO variant | Small-scale comparison / ablation | `configs/mujoco/cartpole_trpo_max_kl.yaml` |
| `ppo` | Proximal Policy Optimization | Practical first-order comparison method | `configs/mujoco/swimmer_ppo_clip.yaml`, `configs/atari/seaquest_ppo_clip.yaml` |
| `random` | Random policy baseline | Sanity baseline / plotting baseline | `configs/mujoco/swimmer_random.yaml`, `configs/atari/pong_random.yaml` |

### Method variants

| Method | Variant control | Supported values | Meaning |
|---|---|---|---|
| `trpo` | usually determined by estimator | `paper_mc`, `mc_baseline`, `gae` | chooses how returns/advantages are formed |
| `natural_pg` | same estimator logic as TRPO family | `paper_mc`, `mc_baseline`, `gae` | same surrogate / Fisher geometry, different update rule |
| `trpo_max_kl` | fixed method | `paper_mc` recommended | uses max-KL acceptance logic (small-scale / special use) |
| `ppo` | `--method-variant` or config `method.variant` | `clip`, `kl_penalty` | clipped PPO surrogate or KL-penalty PPO |
| `random` | none | `default` | no learning, just rollout/evaluation |

### Estimator modes

These live in `algo.estimator` and matter mainly for the trainable policy-gradient methods.

| Estimator | Meaning | Uses value function? | Best use |
|---|---|---|---|
| `paper_mc` | paper-faithful Monte Carlo returns / Q-estimates for single-path TRPO | No | faithful TRPO single-path reproduction |
| `mc_baseline` | Monte Carlo returns with learned value baseline | Yes | lower-variance actor-critic style experiments |
| `gae` | Generalized Advantage Estimation | Yes | PPO and modernized actor-critic training |

### Where these live in code

| Concept | File |
|---|---|
| TRPO / NPG method wrappers | `trpo_repro/methods/trpo_method.py` |
| PPO method wrapper | `trpo_repro/methods/ppo_method.py` |
| Random baseline | `trpo_repro/methods/random_policy.py` |
| TRPO optimizer logic | `trpo_repro/algos/trpo.py` |
| PPO optimizer logic | `trpo_repro/algos/ppo.py` |
| Advantage / return utilities | `trpo_repro/algos/advantages.py` |
| Policy networks | `trpo_repro/models/policies.py` |
| Value networks | `trpo_repro/models/value_functions.py` |

---

## Runtime controls and CLI overrides

### Core training CLI

`train.py` supports these important overrides:

| Flag | What it changes | Notes |
|---|---|---|
| `--config` | YAML config path | required |
| `--seed` | `train.seed` | changes run seed |
| `--method` | `method.name` | switch methods without editing YAML |
| `--method-variant` / `--method_variant` | `method.variant` | mainly useful for PPO |
| `--device` | training device | `cpu`, `cuda`, etc. |
| `--output-dir` | where outputs go | useful for Colab SSD runs |
| `--overwrite` | reset existing run dir | deletes previous contents of the target run dir |
| `--memory-mode` | `train.memory_mode` | `standard` or `safe` |
| `--progress-mode` | `train.progress_mode` | `auto`, `terminal`, `notebook`, `off` |
| `--obs-storage` / `--obs_storage` | `train.obs_storage` | `auto`, `ram`, `memmap` |
| `--full-batch-chunk-size` / `--full_batch_chunk_size` | `algo.full_batch_chunk_size` | controls chunked TRPO safe-mode batch size |
| `--fvp-subsample-fraction` / `--fvp_subsample_fraction` | `algo.fvp_subsample_fraction` | optional Fisher-vector-product subsampling |

### What `--overwrite` means

When you pass `--overwrite`, the target output directory is cleared and recreated. This is useful when rerunning the same config/seed combination and you do **not** want mixed logs from old and new runs.

---

## Memory mode, observation storage, chunking, and FVP subsampling

### `train.memory_mode`

| Value | Meaning | When to use it |
|---|---|---|
| `standard` | original full-memory / full-batch path | locomotion, small experiments, high-RAM machines |
| `safe` | memory-safe path for large batches; uses chunked consumption and storage controls | Atari on laptops or constrained machines |

### `train.obs_storage`

| Value | Meaning | Notes |
|---|---|---|
| `auto` | choose based on memory mode | `ram` for standard, `memmap` for safe by default |
| `ram` | keep rollout observations in RAM | good for Colab high-RAM runs |
| `memmap` | disk-backed observation storage | good for low-RAM machines |

### `algo.full_batch_chunk_size`

Controls chunk size for the safe-mode full-batch computations. Larger chunks are usually faster **until** they become memory/bandwidth bound. Typical values to try on Atari are `4096`, `8192`, and sometimes `16384`.

### `algo.fvp_subsample_fraction`

Optional Fisher-vector-product subsampling for TRPO/NPG-style second-order updates.

- unset / `null` → use the full batch for Fisher-vector products
- `0.1` → use 10% of the batch
- `10` → also treated as 10%

This affects the **Fisher metric computation**, not the policy objective itself.

---

## Progress bar behavior

`train.progress_mode` controls how progress bars render.

| Mode | Best use |
|---|---|
| `auto` | default; chooses notebook-safe or terminal-safe behavior automatically |
| `terminal` | local shell / terminal runs |
| `notebook` | Jupyter / Colab |
| `off` | long runs where you want minimum UI overhead |

For long Colab Atari runs, `off` is often the cleanest option.

---

## Output structure

Each run writes to:

```text
outputs/<run_name>/seed_<seed>/
├── checkpoints/
├── config_resolved.yaml
├── metrics.csv
├── metrics.jsonl
└── run_metadata.json
```

### What each file is for

| File | Purpose |
|---|---|
| `config_resolved.yaml` | exact resolved config actually used for the run |
| `run_metadata.json` | method/environment/seed/runtime summary |
| `metrics.csv` | epoch-wise metrics in tabular form |
| `metrics.jsonl` | same information as line-delimited JSON |
| `checkpoints/` | periodic saved model checkpoints |

Comparison plots are written under:

```text
outputs/comparisons/<env_slug>/
```

---

## Important logged metrics

These are the most useful columns in `metrics.csv`.

### Generic rollout / performance metrics

| Metric | Meaning |
|---|---|
| `iteration` / `epoch` | training iteration index |
| `env_steps` | cumulative environment steps seen so far |
| `batch_env_steps` | steps collected in the current epoch |
| `episodes_in_batch` | complete episodes used in the epoch batch |
| `train_return_mean` | mean episodic return over episodes collected this epoch |
| `train_return_std` | standard deviation of episodic return this epoch |
| `train_len_mean` | mean episode length this epoch |
| `ep_return_mean`, `ep_return_std`, `ep_len_mean` | aliases / legacy-compatible names for the same quantities |

### TRPO / NPG style optimization metrics

| Metric | Meaning |
|---|---|
| `policy_loss_before`, `policy_loss_after` | surrogate objective before/after update |
| `entropy` | policy entropy for the batch |
| `approx_kl` | estimated KL change after update |
| `line_search_success` | whether TRPO line search accepted a step |
| `cg_norm` | size / norm-related diagnostic from conjugate gradient |

### PPO-specific metrics

| Metric | Meaning |
|---|---|
| `clip_fraction` | fraction of updates where the PPO ratio was clipped |
| `kl_coef` | KL penalty coefficient (relevant for `kl_penalty` PPO) |
| `value_loss_before`, `value_loss_after` | value-function fitting diagnostics |

### Timing metric

| Metric | Meaning |
|---|---|
| `wall_time_sec` | total epoch wall-clock time |

### About `train_return_mean`

This is the main metric you will most often plot. It is the **mean raw episodic return** collected in that epoch. It is **not** advantage-normalized, reward-normalized, or otherwise rescaled by the logging layer.

---

## Aggregation and plotting

### Aggregate one method over seeds

```bash
python3 scripts/aggregate_results.py \
  --runs-root outputs/swimmer_single_path \
  --metric train_return_mean \
  --x-axis iteration
```

### Compare multiple methods on the same environment

```bash
python3 scripts/aggregate_results.py \
  --runs-root outputs/swimmer_single_path \
  --runs-root outputs/swimmer_natural_pg \
  --runs-root outputs/swimmer_ppo_clip \
  --compare \
  --metric train_return_mean \
  --x-axis iteration
```

### Important options

| Flag | Meaning |
|---|---|
| `--runs-root` | root directory for a method run (`outputs/<run_name>` or specific `seed_*` dir); can be repeated |
| `--metric` | metric column to plot, e.g. `train_return_mean` |
| `--x-axis` | `iteration`, `epoch`, or `env_steps` |
| `--compare` | compare multiple methods on one environment |
| `--labels` | optional custom labels |
| `--save` | explicit output path for the plot |
| `--title` | custom plot title |
| `--allow-legacy-runs` | allow runs that do not have `run_metadata.json` |

### Same-environment safety check

In compare mode, the script checks that all runs have the same `env_id`. This helps prevent accidental comparison of different tasks on one plot.

---

## Evaluation

### Example

```bash
python3 scripts/evaluate.py \
  --config configs/mujoco/swimmer_single_path.yaml \
  --checkpoint outputs/swimmer_single_path/seed_0/checkpoints/epoch_0050.pt \
  --episodes 10 \
  --device cpu \
  --deterministic
```

### Evaluation options

| Flag | Meaning |
|---|---|
| `--config` | config used to rebuild the env/model |
| `--checkpoint` | checkpoint to load |
| `--episodes` | number of evaluation episodes |
| `--device` | `cpu` or `cuda` |
| `--seed` | evaluation seed |
| `--deterministic` | use deterministic actions where supported |

---

## Reproduce-paper helper script

The repo also includes:

```bash
python3 scripts/reproduce_paper.py --suite mujoco --seeds 0 1 2 3 4
python3 scripts/reproduce_paper.py --suite atari --seeds 0
```

This is a convenience launcher for a fixed suite of configs.

---

## Colab workflow (short version)

See the full notebook in `notebooks/colab_quickstart.ipynb`.

### Typical pattern

1. keep the repo on Google Drive
2. copy it to `/content/trpo` at the start of the session
3. run training from `/content/trpo`
4. store outputs on `/content/...` while running
5. copy outputs back to Drive when done

### Minimal Colab commands

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
rm -rf /content/trpo
cp -r "/content/drive/MyDrive/Colab Notebooks/839/trpo" /content/trpo
cd /content/trpo
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')
```

```bash
python3 scripts/train.py \
  --config configs/atari/seaquest_single_path.yaml \
  --memory-mode safe \
  --obs_storage ram \
  --full_batch_chunk_size 8192 \
  --fvp_subsample_fraction 0.1 \
  --device cuda \
  --progress-mode off \
  --output-dir /content/trpo_runs/seaquest_single_path/seed_0 \
  --overwrite
```

```bash
mkdir -p "/content/drive/MyDrive/Colab Notebooks/839/trpo_outputs"
cp -r /content/trpo_runs "/content/drive/MyDrive/Colab Notebooks/839/trpo_outputs/"
```

---

## Available configs in this repo

### Mujoco / classic control / locomotion

| Config |
|---|
| `configs/mujoco/cartpole_linear.yaml` |
| `configs/mujoco/cartpole_random.yaml` |
| `configs/mujoco/cartpole_natural_pg.yaml` |
| `configs/mujoco/cartpole_trpo_max_kl.yaml` |
| `configs/mujoco/cartpole_ppo_clip.yaml` |
| `configs/mujoco/swimmer_single_path.yaml` |
| `configs/mujoco/swimmer_random.yaml` |
| `configs/mujoco/swimmer_natural_pg.yaml` |
| `configs/mujoco/swimmer_ppo_clip.yaml` |
| `configs/mujoco/swimmer_ppo_kl_penalty.yaml` |
| `configs/mujoco/hopper_single_path.yaml` |
| `configs/mujoco/hopper_random.yaml` |
| `configs/mujoco/hopper_natural_pg.yaml` |
| `configs/mujoco/hopper_ppo_clip.yaml` |
| `configs/mujoco/walker2d_single_path.yaml` |
| `configs/mujoco/walker2d_random.yaml` |
| `configs/mujoco/walker2d_natural_pg.yaml` |
| `configs/mujoco/walker2d_ppo_clip.yaml` |
| `configs/mujoco/walker_ppo_clip.yaml` |
| `configs/mujoco/walker_natural_pg.yaml` |
| `configs/mujoco/walker_random.yaml` |
| legacy modernized configs: `configs/mujoco/cartpole_mc_baseline.yaml`, `configs/mujoco/cartpole_modern_gae.yaml` |

### Atari

| Config |
|---|
| `configs/atari/beamrider_single_path.yaml` |
| `configs/atari/breakout_single_path.yaml` |
| `configs/atari/enduro_single_path.yaml` |
| `configs/atari/pong_single_path.yaml` |
| `configs/atari/qbert_single_path.yaml` |
| `configs/atari/seaquest_single_path.yaml` |
| `configs/atari/spaceinvaders_single_path.yaml` |
| `configs/atari/pong_random.yaml` |
| `configs/atari/beamrider_ppo_clip.yaml` |
| `configs/atari/breakout_ppo_clip.yaml` |
| `configs/atari/enduro_ppo_clip.yaml` |
| `configs/atari/pong_ppo_clip.yaml` |
| `configs/atari/qbert_ppo_clip.yaml` |
| `configs/atari/seaquest_ppo_clip.yaml` |
| `configs/atari/spaceinvaders_ppo_clip.yaml` |
| `configs/atari/seaquest_ppo_kl_penalty.yaml` |

---

## Repository layout

```text
trpo_repro/
├── algos/          # optimization logic (TRPO, PPO, advantages, CG, line search)
├── data/           # rollout storage and trajectory buffers
├── envs/           # env factories and wrappers for Atari / Mujoco
├── methods/        # method wrappers registered by --method
├── models/         # policies, CNN/MLP bodies, value functions
└── utils/          # general utilities + torch utilities
```

### Important files to know

| File | Why you care |
|---|---|
| `scripts/train.py` | main experiment launcher |
| `scripts/aggregate_results.py` | plotting / seed aggregation / method comparison |
| `scripts/evaluate.py` | evaluate checkpoints |
| `scripts/reproduce_paper.py` | convenience suite launcher |
| `trpo_repro/runner.py` | main training loop |
| `trpo_repro/algos/trpo.py` | TRPO / NPG optimization logic |
| `trpo_repro/algos/ppo.py` | PPO optimization logic |
| `trpo_repro/data/buffer.py` | rollout buffer implementation |
| `trpo_repro/algos/advantages.py` | return / advantage computations |
| `trpo_repro/models/policies.py` | Gaussian / categorical policies |
| `trpo_repro/models/value_functions.py` | value-function models |

---

## Notes and caveats

- `paper_mc` is the repo’s paper-faithful TRPO single-path mode.
- `safe` memory mode changes **storage and execution order**, not the objective being optimized.
- Atari on laptops generally benefits from `memory_mode: safe`.
- Colab high-RAM runs often work well with `memory_mode: safe`, `obs_storage: ram`, and a tuned `full_batch_chunk_size`.
- `fvp_subsample_fraction` is an optional TRPO/NPG acceleration and should be reported if you use it in final experiments.
- The repo still contains legacy / modernized estimator options (`mc_baseline`, `gae`) even if your report focuses mostly on `paper_mc` for TRPO.

