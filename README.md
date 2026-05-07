# TRPO / PPO Policy Optimization Experiments

This repository contains the code, completed run artifacts, and analysis notebooks
for an empirical comparison of Trust Region Policy Optimization (TRPO), Natural
Policy Gradient (NPG), and Proximal Policy Optimization (PPO-Clip). The project
focuses on reproducing the core behavior of single-path TRPO, comparing it against
NPG and PPO baselines, and regenerating the final plots and summary tables directly
from saved experiment outputs.

The raw run folders live in [`outputs/`](outputs/). They include metrics,
resolved configs, launch metadata, runtime summaries, and environment snapshots.
Large PyTorch checkpoint files are intentionally ignored by git so the public
repository stays readable and practical to clone.

## Completed Experiments

| Suite | Tasks | Methods | Seeds in `outputs/` | Result artifacts |
|---|---|---|---|---|
| MuJoCo locomotion | `Hopper-v5`, `Swimmer-v5`, `Walker2d-v5` | TRPO, NPG, PPO-Clip | 3 seeds per method/task | training curves and CSV summaries |
| Atari | `BeamRider-v5`, `Enduro-v5`, `Pong-v5`, `Qbert-v5`, `Seaquest-v5`, `SpaceInvaders-v5` | TRPO, PPO-Clip | 1 seed per method/game | training curves and CSV summaries |
| Hopper NPG ablation | `Hopper-v5` | NPG 1x, 3x, and 9x step sizes, with TRPO/PPO overlay | 3 seeds per NPG variant | ablation curves and CSV summaries |

The fastest way to inspect the results is:

- open [`docs/aggregate_results_plots.ipynb`](docs/aggregate_results_plots.ipynb) for the notebook that regenerates the plots, tables, and run audits from `outputs/`
- browse [`docs/figures_and_summaries/`](docs/figures_and_summaries/) for saved PDF figures and CSV summaries
- read [`docs/instruction_manual.md`](docs/instruction_manual.md) for the full CLI reference, config inventory, and operational details

## What Is Included

- TRPO single-path training with analytic KL-Hessian / Fisher-vector products
- Natural Policy Gradient using the same local policy geometry without line search
- Empirical-FIM TRPO ablation path
- PPO-Clip and PPO-KL-Penalty variants
- Random policy baseline
- Single-process and parallel rollout collection
- Laptop-safe memory mode for large Atari TRPO/NPG runs
- MuJoCo and Atari config suites
- Checkpoint evaluation, aggregation, plotting, and report audit utilities
- Colab-friendly notebooks for running experiments and inspecting results

## Repository Map

| Path | Purpose |
|---|---|
| `scripts/train.py` | main training entry point |
| `scripts/evaluate.py` | checkpoint evaluation |
| `scripts/aggregate_results.py` | seed aggregation, plotting, report tables, run audits |
| `configs/` | YAML experiment configs |
| `trpo_repro/` | algorithms, methods, models, rollouts, and utilities |
| `outputs/` | public run metrics, configs, metadata, summaries, and local checkpoints ignored by git |
| `docs/instruction_manual.md` | detailed manual and CLI reference |
| `docs/colab_runmanager.ipynb` | Colab run-management notebook |
| `docs/aggregate_results_plots.ipynb` | plotting, PDF export, tables, and report sanity checks |
| `docs/figures_and_summaries/` | saved PDF figures and CSV summaries generated from the completed runs |

## Quick Start

Install the Python dependencies:

```bash
python3 -m pip install -r requirements.txt
```

For Atari, install and accept the ROM license:

```bash
python3 -m pip install --upgrade ale-py "autorom[accept-rom-license]"
AutoROM --accept-license
```

Run a MuJoCo TRPO experiment:

```bash
python3 scripts/train.py \
  --config configs/mujoco/swimmer_single_path.yaml \
  --overwrite
```

Run a PPO comparison:

```bash
python3 scripts/train.py \
  --config configs/mujoco/swimmer_ppo_clip.yaml \
  --overwrite
```

Run on CUDA with parallel rollout workers:

```bash
python3 scripts/train.py \
  --config configs/atari/seaquest_single_path.yaml \
  --device cuda \
  --num-workers 4 \
  --memory-mode safe \
  --obs-storage ram \
  --full-batch-chunk-size 8192 \
  --overwrite
```

Aggregate one method over seeds:

```bash
python3 scripts/aggregate_results.py \
  --runs-root outputs/swimmer_single_path \
  --metric train_return_mean \
  --x-axis epoch \
  --summary
```

Compare methods on one environment:

```bash
python3 scripts/aggregate_results.py \
  --runs-root outputs/swimmer_single_path \
  --runs-root outputs/swimmer_natural_pg \
  --runs-root outputs/swimmer_ppo_clip \
  --labels TRPO NPG PPO \
  --metric train_return_mean \
  --x-axis epoch \
  --formats pdf \
  --summary
```

Evaluate a checkpoint:

```bash
python3 scripts/evaluate.py \
  --config configs/mujoco/swimmer_single_path.yaml \
  --checkpoint outputs/swimmer_single_path/seed_0/checkpoints/epoch_0050.pt \
  --episodes 10 \
  --device cpu
```

## Outputs And Analysis

Training writes runs under:

```text
outputs/<run_name>/seed_<k>/
```

Each seed directory contains the resolved configs, launch metadata, runtime and
CUDA environment snapshots, per-epoch metrics, summaries, failures if any, and
checkpoints. These artifacts are intentionally verbose so that report tables and
plots can be regenerated after the fact.

Use `docs/aggregate_results_plots.ipynb` to:

- plot training curves with seed uncertainty bands
- save report-ready PDF figures
- regenerate locomotion, Atari, and NPG ablation tables
- print hyperparameters, CUDA/runtime details, and model parameter counts

Saved public-facing figures and summary CSVs are collected under
[`docs/figures_and_summaries/`](docs/figures_and_summaries/).

## Detailed Documentation

The long-form operational documentation now lives in:

- [`docs/instruction_manual.md`](docs/instruction_manual.md)

That manual preserves the detailed material formerly in this README and the old
CLI reference, including:

- full CLI flag tables
- method and estimator variants
- runtime, memory, progress, Fisher/FVP, and PPO scheduling controls
- Colab workflows
- output-file and metric inventories
- aggregation recipes
- config inventory
- code map
- paper-alignment notes
- recommended freeze-and-run practice

## Project Framing

This repo is designed for empirical comparison rather than chasing a single
state-of-the-art number. The completed experiments support three core use cases:

1. paper-faithful single-path TRPO reproduction,
2. controlled comparisons against NPG and empirical-FIM variants,
3. practical PPO baselines with modern advantage estimation and minibatch updates.

For the final report workflow, the intended loop is simple: run experiments,
preserve the output folders, regenerate figures and tables from those folders,
and keep the top-level README readable enough for a new reader to understand the
project before diving into the manual.

## References

Papers:

- John Schulman, Sergey Levine, Philipp Moritz, Michael I. Jordan, and Pieter Abbeel. "Trust Region Policy Optimization." ICML 2015. https://arxiv.org/abs/1502.05477
- John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. "Proximal Policy Optimization Algorithms." arXiv 2017. https://arxiv.org/abs/1707.06347
- Sham M. Kakade. "A Natural Policy Gradient." NeurIPS 2001. https://papers.nips.cc/paper/2073-a-natural-policy-gradient

Reference implementations:

- John Schulman's `joschu/modular_rl`: TRPO and related policy-gradient algorithms. https://github.com/joschu/modular_rl
- OpenAI Baselines: TensorFlow implementations of PPO, TRPO, and other RL baselines. https://github.com/openai/baselines
- Ray RLlib: scalable reinforcement-learning library with PPO and related policy-optimization implementations. https://github.com/ray-project/ray/tree/master/rllib
- Ilya Kostrikov's `ikostrikov/pytorch-trpo`: PyTorch TRPO implementation. https://github.com/ikostrikov/pytorch-trpo
