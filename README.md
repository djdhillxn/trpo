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

- start with [`docs/README.md`](docs/README.md) for the documentation and result-artifact map
- open [`docs/aggregate_results_plots.ipynb`](docs/aggregate_results_plots.ipynb) for the notebook that regenerates the plots, tables, and run audits from `outputs/`
- browse [`docs/figures_and_summaries/`](docs/figures_and_summaries/) for saved PDF figures and CSV summaries
- read [`docs/instruction_manual.md`](docs/instruction_manual.md) for the full CLI reference, config inventory, and operational details

## Repository Map

| Path | Purpose |
|---|---|
| `scripts/train.py` | main training entry point |
| `scripts/evaluate.py` | checkpoint evaluation |
| `scripts/aggregate_results.py` | seed aggregation, plotting, report tables, run audits |
| `configs/` | YAML experiment configs |
| `trpo_repro/` | algorithms, methods, models, rollouts, and utilities |
| `outputs/` | public run metrics, configs, metadata, summaries, and local checkpoints ignored by git |
| `docs/README.md` | documentation overview and result-artifact workflow |
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

The long-form operational documentation lives in
[`docs/instruction_manual.md`](docs/instruction_manual.md). It preserves the
detailed CLI reference, config inventory, output schemas, aggregation recipes,
runtime notes, and paper-alignment notes while keeping this README focused on the
public project overview.

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
## RLHF extension: Qwen2.5 + HelpSteer3 + token-level PPO

This repository now includes an application-oriented RLHF extension under `trpo_repro/rlhf/`. The extension adapts the same conservative policy-optimization story from the original TRPO/PPO project to LLM post-training: supervised fine-tuning, pairwise reward modeling, KL-controlled token-level PPO using LoRA adapters, and policy-suite evaluation of Base/SFT/PPO responses.

The final long-context run uses Qwen2.5-0.5B-Instruct with HelpSteer3, 4096-token SFT/reward-model training, and 3072-token PPO prompts with 512-token PPO rollouts. The primary evaluation allows up to 1024 new tokens at inference time while keeping the prompt-plus-response budget at 4096. Across all 2017 validation prompts, cap-hit rates fall from roughly 30% in the earlier 512-token suite to 8.08% for Base, 10.16% for SFT, and 11.60% for PPO. Base still wins most often; PPO beats Base on 38.92% of prompts and has a mean reward delta of `-0.2137`, while PPO has a small positive mean delta of `+0.0343` against SFT.

The longer evaluation also exposes failures that the shorter cap could hide. PPO has the highest measured repetition rate, and manual review finds both useful local improvements and severe reward-model mismatches. The result is therefore evidence for a stable, inspectable RLHF pipeline, not a claim that PPO globally improves this instruction model.

Start here: [`docs/rlhf_readme.md`](docs/rlhf_readme.md).

Additional RLHF notes:

- [`docs/rlhf_experiments.md`](docs/rlhf_experiments.md): experiment timeline, failed runs, final metrics.
- [`docs/rlhf_evaluation_history.md`](docs/rlhf_evaluation_history.md): primary 1024-token results and archived 512-token baseline.
- [`docs/rlhf_qualitative_audit.md`](docs/rlhf_qualitative_audit.md): full-suite diagnostics and manually reviewed examples.
- [`docs/rlhf_curation_guide.md`](docs/rlhf_curation_guide.md): how to reproduce and extend the qualitative review.
- [`docs/rlhf_future_work.md`](docs/rlhf_future_work.md): research directions motivated by the observed failures.

Quick commands:

```bash
pip install -r requirements-rlhf.txt
pip install -e .
python scripts/rlhf_train_sft_policy.py --config configs/rlhf/qwen25_05b_helpsteer3_sft.yaml
python scripts/rlhf_train_reward_model.py --config configs/rlhf/qwen25_05b_helpsteer3_reward.yaml
python scripts/rlhf_train_ppo.py --config configs/rlhf/qwen25_05b_helpsteer3_ppo.yaml
python scripts/rlhf_evaluate_policy_suite.py --config configs/rlhf/qwen25_05b_helpsteer3_eval_suite.yaml
python scripts/rlhf_audit_policy_suite.py \
  --eval-dir outputs/rlhf/qwen25_05b_helpsteer3_eval_suite_4096_ep2_u400_eval1024 \
  --baseline-dir outputs/rlhf/qwen25_05b_helpsteer3_eval_suite_4096_ep2_u400 \
  --selection-file configs/rlhf/qwen25_05b_helpsteer3_eval1024_curation.json
```
