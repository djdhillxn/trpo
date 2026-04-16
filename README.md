# TRPO paper reproduction (modular PyTorch codebase)

This repository is a clean, modular implementation of **Trust Region Policy Optimization (TRPO)** aimed at reproducing the **single-path** results from the original TRPO paper as closely as is practical with a modern Python stack.

It targets the benchmark families used in the paper:

- continuous-control locomotion tasks: **Swimmer, Hopper, Walker**
- Atari games: **Beam Rider, Breakout, Enduro, Pong, Q*bert, Seaquest, Space Invaders**
- a small **CartPole** config is also included as a simple sanity-check baseline

## Important scope note

The original paper evaluates **single-path TRPO** and **vine TRPO**. This codebase implements the **single-path algorithm fully** and includes environment snapshot hooks that make it straightforward to extend to vine-style branching when the simulator exposes reliable state clone / restore operations.

I did **not** ship a full vine trainer here because exact vine reproduction depends heavily on simulator-specific state-reset semantics and common-random-number handling. With modern Gymnasium / ALE / MuJoCo stacks, that usually deserves a separate, environment-specific engineering pass.

So the repo is designed to be:

1. research-clean,
2. easy to inspect and modify,
3. close to the paper’s practical TRPO core, and
4. honest about where exact paper parity is not guaranteed.

## What is implemented

- on-policy trajectory collection
- Monte Carlo returns-to-go (paper-aligned default)
- optional GAE if you want a more modern baseline
- TRPO update with:
  - surrogate objective
  - mean KL trust region
  - conjugate gradient natural-gradient step
  - Hessian-vector product via autograd
  - backtracking line search
- separate value function regression
- continuous Gaussian policies
- discrete categorical policies
- MLP policy for locomotion / low-dimensional control
- CNN policy for Atari
- YAML configs for the paper benchmark suite
- JSONL logging + CSV summary + checkpoints
- evaluation script
- suite runner for sweeping all paper tasks

## Repository layout

```text
trpo_repro_project/
├── configs/
│   ├── atari/
│   └── mujoco/
├── scripts/
├── trpo_repro/
│   ├── algos/
│   ├── data/
│   ├── envs/
│   ├── models/
│   └── utils/
├── pyproject.toml
└── requirements.txt
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### MuJoCo

For the locomotion experiments you need Gymnasium MuJoCo support installed. The requirements file requests the extra, but you still need a working MuJoCo runtime.

### Atari

For Atari you need `ale-py` and ROMs. A common workflow is:

```bash
pip install autorom[accept-rom-license]
AutoROM --accept-license
```

## Quick start

### Swimmer (paper-style single-path TRPO)

```bash
python scripts/train.py --config configs/mujoco/swimmer_single_path.yaml
```

### Pong

```bash
python scripts/train.py --config configs/atari/pong_single_path.yaml
```

### Evaluate a checkpoint

```bash
python scripts/evaluate.py \
  --config configs/mujoco/swimmer_single_path.yaml \
  --checkpoint outputs/swimmer_single_path/checkpoints/epoch_0200.pt \
  --episodes 10
```

### Run the full MuJoCo or Atari suite

```bash
python scripts/reproduce_paper.py --suite mujoco
python scripts/reproduce_paper.py --suite atari
```

## Design decisions and how they map to the paper

### 1. Single-path TRPO is the primary target

The paper explicitly describes **single-path** as the model-free variant and **vine** as the variant that requires resetting to sampled states. In modern research code, the single-path version is the realistic baseline to reproduce first.

### 2. Paper-like benchmark configs are included

The YAML files use paper-like settings such as:

- `max_kl = 0.01`
- `gamma = 0.99`
- large batch sizes / simulator steps per iteration
- low-capacity MLP for continuous control
- Atari preprocessing consistent with the classic DeepMind-style pipeline

### 3. “Exact paper reproduction” is limited by environment drift

The original paper used older MuJoCo tasks and ALE conventions. Modern Gymnasium / ALE / MuJoCo environment versions are not byte-for-byte identical to those 2015 setups, so you should treat this repo as a **faithful modern reproduction attempt**, not a claim of exact numerical identity.

That matters especially for:

- reward scales
- termination criteria
- MuJoCo XML/model revisions
- ALE sticky actions / frameskip conventions
- action-set defaults

## Recommended reproduction procedure

### MuJoCo

Run 5 seeds each for:

- `configs/mujoco/swimmer_single_path.yaml`
- `configs/mujoco/hopper_single_path.yaml`
- `configs/mujoco/walker2d_single_path.yaml`

Example:

```bash
for seed in 0 1 2 3 4; do
  python scripts/train.py --config configs/mujoco/swimmer_single_path.yaml --seed $seed
done
```

### Atari

Run 1–5 seeds each depending on budget for:

- `beamrider_single_path.yaml`
- `breakout_single_path.yaml`
- `enduro_single_path.yaml`
- `pong_single_path.yaml`
- `qbert_single_path.yaml`
- `seaquest_single_path.yaml`
- `spaceinvaders_single_path.yaml`

## Output structure

Each run writes to:

```text
outputs/<run_name>/
├── checkpoints/
├── config_resolved.yaml
├── metrics.csv
└── metrics.jsonl
```

## Extending to vine TRPO

The code already includes `envs/snapshots.py`, which provides environment snapshot helpers for resettable simulators. A vine trainer can be added by:

1. collecting trunk rollouts,
2. saving state snapshots at selected trunk states,
3. branching rollouts from those states with sampled actions,
4. computing branch-based local Q estimates,
5. replacing the single-path surrogate estimator with the vine estimator.

## Practical notes

- Start with smaller `steps_per_epoch` for smoke tests.
- For serious MuJoCo reproduction use the paper-sized batch settings.
- For Atari, training is expensive; expect long wall-clock times.
- If you care about closest historical comparability, pin ROM/action/wrapper conventions carefully.

## License

Code in this repository is provided as example research code. Review dependency licenses separately.
