# Figures and Summaries

This folder contains the public-facing result artifacts generated from the run
folders in [`../../outputs/`](../../outputs/). The plots and CSVs are produced by
[`../aggregate_results_plots.ipynb`](../aggregate_results_plots.ipynb), which uses
the shared aggregation utilities in [`../../scripts/aggregate_results.py`](../../scripts/aggregate_results.py).

## Contents

| Folder | What it contains |
|---|---|
| [`locomotion/`](locomotion/) | PDF training curves and CSV summaries for `Hopper-v5`, `Swimmer-v5`, and `Walker2d-v5`, comparing TRPO, NPG, and PPO-Clip across three seeds. |
| [`atari/`](atari/) | PDF training curves and CSV summaries for the six Atari tasks, comparing TRPO and PPO-Clip for the completed seed-0 runs. |
| [`npg_ablation/`](npg_ablation/) | PDF and CSV outputs for the Hopper NPG step-size ablation, including the NPG-only comparison and the combined TRPO/NPG/PPO overlay. |

For each plotted comparison, the `.pdf` file is the report-ready figure, the main
`.csv` file contains the aggregated plotting data, and the `__summary.csv` file
contains the compact table used for sanity checks and report summaries.

The raw metrics, configs, launch metadata, runtime metadata, and environment
snapshots remain in [`../../outputs/`](../../outputs/). PyTorch checkpoint weights
are intentionally not tracked in git.
