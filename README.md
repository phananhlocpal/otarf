# OTARF

Code for the paper **Multivariate time series forecasting with incomplete data: an observation-anchored model for air quality monitoring networks** (model name: OTARF, Observation-gated Target-Anchored Residual Forecaster).

OTARF forecasts every channel of a monitoring network from its own history, expressed relative to the
last valid observation (the anchor) together with the observation mask and age. It then adds a signed
correction gathered by attention from the history segments of other channels that still contain
measurements. When no valid source exists, the output falls back to the local forecast.

This repository rebuilds the datasets, trains OTARF, its ablation variants and the four baselines
(official SOFTS, TimeFilter, iTransformer and CRIB code, trained with the same recipe; the first three with the same
observation information, CRIB with its authors' zero-filled input), scores outages of one target, of several targets and
of whole stations, and regenerates every table, figure and statistical decision of the paper.

## Layout

```
otarf/                  library
  core.py               OTARF as described in the paper, and its ablation variants
  variants.py           variant names -> model
  data.py               chronological windows, masks, ages, anchors
  augmentation.py       masked-history training view
  training.py           training recipe (observed-label MAE, masked view, checkpoint on validation nMAE)
  evaluate.py           external evaluator (every model is scored by this file)
  masking.py            controlled 6/24/96-hour outage of one target
  statistics.py         paired temporal block bootstrap
  baselines.py          wrappers around the unchanged official SOFTS / TimeFilter / iTransformer / CRIB code
  outages.py            several targets or whole stations out at once, hourly threshold events, naive references
  inspect.py            inference-only view of a trained OTARF: attention by source group, source knockout
  nonorm.py             switches off the window normalization of the official baselines (mechanism test)
  events.py             exceedance events on the 24-hour mean (PM2.5 / PM10)
configs/                hyper-parameters used in the paper and the locked baseline commits
scripts/                command-line entry points (see below)
```

## Environment

Python 3.10, PyTorch 2.3.1 and one CUDA GPU. OTARF needs well under 1 GB of GPU memory; the largest
baseline runs are TimeFilter on Beijing with mask and age variates (about 8.4 GB) and CRIB on Beijing (about 13.3 GiB).

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 1. Data

```bash
bash scripts/download_data.sh             # Beijing (UCI 501), London (KDD Cup 2018, Monash), Shanghai (UCI 394)
python scripts/prepare_data.py            # writes data/processed/<dataset>/
```

The Shanghai file sits inside a RAR archive in the UCI download; the script extracts it with `unrar`,
`unar` or `7z`. Preparation is deterministic. The processed files should have these SHA-256 values:

| File | SHA-256 |
|---|---|
| beijing_multisite/data.csv | `0bd9ec2540fc95d28613ca6da7ce1c00c538fc42abe116f71c169ba22023afb8` |
| beijing_multisite/observations.npz | `6df743f66412610d200751e31008207faed8f8829bd88b94d7534de77288dad5` |
| kdd2018_london/data.csv | `4d9c5b37a9cb9df3f0c62710349577d4d81bad4a4ad156a65a6bc8c0bf1c239d` |
| kdd2018_london/observations.npz | `07312b9749c4534102e879653fef2ea9f8668fd22f81f18eae142405a9c06f72` |
| shanghai_confirmation/data.csv | `e54c5dbf99d77a74d8e529268516dceaa4d774028d8503e7dd75a536a49343b7` |
| shanghai_confirmation/observations.npz | `1fa3ec9b791f446e17ec4d685dceea67b1b58a5992f3c7ff197aece513a47d0a` |

`data.csv` holds the forward-filled series used as model input; `observations.npz` keeps the original
values and the observation mask, which are used for the loss and for every metric.

## 2. Baselines

```bash
bash scripts/fetch_baselines.sh           # clones SOFTS, TimeFilter, iTransformer and CRIB into external/ at the locked commits
```

The baseline model code is never modified. Before each run the SHA-256 of every Python file of
SOFTS, TimeFilter, iTransformer and CRIB is checked against `configs/baselines.lock.json`.

## 3. Training and evaluation

One command trains one model on one dataset, horizon and seed, evaluates the best and last checkpoints on
validation and test, and scores the controlled 6, 24 and 96-hour outage of one target.

```bash
# OTARF and its variants
python scripts/train_otarf.py --dataset beijing_multisite --horizon 24 --seed 2021
python scripts/train_otarf.py --dataset beijing_multisite --horizon 24 --seed 2021 --variant no_cross

# Baselines as compared in the paper: official model code, OTARF training recipe, mask and log(1+age) variates
python scripts/train_baseline.py --model TimeFilter --recipe indicator --dataset kdd2018_london --horizon 24 --seed 2021

# Baselines with their own official trainers (optional; not used in the paper)
python scripts/train_baseline.py --model SOFTS --recipe official --dataset shanghai_confirmation --horizon 24 --seed 2021
```

Datasets: `beijing_multisite`, `kdd2018_london`, `shanghai_confirmation`. Horizons: 12, 24, 48. Seeds: 2021, 2022, 2023.

Variants (`--variant`):

| Name | Meaning |
|---|---|
| `otarf` | OTARF as in the paper (default) |
| `no_anchor` | history given as the filled standardized series; anchor removed from the base forecast and from all features |
| `no_cross` | cross-variable correction set to zero |
| `mean_pool` | attention replaced by a uniform mean over the same valid source segments |
| `state_blind` | mask, age and anchor validity replaced by constants at the input |
| `state_blind_no_aug` | as `state_blind`, trained on natural histories only |

Each run writes to `runs/<model>-<variant or recipe>/<dataset>/h<H>/s<seed>/`: `result.json`,
`learning_curve.json`, checkpoints, `best/` and `last/` metrics and per-origin errors, and `masked/`.

iTransformer, CRIB and the peak-weighted OTARF (H = 12, 24 and 48, same seeds) are trained with:

```bash
python scripts/train_extended_baseline.py --model iTransformer --dataset beijing_multisite --horizon 24 --seed 2021
python scripts/train_extended_baseline.py --model CRIB --dataset kdd2018_london --horizon 24 --seed 2021
python scripts/train_otarf.py --dataset beijing_multisite --horizon 24 --seed 2021 --peak
```

iTransformer receives the mask and log(1+age) of every channel as extra variates. CRIB keeps its authors' input (the
series multiplied by the mask, no age) because its attention runs jointly over all patch x variate tokens, and uses a
96-hour window instead of the authors' 24 hours. `--peak` multiplies the error of observed labels at or above the train
95th percentile of their target by 3 (`configs/otarf.json`, key `peak`).

To launch every run behind the paper (153 training runs plus inference; sections can be selected):

```bash
bash scripts/reproduce_all.sh                          # otarf baselines ablation inspect extended outages nonorm events24 (add "official" for the optional runs)
bash scripts/reproduce_all.sh otarf baselines ablation # the runs behind the result tables only
```

## 4. Tables, figures and decisions

```bash
python scripts/inspect_otarf.py --dataset beijing_multisite   # inference only; repeat for each dataset
python scripts/summarize.py        # results/*.csv and results/decisions.json (bootstrap: about 5 minutes)
python scripts/make_figures.py     # results/figures/ (Figures 1, 3 and 4)
python scripts/draw_overview.py    # schematic of the model (Figure 2)
```

| Paper | Output |
|---|---|
| Figure 1: outage lengths in the three datasets | `figures/gap_lengths` (from `data/processed`) |
| Figure 2: schematic of the model | `scripts/draw_overview.py` |
| Natural histories, mean ± SD | `results/table_natural.csv` |
| Natural histories by the target's own history state | `results/table_history_state.csv` |
| 24 h and 96 h outages with the last-value reference; Figure 3 | `results/table_masked.csv`, `figures/masking_curves` |
| Ablation table (OTARF lower than each variant, %, validation) | `results/table_ablation_paper.csv` (test sMAE: `results/table_ablation.csv`) |
| Labels above the train 95th percentile, rising and stable labels | `results/table_tail.csv`, `results/table_ramp_stable.csv` |
| MAE in concentration units, H = 24 (and above / below the 95th percentile) | `results/table_raw_h24.csv`, `results/table_pollutant_tail_h24.csv` |
| Cost at H = 24 | `results/table_cost_h24.csv` |
| Anchor age under the 96-hour outage | `results/table_anchor_age96.csv` |
| Where the cross-variable branch takes its information from (attention by source group, knockout) | `results/table_sources.csv` (after `scripts/inspect_otarf.py`) |
| Figure 4: 96-hour case study (origins chosen by `scripts/select_cases.py`, shipped in `configs/case_selection.json`) | `figures/case_96h` (after `scripts/inspect_otarf.py`) |
| Paired temporal block bootstrap | `results/comparison.csv`, `results/ablation_comparison.csv`, `results/bootstrap_*.csv` |
| Pre-specified decision rules | `results/decisions.json` |

`summarize.py` was checked on the stored arrays of the paper's runs: all 717 table and analysis cells, the 153 source-analysis cells, the case selection,
144 comparison rows, 90 ablation rows and every decision match the paper.

### Extended comparison: outages of several channels, threshold events, naive references

```bash
python scripts/evaluate_outages.py --model OTARF --dataset beijing_multisite --seed 2021   # inference only, needs the finished run
python scripts/evaluate_outages.py --model persistence --dataset beijing_multisite --horizon 12
python scripts/summarize_extensions.py        # results/extensions/*.csv and *_decision.json
```

| Paper | Output |
|---|---|
| iTransformer and CRIB against OTARF (H = 24, natural and 6/24/96 h outage) and the pre-specified rule | `results/extensions/A_new_baselines.csv`, `A_decision.json` |
| Peak-weighted loss against OTARF (validation rule, test as secondary) | `results/extensions/C_peak_vs_otarf.csv`, `C_decision.json` |
| One target, 25/50% of the targets, 25/50% of the stations out for 24 or 96 h | `results/extensions/B_multi_outage.csv`, `B_decision.json` |
| Hourly threshold events (precision, recall, F1) | `results/extensions/E4_events.csv` |
| Persistence and seasonal naive references | `results/extensions/E0_naive.csv` |
| sMAE by the length of the real outage at the end of the history | `results/extensions/E1_real_gaps.csv` |

Thresholds for the event analysis are applied to hourly values and are not the official AQI: PM2.5 75 and PM10 150
ug/m3 (grade II 24-hour limits of the Chinese national standard GB 3095-2012) for Beijing and Shanghai; the first value
of the 'Moderate' and 'High' bands of the UK Daily Air Quality Index (PM2.5 36 and 54, PM10 51 and 76 ug/m3) for London.
In a station-wise outage every channel of the chosen stations (pollutants and meteorology) is removed; Shanghai has no
station structure and only has the target-wise scenarios.

The extension code was checked against the stored arrays of the paper's runs: the outage scenarios of `otarf/outages.py`
reproduce the per-origin arrays of the persistence and seasonal references for every scenario of London and Shanghai; the
iTransformer, CRIB and peak-weighted OTARF models built by this code load the trained checkpoints strictly and reproduce
the stored test sMAE (London, seed 2021, natural history and 96-hour outage); and the peak thresholds equal the ones
saved by the original runs. Training itself uses the unchanged `fit` of `otarf/training.py`.

### Final comparison: five models, window normalization switched off, 24-hour-mean events

```bash
# The four comparison models with their window normalization switched off (H = 24): SOFTS and iTransformer through the
# official flag use_norm=0, TimeFilter and CRIB by replacing their normalization module with an identity (otarf/nonorm.py).
python scripts/train_extended_baseline.py --model TimeFilter --dataset kdd2018_london --horizon 24 --seed 2021 --no-window-norm
# Exceedance events on the 24-hour mean (inference only, H = 24 runs must be finished)
python scripts/evaluate_events24.py --model OTARF --dataset beijing_multisite --seed 2021
python scripts/summarize_v6.py      # results/v6/*.csv and all_horizons_decision.json
```

| Paper | Output |
|---|---|
| Tables 3 and 4: natural history and 6/24/96 h outage, five models, H = 12/24/48, with the decision rule | `results/v6/all_horizons.csv`, `all_horizons_bootstrap.csv`, `all_horizons_decision.json` |
| Table 5, upper panel: window normalization on and off (H = 24) | `results/v6/all_horizons.csv` and `results/v6/nonorm_vs_otarf.csv` |
| Table 5, lower panel: PM2.5 MAE in ug/m3 | `results/v6/pm25_ugm3.csv` |
| Table 6: real outages at the end of the history | `results/extensions/E1_real_gaps.csv` |
| Table 9: high-concentration, rapid-increase and stable labels with OTARF-peak | `results/v6/label_groups.csv` |
| Figure 5: F1 of exceedance of the 24-hour-mean threshold | `results/v6/events24.csv` |

Daily events: a 24-hour interval is scored when at least 20 of its hours are observed (Beijing, Shanghai) or 18
(London); thresholds are PM2.5 75 ug/m3 (Grade II 24-hour limit of GB 3095-2012) for Beijing and Shanghai and 36 ug/m3
(start of the UK DAQI "Moderate" band) for London. In the paper one CRIB run without normalization (Beijing, seed 2022)
diverged at epoch 6 (non-finite gradient); `summarize_v6.py` accepts two seeds for the runs without normalization.

`summarize_v6.py` and `otarf/events.py` were checked against the stored arrays of the paper's runs: the decision
counts, the comparison against the models without normalization (46 of 48 favourable), the PM2.5 errors in ug/m3 and
the event F1 values reproduce the paper (bootstrap counts can differ by one cell with fewer resamples); the daily means
of `otarf/events.py` match the stored arrays exactly (Shanghai, natural and 96-hour outage). A CRIB model built with
`window_norm=False` loads the trained checkpoint of the paper strictly and reproduces its stored errors (London, seed 2021,
96-hour outage).

## Evaluation protocol in brief

* Chronological 70/10/20 % split; normalisation fitted on train only; input window 96 hours.
* nMAE (normalized MAE; called sMAE in older output files): MAE divided by the train SD of each target, on observed labels only, averaged uniformly over
  targets that have at least one label in the scored split.
* Controlled outage: at each origin one target is chosen by a fixed SHA-256 rule; its last 6, 24 or
  96 hours are replaced by the last valid value before the gap; only that target's labels are scored.
  The natural (0-hour) score uses all targets, so it is a different label set. The last-value reference
  repeats the value before the gap on the same labels.
* Bootstrap: paired moving blocks of H and 168 hours on seed-mean per-origin errors, 20 000 resamples,
  Bonferroni within each family (36 comparisons per scenario against the four baselines, 12 against the models without
  window normalization, 15 for the ablations).
  It describes temporal uncertainty of the paired difference, not variation across training runs.

## Reproducibility notes

* Seeds, deterministic cuDNN and `torch.use_deterministic_algorithms(True, warn_only=True)` are set for
  every run. TimeFilter's official gate uses a CUDA operation without a deterministic kernel, so its
  runs can differ slightly between executions. Results can also differ slightly across GPU models.
* OTARF and the baselines use the same objective, masked-history augmentation and checkpoint rule; the
  baselines keep their official architecture, learning rate and batch size.

## Data licences

Beijing Multi-Site and PM2.5 Data of Five Chinese Cities: CC BY 4.0 (UCI Machine Learning Repository).
KDD Cup 2018 with missing values: see the Zenodo record 10.5281/zenodo.4656719 and the original competition terms.
