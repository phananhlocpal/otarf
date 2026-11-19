#!/usr/bin/env python3
"""Tables of the final paper that involve all five models (no model is run; reads finished runs).

Writes results/v6/:
  all_horizons.csv, all_horizons_bootstrap.csv, all_horizons_decision.json
        test nMAE of OTARF, SOFTS, TimeFilter, iTransformer, CRIB at H = 12/24/48 with natural history and a 6/24/96 h
        outage of one target; paired block bootstrap (blocks H and 168 h, Bonferroni family 36) and the decision rule
  nonorm_vs_otarf.csv   H = 24: OTARF against the four models with window normalization switched off (family 12)
  pm25_ugm3.csv         PM2.5 MAE in ug/m3 at H = 24, normalization on and off
  label_groups.csv      nMAE on high-concentration, rapid-increase and stable labels, six models (with OTARF-peak), H = 12/24/48
  events24.csv          F1 of 24-hour-mean exceedance events at H = 24 (needs scripts/evaluate_events24.py)
Runs needed: train_otarf.py (also --peak at H = 12/24/48), train_baseline.py --recipe indicator,
train_extended_baseline.py (iTransformer, CRIB at H = 12/24/48; the four models with --no-window-norm at H = 24).
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from otarf.events import f1_scores
from otarf.statistics import macro, paired_intervals

DATASETS = ["beijing_multisite", "kdd2018_london", "shanghai_confirmation"]
CITY = {"beijing_multisite": "Beijing", "kdd2018_london": "London", "shanghai_confirmation": "Shanghai"}
SEEDS, HORIZONS = (2021, 2022, 2023), (12, 24, 48)
SCENARIOS = {"clean": "natural", "missing6": "gap6", "missing24": "gap24", "missing96": "gap96"}
BASELINES = ("SOFTS", "TimeFilter", "iTransformer", "CRIB")
TAGS = {"OTARF": "OTARF-otarf", "OTARF-peak": "OTARF-peak", "SOFTS": "SOFTS-indicator", "TimeFilter": "TimeFilter-indicator",
        "iTransformer": "iTransformer-indicator", "CRIB": "CRIB-indicator"}
DRAWS, RNG = 20000, 20261117
RUNS, OUT = ROOT / "runs", ROOT / "results" / "v6"


def load(tag, dataset, h, seed, scenario):
    f = RUNS / tag / dataset / f"h{h}" / f"s{seed}"
    if not (f / "result.json").exists():
        return None
    path = f / "best" / "test_origin_errors.npz" if scenario == "natural" else f / "masked" / f"test_{scenario}.npz"
    if not path.exists():
        return None
    with np.load(path) as z:
        return {k: z[k].copy() for k in z.files}


def seeds(tag, dataset, h, scenario, minimum=3):
    arrs = [x for x in (load(tag, dataset, h, s, scenario) for s in SEEDS) if x is not None]
    return arrs if len(arrs) >= minimum else None


def save(name, rows):
    if not rows:
        print("skipped (runs missing):", name)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval="")
        w.writeheader()
        w.writerows(rows)
    print("wrote", OUT / name)


def intervals(ours, theirs, h, family):
    counts = ours[0]["count"]
    for x in theirs:
        if not np.array_equal(x["count"], counts):
            raise ValueError("Origin alignment failed")
    return [dict(block=b, **paired_intervals(np.mean([a["abs_z"] for a in ours], 0), np.mean([a["abs_z"] for a in theirs], 0),
                                            counts, b, DRAWS, np.random.default_rng(RNG + b), family)) for b in (h, 168)]


def clear(cis, favorable=True):
    if favorable:
        return all(c["family_high"] is not None and c["family_high"] < 0 for c in cis)
    return all(c["family_low"] is not None and c["family_low"] > 0 for c in cis)


def all_horizons():
    rows, boots, decision = [], [], {}
    for sc, key in SCENARIOS.items():
        cells = []
        for d in DATASETS:
            for h in HORIZONS:
                series = {m: seeds(TAGS[m], d, h, key) for m in ("OTARF",) + BASELINES}
                if series["OTARF"] is None:
                    continue
                row = dict(scenario=sc, city=CITY[d], H=h)
                for m, arrs in series.items():
                    if arrs is not None:
                        vals = [macro(a["abs_z"], a["count"]) for a in arrs]
                        row[m], row[m + "_sd"] = float(np.mean(vals)), float(np.std(vals, ddof=1))
                for b in BASELINES:
                    if series[b] is None:
                        continue
                    cis = intervals(series["OTARF"], series[b], h, 36)
                    boots += [dict(scenario=sc, city=CITY[d], H=h, baseline=b, **c) for c in cis]
                    row[b + "_clear"] = clear(cis)
                    cells.append((row["OTARF"] < row[b], row[b + "_clear"], 100 * (row["OTARF"] - row[b]) / row[b]))
                rows.append(row)
        if cells:
            decision[sc] = {"comparisons": len(cells), "favorable": sum(c[0] for c in cells), "clear": sum(c[1] for c in cells),
                            "better": len(cells) == 36 and all(c[0] for c in cells) and sum(c[1] for c in cells) >= 28,
                            "worst_cell_pct": max(c[2] for c in cells), "non_inferior": max(c[2] for c in cells) <= 1.0}
    save("all_horizons.csv", rows)
    save("all_horizons_bootstrap.csv", boots)
    if decision:
        (OUT / "all_horizons_decision.json").write_text(json.dumps(decision, indent=1))


def nonorm():
    rows = []
    for sc, key in SCENARIOS.items():
        for d in DATASETS:
            ours = seeds(TAGS["OTARF"], d, 24, key)
            if ours is None:
                continue
            row = dict(scenario=sc, city=CITY[d], OTARF=float(np.mean([macro(a["abs_z"], a["count"]) for a in ours])))
            for m in BASELINES:
                # One CRIB run without normalization diverged in the paper (Beijing, seed 2022): two seeds are accepted.
                off = seeds(f"{m}-nonorm", d, 24, key, minimum=2)
                if off is None:
                    continue
                row[m] = float(np.mean([macro(a["abs_z"], a["count"]) for a in off]))
                row[m + "_seeds"] = len(off)
                cis = intervals(ours, off, 24, 12)
                row[m + "_clear_otarf"], row[m + "_clear_model"] = clear(cis), clear(cis, False)
            rows.append(row)
    save("nonorm_vs_otarf.csv", rows)


def pm25(arrays, natural_arrays):
    names = [str(x) for x in natural_arrays["target_names"]]
    pm = np.array([n.endswith("PM2.5") or n.startswith("PM_") for n in names])
    sigma = natural_arrays["abs_raw"].sum(0) / np.maximum(natural_arrays["abs_z"].sum(0), 1e-12)
    raw = arrays["abs_raw"] if "abs_raw" in arrays else arrays["abs_z"] * sigma[None, :]
    return float(raw[:, pm].sum() / arrays["count"][:, pm].sum())


def ugm3():
    rows = []
    for d in DATASETS:
        for tag in [TAGS["OTARF"]] + [TAGS[m] for m in BASELINES] + [f"{m}-nonorm" for m in BASELINES]:
            row = dict(city=CITY[d], run=tag)
            for sc, key in SCENARIOS.items():
                vals = []
                for s in SEEDS:
                    nat, arr = load(tag, d, 24, s, "natural"), load(tag, d, 24, s, key)
                    if nat is not None and arr is not None:
                        vals.append(pm25(arr, nat))
                if vals:
                    row[sc] = float(np.mean(vals))
            if len(row) > 2:
                rows.append(row)
    save("pm25_ugm3.csv", rows)


def label_groups():
    groups = {"high": ("tail_abs_z", "tail_count"), "rapid": ("ramp_abs_z", "ramp_count"),
              "stable": ("stable_abs_z", "stable_count"), "all": ("abs_z", "count")}
    rows = []
    for d in DATASETS:
        for h in HORIZONS:
            row = dict(city=CITY[d], H=h)
            for m in ("OTARF", "OTARF-peak") + BASELINES:
                arrs = seeds(TAGS[m], d, h, "natural")
                if arrs is None:
                    continue
                for g, (ka, kc) in groups.items():
                    row[f"{m}_{g}"] = float(np.mean([macro(a[ka], a[kc]) for a in arrs]))
            rows.append(row)
    save("label_groups.csv", rows)


def events24():
    rows = []
    for d in DATASETS:
        for m in ("OTARF", "OTARF-peak") + BASELINES:
            for scenario in ("natural", "single96"):
                for variable in ("PM2.5", "PM10"):
                    scores = []
                    for s in SEEDS:
                        path = RUNS / TAGS[m] / d / "h24" / f"s{s}" / "events24" / f"daily_{scenario}.npz"
                        if path.exists():
                            with np.load(path) as z:
                                scores.append(f1_scores({k: z[k] for k in z.files}, d, variable))
                    if len(scores) == len(SEEDS):
                        p, r, f, n = (np.array(x) for x in zip(*scores))
                        rows.append(dict(city=CITY[d], model=m, scenario=scenario, variable=variable, f1=float(f.mean()),
                                         f1_sd=float(f.std(ddof=1)), precision=float(p.mean()), recall=float(r.mean()), positives=int(n[0])))
    save("events24.csv", rows)


if __name__ == "__main__":
    all_horizons()
    nonorm()
    ugm3()
    label_groups()
    events24()
