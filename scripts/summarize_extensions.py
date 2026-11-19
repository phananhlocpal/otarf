#!/usr/bin/env python3
"""Tables and pre-specified decision rules of the extended comparison (results/extensions/).

Reads runs/<tag>/<dataset>/h24/s<seed>/ written by train_otarf.py, train_baseline.py, train_extended_baseline.py and
evaluate_outages.py. Tables whose runs are missing are skipped, so partial reproductions give partial output.

  A_new_baselines.csv     OTARF vs iTransformer and vs CRIB at H = 24 (natural, 6/24/96 h outage of one target), seed means,
                          paired temporal block bootstrap (blocks 24 and 168 h, Bonferroni, family 6 = 3 datasets x 2 baselines)
  A_decision.json         rule A: OTARF is called better in a scenario if all 6 comparisons favor it and at least 5 are clear
                          in both block lengths
  C_peak_vs_otarf.csv     OTARF with the peak-weighted loss vs OTARF, validation and test: natural, q95 tail, 24 h outage
  C_decision.json         rule C (validation): tail sMAE down >= 1% on average in >= 2 datasets with >= 2/3 seeds favorable,
                          natural sMAE not worse than +1% on average, no dataset worse than +2% (natural and 24 h outage)
  B_multi_outage.csv      sMAE by scenario (one target, 25/50% of targets, 25/50% of stations; 24 and 96 h) per model
  B_decision.json         rule B (descriptive): is OTARF lower than every baseline in every cell
  E4_events.csv           precision / recall / F1 of hourly threshold exceedance, natural and 96 h outage of one target
  E0_naive.csv            persistence and seasonal naive sMAE (no training), H = 12/24/48
  E1_real_gaps.csv        sMAE at H = 24 by the length of the real outage at the end of the history (0, 1-6, 7-24, 25-96, >96 h)
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from otarf.io import read_json, write_json
from otarf.statistics import macro, paired_intervals

DATASETS = ["beijing_multisite", "kdd2018_london", "shanghai_confirmation"]
CITY = {"beijing_multisite": "Beijing", "kdd2018_london": "London", "shanghai_confirmation": "Shanghai"}
SEEDS = (2021, 2022, 2023)
H = 24
DRAWS = 20000
RNG = 20261116
TAGS = {"OTARF": "OTARF-otarf", "OTARF-peak": "OTARF-peak", "SOFTS": "SOFTS-indicator", "TimeFilter": "TimeFilter-indicator",
        "iTransformer": "iTransformer-indicator", "CRIB": "CRIB-indicator"}
BASELINES = ("SOFTS", "TimeFilter", "iTransformer", "CRIB")
RUNS = ROOT / "runs"
OUT = ROOT / "results" / "extensions"


def folder(tag, dataset, h, seed):
    f = RUNS / tag / dataset / f"h{h}" / f"s{seed}"
    return f if (f / "result.json").exists() or (f / "outages").exists() else None


def arrays(model, dataset, scenario, split="test", h=H):
    """Per-seed dicts of per-origin arrays, or None when any seed is missing.
    scenario: natural, gap6/24/96 (training-time outage evaluation) or an outages scenario (test only)."""
    out = []
    for seed in SEEDS:
        f = folder(TAGS[model], dataset, h, seed)
        if f is None:
            return None
        if scenario == "natural":
            path = f / "best" / f"{split}_origin_errors.npz"
        elif scenario.startswith("gap"):
            path = f / "masked" / f"{split}_{scenario}.npz"
        else:
            path = f / "outages" / f"test_{scenario}.npz"
        if not path.exists():
            return None
        with np.load(path) as z:
            out.append({k: z[k].copy() for k in z.files})
    return out


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


def smae(arrs, key="abs_z", count="count"):
    return [macro(a[key], a[count]) for a in arrs]


def compare(ours, theirs, family, label):
    for k in ("origin_hour", "target_names", "count"):
        if not np.array_equal(ours[0][k], theirs[0][k]):
            raise ValueError("Origin alignment failed on " + k)
    so, st = smae(ours), smae(theirs)
    row = dict(label, otarf=float(np.mean(so)), other=float(np.mean(st)), ours_lower_pct=100 * (np.mean(st) - np.mean(so)) / np.mean(st),
               favorable_seeds=int(sum(a < b for a, b in zip(so, st))))
    cis = []
    for block in (H, 168):
        ci = paired_intervals(np.mean([a["abs_z"] for a in ours], 0), np.mean([a["abs_z"] for a in theirs], 0), ours[0]["count"],
                              block, DRAWS, np.random.default_rng(RNG + block), family)
        cis.append(dict(label, block=block, **ci))
    row["clear_favorable"] = all(c["family_high"] is not None and c["family_high"] < 0 for c in cis)
    return row, cis


def rule_a():
    rows, boots, decision = [], [], {}
    for sc in ("natural", "gap6", "gap24", "gap96"):
        cells = []
        for d in DATASETS:
            ours = arrays("OTARF", d, sc)
            for other in ("iTransformer", "CRIB"):
                theirs = arrays(other, d, sc)
                if ours is None or theirs is None:
                    continue
                row, cis = compare(ours, theirs, 6, dict(scenario=sc, city=CITY[d], baseline=other))
                rows.append(row)
                boots += cis
                cells.append(row)
        if len(cells) == 6:
            n_clear = sum(r["clear_favorable"] for r in cells)
            all_point = all(r["otarf"] < r["other"] for r in cells)
            decision[sc] = {"all_point_favorable": bool(all_point), "clear_cells": int(n_clear), "of": 6, "claim_better": bool(all_point and n_clear >= 5)}
    save("A_new_baselines.csv", rows)
    save("A_new_baselines_bootstrap.csv", boots)
    if decision:
        write_json(OUT / "A_decision.json", decision)
    return decision


def rule_c():
    rows, val = [], {}
    for split in ("val", "test"):
        for d in DATASETS:
            nat = {m: arrays(m, d, "natural", split) for m in ("OTARF", "OTARF-peak")}
            g24 = {m: arrays(m, d, "gap24", split) for m in ("OTARF", "OTARF-peak")}
            if any(v is None for v in list(nat.values()) + list(g24.values())):
                continue
            row = dict(split=split, city=CITY[d])
            for name, arrs, fn in (("natural", nat, smae), ("tail_q95", nat, lambda a: smae(a, "tail_abs_z", "tail_count")),
                                   ("gap24", g24, smae)):
                core, peak = fn(arrs["OTARF"]), fn(arrs["OTARF-peak"])
                row[f"{name}_otarf"] = float(np.mean(core))
                row[f"{name}_peak"] = float(np.mean(peak))
                row[f"{name}_change_pct"] = float(100 * (np.mean(peak) - np.mean(core)) / np.mean(core))
                row[f"{name}_favorable_seeds"] = int(sum(p < c for p, c in zip(peak, core)))
            rows.append(row)
            if split == "val":
                val[d] = row
    save("C_peak_vs_otarf.csv", rows)
    if len(val) != 3:
        return {}
    tail_mean = float(np.mean([val[d]["tail_q95_change_pct"] for d in DATASETS]))
    tail_cities = sum(val[d]["tail_q95_change_pct"] < 0 and val[d]["tail_q95_favorable_seeds"] >= 2 for d in DATASETS)
    nat_mean = float(np.mean([val[d]["natural_change_pct"] for d in DATASETS]))
    cond = {"tail_down_1pct_and_2_datasets": bool(tail_mean <= -1 and tail_cities >= 2),
            "natural_mean_within_1pct": bool(nat_mean <= 1),
            "no_dataset_natural_worse_than_2pct": bool(all(val[d]["natural_change_pct"] <= 2 for d in DATASETS)),
            "no_dataset_gap24_worse_than_2pct": bool(all(val[d]["gap24_change_pct"] <= 2 for d in DATASETS))}
    decision = {"validation_tail_mean_change_pct": tail_mean, "validation_natural_mean_change_pct": nat_mean, "conditions": cond,
                "accept_peak": bool(all(cond.values()))}
    write_json(OUT / "C_decision.json", decision)
    return decision


def rule_b():
    scenarios = ["single24", "single96"] + [f"{k}{p}_g{g}" for k in ("rand", "station") for p in (25, 50) for g in (24, 96)]
    rows, decision = [], {}
    for sc in scenarios:
        cells = []
        for d in DATASETS:
            if sc.startswith("station") and d == "shanghai_confirmation":
                continue
            ours = arrays("OTARF", d, sc)
            if ours is None:
                continue
            row = dict(scenario=sc, city=CITY[d], OTARF=float(np.mean(smae(ours))))
            pers = RUNS / "naive-persistence" / d / "h24" / "s0" / "outages" / f"test_{sc}.npz"
            if pers.exists():
                with np.load(pers) as z:
                    row["persistence"] = macro(z["abs_z"].astype(np.float64), z["count"].astype(np.int64))
            for m in BASELINES + ("OTARF-peak",):
                other = arrays(m, d, sc)
                if other is None:
                    continue
                row[m] = float(np.mean(smae(other)))
                if m in BASELINES:
                    cells.append((d, m, row["OTARF"] < row[m]))
            rows.append(row)
        if cells:
            decision[sc] = {"cells": len(cells), "otarf_lower_cells": int(sum(c[2] for c in cells)), "reversed": [f"{CITY[d]}:{m}" for d, m, ok in cells if not ok]}
    save("B_multi_outage.csv", rows)
    if decision:
        write_json(OUT / "B_decision.json", decision)
    return decision


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f = 2 * p * r / (p + r) if p == p and r == r and p + r > 0 else float("nan")
    return p, r, f


def events():
    rows = []
    for d in DATASETS:
        for m in list(TAGS) + ["persistence", "seasonal"]:
            naive = m in ("persistence", "seasonal")
            per_seed = []
            for s in ([0] if naive else SEEDS):
                f = (RUNS / f"naive-{m}" / d / "h24" / "s0") if naive else folder(TAGS[m], d, H, s)
                if f is not None and (f / "outages" / "events.json").exists():
                    per_seed.append(read_json(f / "outages" / "events.json")["scenarios"])
            if not per_seed:
                continue
            for sc in ("natural", "single24", "single96"):
                for key in per_seed[0][sc]:
                    ps = [prf(*x[sc][key][:3]) for x in per_seed]
                    with np.errstate(all="ignore"):
                        rows.append(dict(city=CITY[d], model=m, scenario=sc, event=key, seeds=len(per_seed),
                                         precision=float(np.nanmean([x[0] for x in ps])) if not np.isnan([x[0] for x in ps]).all() else float("nan"),
                                         recall=float(np.nanmean([x[1] for x in ps])) if not np.isnan([x[1] for x in ps]).all() else float("nan"),
                                         f1=float(np.nanmean([x[2] for x in ps])) if not np.isnan([x[2] for x in ps]).all() else float("nan"),
                                         positives=int(per_seed[0][sc][key][0] + per_seed[0][sc][key][2])))
    save("E4_events.csv", rows)


def naive():
    rows = []
    for d in DATASETS:
        for h in (12, 24, 48):
            for sc in ("natural", "single6", "single24", "single96"):
                row = dict(city=CITY[d], horizon=h, scenario=sc)
                for k in ("persistence", "seasonal"):
                    p = RUNS / f"naive-{k}" / d / f"h{h}" / "s0" / "outages" / f"test_{sc}.npz"
                    if p.exists():
                        with np.load(p) as z:
                            row[k] = macro(z["abs_z"].astype(np.float64), z["count"].astype(np.int64))
                rows.append(row)
    save("E0_naive.csv", [r for r in rows if len(r) > 3])


def real_gaps():
    bins = [("0", 0, 0), ("1-6", 1, 6), ("7-24", 7, 24), ("25-96", 25, 96), (">96", 97, 10**9)]
    rows = []
    for d in DATASETS:
        meta = read_json(ROOT / "data/processed" / d / "metadata.json")
        ids = np.array([meta["columns"].index(x) for x in meta["targets"]])
        origins = np.arange(meta["test_start"], meta["rows"] - H + 1)
        with np.load(ROOT / "data/processed" / d / "observations.npz") as f:
            observed = f["observed"][:, ids].astype(bool)
        last = np.full(len(ids), -1, np.int64)
        age = np.empty(observed.shape, np.int64)
        for t in range(len(observed)):
            last = np.where(observed[t], t, last)
            age[t] = np.where(last >= 0, t - last, t + 1)
        age = age[origins - 1]
        arrs = {m: arrays(m, d, "natural") for m in TAGS}
        if any(v is None for v in arrs.values()):
            continue
        for name, lo, hi in bins:
            sel = (age >= lo) & (age <= hi)
            row = dict(city=CITY[d], bin=name, labels=int(arrs["OTARF"][0]["count"][sel].sum()))
            for m, a in arrs.items():
                row[m] = float(np.mean([macro(np.where(sel, x["abs_z"], 0.0), np.where(sel, x["count"], 0)) for x in a]))
            rows.append(row)
    save("E1_real_gaps.csv", rows)


if __name__ == "__main__":
    for name, fn in (("rule A", rule_a), ("rule C", rule_c), ("rule B", rule_b)):
        print(name, json.dumps(fn(), indent=1))
    events()
    naive()
    real_gaps()
