#!/usr/bin/env python3
"""Collect finished runs into the tables, checks and decisions of the paper (results/*.csv, results/decisions.json).

Reads runs/<model>-<variant or recipe>/<dataset>/h<H>/s<seed>/ as written by the training scripts.
Tables whose runs are missing are skipped with a message, so partial reproductions give partial output.

Tables (test, mean of three seeds)
  table_natural.csv       natural histories, OTARF vs SOFTS / TimeFilter (recipe 'indicator'), mean and SD
  table_masked.csv        6/24/96-hour outage of one target, plus last-value persistence on the same labels
  table_raw_h24.csv       MAE in native units per pollutant, H = 24
  table_ablation.csv      OTARF and its five variants at H = 24 (natural, 24 h, 96 h): test sMAE, and the
                          paper's ablation table (OTARF lower than the variant, %, validation) with clear marks
  table_tail.csv          labels above the train 95th percentile: macro sMAE and mean signed error
  table_ramp_stable.csv   rising labels (>= 1 train SD above the last observed value) and stable labels
                          (|change| <= 0.25 SD): macro sMAE and mean signed error
  table_history_state.csv natural histories split by the target's own history at the origin
                          (complete / partial / outage), OTARF lower than the better baseline (%)
  table_pollutant_tail_h24.csv native-unit MAE above / below the train 95th percentile, per pollutant
  table_anchor_age96.csv  age of the anchor under the 96-hour outage (H = 24 test origins)
  table_sources.csv       attention by source group (ratio to uniform) and source-group knockout
                          (needs scripts/inspect_otarf.py outputs in runs/inspect/)
  table_cost_h24.csv      parameters, training minutes, peak GPU memory (median of seeds)
Checks
  check_official.csv      retrained baselines vs their official pipelines (natural test)
Statistics
  bootstrap_*.csv         paired temporal block bootstrap on seed-mean per-origin errors (OTARF minus other;
                          negative favours OTARF), blocks H and 168 h, Bonferroni within each family
  decisions.json          the pre-specified decision rules of the paper applied to the above
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from otarf.io import read_json
from otarf.masking import selected_targets
from otarf.statistics import macro, paired_intervals

DATASETS = ["beijing_multisite", "kdd2018_london", "shanghai_confirmation"]
CITY = {"beijing_multisite": "Beijing", "kdd2018_london": "London", "shanghai_confirmation": "Shanghai"}
SEEDS = (2021, 2022, 2023)
HORIZONS = (12, 24, 48)
GAPS = (6, 24, 96)
SCENARIOS = ("natural",) + tuple(f"gap{g}" for g in GAPS)
MODELS = {"OTARF": "OTARF-otarf", "SOFTS": "SOFTS-indicator", "TimeFilter": "TimeFilter-indicator"}
ABLATIONS = ("no_anchor", "no_cross", "mean_pool", "state_blind", "state_blind_no_aug")
RNG = 20261108


class Runs:
    def __init__(self, root):
        self.root = root
        self.missing = set()

    def folder(self, tag, dataset, h, seed):
        f = self.root / tag / dataset / f"h{h}" / f"s{seed}"
        if not (f / "result.json").exists():
            self.missing.add(str(f.relative_to(self.root)))
            return None
        return f

    def arrays(self, tag, dataset, h, scenario, split="test"):
        """List of per-seed dicts of per-origin arrays, or None if any seed is missing."""
        out = []
        for seed in SEEDS:
            f = self.folder(tag, dataset, h, seed)
            if f is None:
                return None
            path = f / "best" / f"{split}_origin_errors.npz" if scenario == "natural" else f / "masked" / f"{split}_{scenario}.npz"
            with np.load(path) as z:
                out.append({k: z[k].copy() for k in z.files})
        keys = ("origin_hour", "target_names", "count") + (() if scenario == "natural" else ("selected", "removed"))
        for a in out[1:]:
            for k in keys:
                if not np.array_equal(a[k], out[0][k]):
                    raise ValueError(f"Seed alignment failed: {tag} {dataset} h{h} {scenario} {k}")
        return out

    def results(self, tag, dataset, h):
        fs = [self.folder(tag, dataset, h, s) for s in SEEDS]
        return None if any(f is None for f in fs) else [read_json(f / "result.json") for f in fs]


def save(out, name, rows):
    if not rows:
        print("skipped (runs missing):", name)
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (out / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval="")
        w.writeheader()
        w.writerows(rows)
    print("wrote", out / name)


def seed_smae(arrs):
    return [macro(a["abs_z"], a["count"]) for a in arrs]


def compare(ours, theirs, scenario, horizon, family, draws, label):
    """Seed-mean comparison with paired temporal block bootstrap (blocks H and 168 h)."""
    keys = ("origin_hour", "target_names", "count") + (() if scenario == "natural" else ("selected", "removed"))
    for k in keys:
        if not np.array_equal(ours[0][k], theirs[0][k]):
            raise ValueError("Origin alignment failed on " + k)
    counts = ours[0]["count"]
    so, st = seed_smae(ours), seed_smae(theirs)
    o, t = float(np.mean(so)), float(np.mean(st))
    row = dict(label, otarf_smae=o, other_smae=t, ours_lower_pct=100 * (t - o) / t,
               favorable_seeds=int(sum(a < b for a, b in zip(so, st))))
    cis = []
    for block in (horizon, 168):
        ci = paired_intervals(np.mean([a["abs_z"] for a in ours], 0), np.mean([a["abs_z"] for a in theirs], 0),
                              counts, block, draws, np.random.default_rng(RNG + block), family)
        cis.append(dict(label, block=block, **ci))
    row["clear_favorable"] = all(c["family_high"] is not None and c["family_high"] < 0 for c in cis)
    row["clear_adverse"] = all(c["family_low"] is not None and c["family_low"] > 0 for c in cis)
    return row, cis


def persistence(data_root, dataset, horizon, gap, split):
    """sMAE of repeating the last value before the gap on the labels scored in the outage experiment."""
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    d = data_root / dataset
    meta = read_json(d / "metadata.json")
    filled = pd.read_csv(d / "data.csv")[meta["columns"]].to_numpy(dtype=np.float64)
    scaler = StandardScaler().fit(filled[: meta["train_end"]])
    values = scaler.transform(filled)
    with np.load(d / "observations.npz") as f:
        raw, observed = f["raw"], f["observed"]
    ids = np.array([meta["columns"].index(t) for t in meta["targets"]])
    start = meta["train_end"] if split == "val" else meta["test_start"]
    end = meta["test_start"] if split == "val" else meta["rows"]
    origins = np.arange(start, end - horizon + 1)
    sel = selected_targets(dataset, origins, len(ids))
    ch = ids[sel]
    pos = origins[:, None] + np.arange(horizon)[None, :]
    z = (raw[pos, ch[:, None]].astype(np.float64) - scaler.mean_[ch][:, None]) / scaler.scale_[ch][:, None]
    valid = observed[pos, ch[:, None]]
    err = np.where(valid, np.abs(values[origins - gap - 1, ch][:, None] - z), 0.0)
    count = np.zeros((len(origins), len(ids)), np.int64)
    absz = np.zeros((len(origins), len(ids)))
    count[np.arange(len(origins)), sel] = valid.sum(1)
    absz[np.arange(len(origins)), sel] = err.sum(1)
    return macro(absz, count)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, default=ROOT / "runs")
    p.add_argument("--data-root", type=Path, default=ROOT / "data/processed")
    p.add_argument("--out", type=Path, default=ROOT / "results")
    p.add_argument("--draws", type=int, default=20000, help="bootstrap resamples (paper: 20000)")
    p.add_argument("--no-bootstrap", action="store_true")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    R = Runs(a.runs)
    decisions = {}

    # Natural and masked tables (OTARF vs baselines, all horizons).
    natural, masked = [], []
    for d in DATASETS:
        for h in HORIZONS:
            for m, tag in MODELS.items():
                row = dict(city=CITY[d], horizon=h, model=m)
                for sc in SCENARIOS:
                    arr = R.arrays(tag, d, h, sc)
                    if arr is None:
                        break
                    v = seed_smae(arr)
                    row[sc + "_mean"], row[sc + "_sd"] = float(np.mean(v)), float(np.std(v, ddof=1))
                else:
                    natural.append({k: row[k] for k in ("city", "horizon", "model", "natural_mean", "natural_sd")})
                    masked.append({k: v for k, v in row.items() if not k.startswith("natural")})
            if any(r["city"] == CITY[d] and r["horizon"] == h for r in masked):
                for g in GAPS:
                    masked.append(dict(city=CITY[d], horizon=h, model="last value (reference)",
                                       **{f"gap{g}_mean": persistence(a.data_root, d, h, g, "test")}))
    save(a.out, "table_natural.csv", natural)
    save(a.out, "table_masked.csv", masked)

    # Native-unit MAE at H = 24, cost at H = 24.
    raw, cost = [], []
    for d in DATASETS:
        res = {m: R.results(tag, d, 24) for m, tag in MODELS.items()}
        if all(res.values()):
            for pol in sorted(res["OTARF"][0]["metrics"]["best"]["test"]["raw_metrics"]):
                raw.append(dict(city=CITY[d], pollutant=pol, **{m: float(np.mean([r["metrics"]["best"]["test"]["raw_metrics"][pol]["mae"] for r in rs])) for m, rs in res.items()}))
            for m, rs in res.items():
                cost.append(dict(city=CITY[d], model=m, parameters=rs[0]["parameters"],
                                 minutes=float(np.median([r["training_seconds"] / 60 for r in rs])),
                                 peak_mib=float(np.median([r["peak_gpu_bytes"] / 2 ** 20 for r in rs]))))
    save(a.out, "table_raw_h24.csv", raw)
    save(a.out, "table_cost_h24.csv", cost)

    # Upper tail (labels >= train q95 of each target).
    tail = []
    for d in DATASETS:
        for h in HORIZONS:
            for m, tag in MODELS.items():
                arr = R.arrays(tag, d, h, "natural")
                if arr is None:
                    continue
                tail.append(dict(city=CITY[d], horizon=h, model=m,
                                 tail_smae=float(np.mean([macro(x["tail_abs_z"], x["tail_count"]) for x in arr])),
                                 tail_signed=float(np.mean([macro(x["tail_signed_z"], x["tail_count"]) for x in arr]))))
    save(a.out, "table_tail.csv", tail)

    # Rising / stable labels.
    rs = []
    for d in DATASETS:
        for h in HORIZONS:
            for m, tag in MODELS.items():
                arr = R.arrays(tag, d, h, "natural")
                if arr is None:
                    continue
                rs.append(dict(city=CITY[d], horizon=h, model=m,
                               ramp_smae=float(np.mean([macro(x["ramp_abs_z"], x["ramp_count"]) for x in arr])),
                               ramp_signed=float(np.mean([macro(x["ramp_signed_z"], x["ramp_count"]) for x in arr])),
                               stable_smae=float(np.mean([macro(x["stable_abs_z"], x["stable_count"]) for x in arr])),
                               ramp_share=float(arr[0]["ramp_count"].sum() / arr[0]["count"].sum())))
    save(a.out, "table_ramp_stable.csv", rs)

    # Natural histories split by the target's own history state at the origin.
    hs = []
    for d in DATASETS:
        meta = read_json(a.data_root / d / "metadata.json")
        ids = np.array([meta["columns"].index(t) for t in meta["targets"]])
        with np.load(a.data_root / d / "observations.npz") as f:
            obs = f["observed"][:, ids].astype(np.int64)
        prefix = np.vstack([np.zeros((1, len(ids)), np.int64), obs.cumsum(0)])
        for h in HORIZONS:
            arrs = {m: R.arrays(tag, d, h, "natural") for m, tag in MODELS.items()}
            if any(v is None for v in arrs.values()):
                continue
            origins = np.arange(meta["test_start"], meta["test_start"] + len(arrs["OTARF"][0]["origin_hour"]))
            seen = prefix[origins] - prefix[origins - 96]
            state = np.where(~obs[origins - 1].astype(bool), 2, np.where(seen == 96, 0, 1))
            for k, name in enumerate(("complete", "partial", "outage")):
                e = {m: float(np.mean([macro(x["abs_z"] * (state == k), x["count"] * (state == k)) for x in arr])) for m, arr in arrs.items()}
                best = min(e["SOFTS"], e["TimeFilter"])
                c0 = arrs["OTARF"][0]["count"]
                hs.append(dict(city=CITY[d], horizon=h, state=name, share_of_labels=float((c0 * (state == k)).sum() / c0.sum()),
                               **e, otarf_lower_than_better_baseline_pct=100 * (best - e["OTARF"]) / best))
    save(a.out, "table_history_state.csv", hs)

    # Native-unit MAE above / below the train 95th percentile (H = 24).
    pt = []
    for d in DATASETS:
        res = {m: R.results(tag, d, 24) for m, tag in MODELS.items()}
        if not all(res.values()):
            continue
        for pol in sorted(res["OTARF"][0]["metrics"]["best"]["test"]["raw_metrics"]):
            row = dict(city=CITY[d], pollutant=pol)
            for m, rsl in res.items():
                xs = [r["metrics"]["best"]["test"]["raw_metrics"][pol] for r in rsl]
                row[m + "_above"] = float(np.mean([x["tail_mae"] for x in xs]))
                row[m + "_below"] = float(np.mean([(x["mae"] * x["count"] - x["tail_mae"] * x["tail_count"]) / (x["count"] - x["tail_count"]) for x in xs]))
            x0 = res["OTARF"][0]["metrics"]["best"]["test"]["raw_metrics"][pol]
            row["above_share"] = x0["tail_count"] / x0["count"]
            pt.append(row)
    save(a.out, "table_pollutant_tail_h24.csv", pt)

    # Anchor age under the 96-hour outage (data only).
    ag = []
    for d in DATASETS:
        meta = read_json(a.data_root / d / "metadata.json")
        ids = np.array([meta["columns"].index(t) for t in meta["targets"]])
        with np.load(a.data_root / d / "observations.npz") as f:
            obs = f["observed"][:, ids]
        age = np.zeros(obs.shape)
        last = np.full(len(ids), -10 ** 6)
        for t in range(len(obs)):
            last = np.where(obs[t], t, last)
            age[t] = t - last
        o = np.arange(meta["test_start"], meta["rows"] - 24 + 1)
        sel = selected_targets(d, o, len(ids))
        v = age[o - 97, sel] + 96
        ag.append(dict(city=CITY[d], median_anchor_age_h=float(np.median(v)), share_older_than_week=float((v >= 168).mean())))
    save(a.out, "table_anchor_age96.csv", ag)

    # Where the cross-variable branch takes its information from (inference outputs).
    src = []
    for d in DATASETS:
        folder = a.runs / "inspect" / d
        if not (folder / "summary.json").exists():
            continue
        info = read_json(folder / "summary.json")
        for sc in ("natural", "gap24", "gap96"):
            zs = [dict(np.load(folder / f"inspect_s{s}_{sc}.npz")) for s in SEEDS]
            base = [macro(z["none__abs_z"], z["none__count"]) for z in zs]
            for k, g in enumerate(info["groups"] + ["all_sources"]):
                if g not in info["excluded_sets"]:
                    continue
                row = dict(city=CITY[d], scenario=sc, group=g)
                if g != "all_sources":
                    share, uni = [], []
                    for z in zs:
                        m, c, n = z["att_mass"][..., k], z["att_candidates"], z["att_null"]
                        tot = c.sum(-1); use = tot > 0
                        share.append(float(m[use].sum() / (1 - n)[use].sum())); uni.append(float((c[..., k][use] / tot[use]).mean()))
                    if max(uni) == 0:
                        continue
                    row.update(attention_share=float(np.mean(share)), uniform_share=float(np.mean(uni)),
                               attention_ratio=float(np.mean(share) / np.mean(uni)), null_weight=float(np.mean([float(z["att_null"].mean()) for z in zs])))
                ko = [macro(z[f"{g}__abs_z"], z[f"{g}__count"]) for z in zs]
                inc = [100 * (x - b) / b for x, b in zip(ko, base)]
                row.update(otarf=float(np.mean(base)), knocked_out=float(np.mean(ko)), knockout_increase_pct=float(np.mean(inc)),
                           seeds_increase=int(sum(i > 0 for i in inc)))
                src.append(row)
    if src:
        save(a.out, "table_sources.csv", src)

    # Official-pipeline sanity check.
    official = []
    for d in DATASETS:
        for h in HORIZONS:
            for m in ("SOFTS", "TimeFilter"):
                o, r = R.arrays(f"{m}-official", d, h, "natural"), R.arrays(MODELS[m], d, h, "natural")
                if o is None or r is None:
                    continue
                om, rm = float(np.mean(seed_smae(o))), float(np.mean(seed_smae(r)))
                official.append(dict(city=CITY[d], horizon=h, model=m, official=om, retrained=rm, retrained_better=rm < om))
    save(a.out, "check_official.csv", official)
    if official:
        decisions["retrained_better_than_official"] = f"{sum(r['retrained_better'] for r in official)}/{len(official)}"

    # Ablation table (H = 24, test means) and comparisons.
    abl = []
    for v in ("otarf",) + ABLATIONS:
        row = dict(variant=v)
        for d in DATASETS:
            for sc in ("natural", "gap24", "gap96"):
                arr = R.arrays(f"OTARF-{v}", d, 24, sc)
                row[f"{CITY[d]}_{sc}"] = None if arr is None else float(np.mean(seed_smae(arr)))
        abl.append(row)
    if all(all(x is not None for k, x in r.items() if k != "variant") for r in abl):
        save(a.out, "table_ablation.csv", abl)
    else:
        print("skipped (runs missing): table_ablation.csv")

    if a.no_bootstrap:
        (a.out / "decisions.json").write_text(json.dumps(decisions, indent=1))
        report_missing(R)
        return

    # Comparison with baselines: family 18 per scenario and split.
    rows, boots = [], []
    for sc in SCENARIOS:
        for split in ("val", "test"):
            for d in DATASETS:
                for h in HORIZONS:
                    ours = R.arrays(MODELS["OTARF"], d, h, sc, split)
                    for m in ("SOFTS", "TimeFilter"):
                        theirs = R.arrays(MODELS[m], d, h, sc, split)
                        if ours is None or theirs is None:
                            continue
                        r, c = compare(ours, theirs, sc, h, 18, a.draws, dict(scenario=sc, split=split, city=CITY[d], horizon=h, other=m))
                        rows.append(r)
                        boots += c
    save(a.out, "bootstrap_comparison.csv", boots)
    save(a.out, "comparison.csv", rows)
    for sc in SCENARIOS:
        t = [r for r in rows if r["scenario"] == sc and r["split"] == "test"]
        if len(t) == 18:
            info = dict(favorable=sum(r["ours_lower_pct"] > 0 for r in t), clear_favorable=sum(r["clear_favorable"] for r in t),
                        clear_adverse=sum(r["clear_adverse"] for r in t))
            info["better"] = info["favorable"] == 18 and info["clear_favorable"] >= 14
            if sc == "natural":
                info["not_worse_within_1pct"] = all(r["ours_lower_pct"] >= -1 for r in t)
            decisions["comparison_" + sc] = info

    # Ablations: family 15 per scenario and split; rule on validation, 24 h outage (and 96 h for no_anchor).
    arows, aboots = [], []
    for sc in ("natural", "gap24", "gap96"):
        for split in ("val", "test"):
            for d in DATASETS:
                ours = R.arrays("OTARF-otarf", d, 24, sc, split)
                for v in ABLATIONS:
                    theirs = R.arrays(f"OTARF-{v}", d, 24, sc, split)
                    if ours is None or theirs is None:
                        continue
                    r, c = compare(ours, theirs, sc, 24, 15, a.draws, dict(scenario=sc, split=split, city=CITY[d], variant=v))
                    arows.append(r)
                    aboots += c
    save(a.out, "bootstrap_ablation.csv", aboots)
    save(a.out, "ablation_comparison.csv", arows)

    paper_abl = []
    for v in ABLATIONS:
        row = dict(variant=v)
        for d in DATASETS:
            for sc in ("natural", "gap24", "gap96"):
                r = next((x for x in arows if x["variant"] == v and x["city"] == CITY[d] and x["scenario"] == sc and x["split"] == "val"), None)
                if r is not None:
                    row[f"{CITY[d]}_{sc}_otarf_lower_pct"] = r["ours_lower_pct"]
                    row[f"{CITY[d]}_{sc}_clear"] = r["clear_favorable"]
        paper_abl.append(row)
    save(a.out, "table_ablation_paper.csv", paper_abl)

    def supported(v, sc):
        prim = [r for r in arows if r["variant"] == v and r["scenario"] == sc and r["split"] == "val"]
        nat = [r for r in arows if r["variant"] == v and r["scenario"] == "natural" and r["split"] == "val"]
        if len(prim) != 3 or len(nat) != 3:
            return None
        return bool(sum(r["ours_lower_pct"] >= 1 for r in prim) >= 2 and all(r["ours_lower_pct"] > 0 and r["favorable_seeds"] >= 2 for r in prim)
                    and sum(r["clear_favorable"] for r in prim) >= 2 and float(np.mean([r["ours_lower_pct"] for r in nat])) >= -1)
    for v in ABLATIONS:
        decisions[f"ablation_{v}_gap24"] = supported(v, "gap24")
    decisions["ablation_no_anchor_gap96"] = supported("no_anchor", "gap96")

    (a.out / "decisions.json").write_text(json.dumps(decisions, indent=1))
    print("wrote", a.out / "decisions.json")
    report_missing(R)


def report_missing(R):
    if R.missing:
        print(f"{len(R.missing)} run folders missing, e.g.", sorted(R.missing)[:5])


if __name__ == "__main__":
    main()
