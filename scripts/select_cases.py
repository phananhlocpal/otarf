#!/usr/bin/env python3
"""Choose the 96-hour case-study origin per city from stored error arrays (no model run).

Seed 2021, H = 24, test, 96-hour outage of the hashed target. Eligible origins: all 24 future labels of the
masked target observed and at least 72 of the 96 masked hours had a real observation. Score d = mean(SOFTS,
TimeFilter) sMAE minus OTARF sMAE on that origin; choose the eligible origin whose d is closest to the median
of d (earliest on ties). Writes configs/case_selection.json (the paper's choice ships with the repository).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from otarf.io import read_json

DATASETS = ["beijing_multisite", "kdd2018_london", "shanghai_confirmation"]


def choose(runs, data_root, dataset):
    tags = {"OTARF": "OTARF-otarf", "SOFTS": "SOFTS-indicator", "TimeFilter": "TimeFilter-indicator"}
    arr = {m: dict(np.load(runs / t / dataset / "h24" / "s2021" / "masked" / "test_gap96.npz")) for m, t in tags.items()}
    ref = arr["OTARF"]
    rows = np.arange(len(ref["selected"])); sel = ref["selected"]
    count = ref["count"][rows, sel]
    err = {m: a["abs_z"][rows, sel] / np.maximum(count, 1) for m, a in arr.items()}
    d = (err["SOFTS"] + err["TimeFilter"]) / 2 - err["OTARF"]
    eligible = np.where((count == 24) & (ref["removed"] >= 72))[0]
    med = float(np.median(d[eligible]))
    i = int(eligible[np.argmin(np.abs(d[eligible] - med))])
    meta = read_json(data_root / dataset / "metadata.json")
    return {"dataset": dataset, "origin_index": int(meta["test_start"] + i), "target": meta["targets"][int(sel[i])],
            "target_position": int(sel[i]), "stored_smae": {m: float(e[i]) for m, e in err.items()}}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, default=ROOT / "runs")
    p.add_argument("--data-root", type=Path, default=ROOT / "data/processed")
    p.add_argument("--out", type=Path, default=ROOT / "configs/case_selection.json")
    a = p.parse_args()
    cases = [choose(a.runs, a.data_root, d) for d in DATASETS]
    a.out.write_text(json.dumps(cases, indent=1) + "\n")
    for c in cases:
        print(c["dataset"], c["target"], c["origin_index"])
