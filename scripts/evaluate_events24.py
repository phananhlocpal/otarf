#!/usr/bin/env python3
"""Inference only: exceedance events on the 24-hour mean at H = 24 (otarf.events).

Loads the best checkpoint of a finished H = 24 run and writes
runs/<tag>/<dataset>/h24/s<seed>/events24/daily_{natural,single96}.npz.

  python scripts/evaluate_events24.py --model OTARF --dataset beijing_multisite --seed 2021
Models: OTARF, OTARF-peak, SOFTS, TimeFilter, iTransformer, CRIB (their H = 24 runs must be finished).
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from otarf.baselines import build_recipe_baseline
from otarf.data import CIRFData
from otarf.events import HORIZON, MIN_HOURS, THRESHOLDS, daily_means
from otarf.io import read_json, write_json
from otarf.training import set_seed
from otarf.variants import build_model

TAGS = {"OTARF": "OTARF-otarf", "OTARF-peak": "OTARF-peak", "SOFTS": "SOFTS-indicator", "TimeFilter": "TimeFilter-indicator",
        "iTransformer": "iTransformer-indicator", "CRIB": "CRIB-indicator"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=tuple(TAGS))
    p.add_argument("--dataset", required=True)
    p.add_argument("--seed", type=int, default=2021)
    p.add_argument("--config", type=Path, default=ROOT / "configs/otarf.json")
    p.add_argument("--baseline-config", type=Path, default=ROOT / "configs/baselines.json")
    p.add_argument("--data-root", type=Path, default=ROOT / "data/processed")
    p.add_argument("--runs", type=Path, default=ROOT / "runs")
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required.")
    h, device = HORIZON, torch.device("cuda:0")
    directory = (a.data_root / a.dataset).resolve()
    data = CIRFData(directory)
    meta = data.meta
    run = a.runs / TAGS[a.model] / a.dataset / f"h{h}" / f"s{a.seed}"
    if not (run / "checkpoints" / "best.pth").exists():
        raise SystemExit(f"Missing finished run: {run}")
    set_seed(a.seed)
    if a.model.startswith("OTARF"):
        config = read_json(a.config)
        model = build_model("otarf", meta["channels"], h, config).to(device)
        batch = config["train"]["batch_size"]
    else:
        config = read_json(a.baseline_config)
        model = build_recipe_baseline(a.model, ROOT / "external" / a.model, config, meta, directory, run, h, device)
        batch = config["model_args"][a.model]["batch_size"]
    model.load_state_dict(torch.load(run / "checkpoints" / "best.pth", map_location=device, weights_only=True))
    model.eval()

    def predict(x, mark, origins):
        with torch.no_grad():
            return model(torch.from_numpy(x).float().to(device), torch.from_numpy(mark).float().to(device)).cpu().numpy()

    folder = run / "events24"
    folder.mkdir(parents=True, exist_ok=True)
    summary = {}
    for scenario in ("natural", "single96"):
        arrays = daily_means(predict, data, a.dataset, scenario, batch)
        np.savez_compressed(folder / f"daily_{scenario}.npz", **arrays)
        summary[scenario] = {"scored_intervals": int(arrays["scored"].sum()), "origins": int(len(arrays["origin_index"]))}
        print("scenario", scenario, summary[scenario], flush=True)
    write_json(folder / "summary.json", {"dataset": a.dataset, "model": a.model, "seed": a.seed, "min_hours": MIN_HOURS[a.dataset],
                                         "thresholds_ug_m3": THRESHOLDS[a.dataset], "scenarios": summary})


if __name__ == "__main__":
    main()
