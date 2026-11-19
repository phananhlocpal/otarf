#!/usr/bin/env python3
"""Inference only: outages of several channels at once, hourly threshold events, and parameter-free references.

Loads the best checkpoint of a finished run (or runs a naive rule) and scores the scenarios of otarf.outages on the
test split, writing runs/<tag>/<dataset>/h<H>/s<seed>/outages/test_<scenario>.npz and events.json.

  python scripts/evaluate_outages.py --model OTARF --dataset beijing_multisite --seed 2021
  python scripts/evaluate_outages.py --model persistence --dataset beijing_multisite --horizon 12
Models: OTARF, OTARF-peak, SOFTS, TimeFilter, iTransformer, CRIB (need their finished runs) and persistence,
seasonal (no run needed).
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
from otarf.io import read_json, write_json
from otarf.outages import THRESHOLDS, naive_predictor, run_scenario, scenarios_for
from otarf.training import set_seed
from otarf.variants import build_model

TAGS = {"OTARF": "OTARF-otarf", "OTARF-peak": "OTARF-peak", "SOFTS": "SOFTS-indicator", "TimeFilter": "TimeFilter-indicator",
        "iTransformer": "iTransformer-indicator", "CRIB": "CRIB-indicator"}
NAIVE = ("persistence", "seasonal")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=tuple(TAGS) + NAIVE)
    p.add_argument("--dataset", required=True)
    p.add_argument("--horizon", type=int, default=24, choices=(12, 24, 48))
    p.add_argument("--seed", type=int, default=2021)
    p.add_argument("--config", type=Path, default=ROOT / "configs/otarf.json")
    p.add_argument("--baseline-config", type=Path, default=ROOT / "configs/baselines.json")
    p.add_argument("--data-root", type=Path, default=ROOT / "data/processed")
    p.add_argument("--runs", type=Path, default=ROOT / "runs")
    a = p.parse_args()
    h = a.horizon
    directory = (a.data_root / a.dataset).resolve()
    data = CIRFData(directory)
    meta = data.meta
    naive = a.model in NAIVE
    if not naive and h != 24:
        raise SystemExit("Trained models are evaluated at H = 24 only")
    if naive:
        out = a.runs / f"naive-{a.model}" / a.dataset / f"h{h}" / "s0"
        predict = naive_predictor(a.model, h)
        batch, scenarios = 256, (["natural", "single6", "single24", "single96"] if h != 24 else scenarios_for(a.dataset, True))
    else:
        if not torch.cuda.is_available():
            raise SystemExit("A CUDA GPU is required.")
        device = torch.device("cuda:0")
        run = a.runs / TAGS[a.model] / a.dataset / f"h{h}" / f"s{a.seed}"
        if not (run / "checkpoints" / "best.pth").exists():
            raise SystemExit(f"Missing finished run: {run}")
        out = run
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

        scenarios = scenarios_for(a.dataset)
    folder = out / "outages"
    folder.mkdir(parents=True, exist_ok=True)
    events = {}
    for sc in scenarios:
        arrays, ev = run_scenario(predict, data, a.dataset, h, sc, batch)
        np.savez_compressed(folder / f"test_{sc}.npz", **arrays)
        events[sc] = ev
        print("scenario", sc, "done", flush=True)
    write_json(folder / "events.json", {"dataset": a.dataset, "thresholds": THRESHOLDS[a.dataset], "unit": "ug/m3 (hourly values)",
                                        "order": ["tp", "fp", "fn", "tn"], "scenarios": events})


if __name__ == "__main__":
    main()
