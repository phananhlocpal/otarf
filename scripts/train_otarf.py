#!/usr/bin/env python3
"""Train and evaluate OTARF or one of its internal variants on one dataset / horizon / seed.

Example:
  python scripts/train_otarf.py --dataset beijing_multisite --horizon 24 --seed 2021
  python scripts/train_otarf.py --dataset kdd2018_london --horizon 24 --seed 2022 --variant no_cross
  python scripts/train_otarf.py --dataset kdd2018_london --horizon 24 --seed 2022 --peak   # peak-weighted loss

Every run evaluates natural histories and the controlled 6/24/96-hour outage of one target.
"""
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from otarf.data import CIRFData, EvaluationAdapter
from otarf.evaluate import evaluate
from otarf.io import read_json, write_json, sha256
from otarf.masking import GAPS, evaluate_masked
from otarf.training import fit, set_seed
from otarf.variants import VARIANTS, build_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--horizon", type=int, required=True, choices=(12, 24, 48))
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--variant", default="otarf", choices=VARIANTS)
    p.add_argument(
        "--peak",
        action="store_true",
        help="OTARF with the peak-weighted loss (weight 3 on labels at or above the train 95th percentile of their target)",
    )
    p.add_argument("--config", type=Path, default=ROOT / "configs/otarf.json")
    p.add_argument("--data-root", type=Path, default=ROOT / "data/processed")
    p.add_argument("--runs", type=Path, default=ROOT / "runs")
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required.")
    config = read_json(a.config)
    h = a.horizon
    device = torch.device("cuda:0")
    if a.peak and a.variant != "otarf":
        raise SystemExit("--peak applies to the full model only")
    tag = "peak" if a.peak else a.variant
    out = a.runs / f"OTARF-{tag}" / a.dataset / f"h{h}" / f"s{a.seed}"
    out.mkdir(parents=True, exist_ok=True)
    set_seed(a.seed)
    torch.cuda.reset_peak_memory_stats()
    data = CIRFData(a.data_root / a.dataset)
    meta = data.meta
    model = build_model(a.variant, meta["channels"], h, config).to(device)
    write_json(
        out / "run.json",
        {
            "model": "OTARF",
            "variant": a.variant,
            "peak": config["peak"] if a.peak else None,
            "dataset": a.dataset,
            "horizon": h,
            "seed": a.seed,
            "config": config,
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "data_sha256": {
                f: sha256(a.data_root / a.dataset / f)
                for f in ("data.csv", "observations.npz")
            },
        },
    )
    fitted = fit(
        model,
        data,
        config,
        h,
        device,
        out,
        mode="otarf",
        natural_only=(a.variant == "state_blind_no_aug"),
        # The masked view is drawn after the natural forward pass, as in the paper's runs.
        clean_first=True,
        peak=config["peak"] if a.peak else None,
    )
    args = SimpleNamespace(
        pred_len=h,
        seq_len=config["seq_len"],
        label_len=config["label_len"],
        batch_size=config["train"]["batch_size"],
        num_workers=0,
    )
    adapter = EvaluationAdapter(model, data, args, config, device)
    metrics = {}
    for selection in ("best", "last"):
        model.load_state_dict(
            torch.load(
                out / "checkpoints" / f"{selection}.pth",
                map_location=device,
                weights_only=True,
            )
        )
        folder = out / selection
        folder.mkdir(exist_ok=True)
        metrics[selection] = {
            s: evaluate(adapter, "OTARF", meta, data.directory, s, folder)
            for s in ("val", "test")
        }
    best_val = fitted["curve"][fitted["best_epoch"] - 1][
        "validation_observed_macro_smae"
    ]
    if not np.isclose(
        metrics["best"]["val"]["standardized_pollutant_macro_mae"],
        best_val,
        rtol=2e-5,
        atol=2e-5,
    ):
        raise ValueError(
            "Best checkpoint validation sMAE does not match the external evaluator"
        )
    masked = {}
    model.load_state_dict(
        torch.load(
            out / "checkpoints" / "best.pth", map_location=device, weights_only=True
        )
    )
    model.eval()
    folder = out / "masked"
    folder.mkdir(exist_ok=True)

    def predict(x, mark, origins):
        with torch.no_grad():
            return (
                model(
                    torch.from_numpy(x).float().to(device),
                    torch.from_numpy(mark).float().to(device),
                )
                .cpu()
                .numpy()
            )

    masked = {
        str(g): evaluate_masked(
            predict, data, a.dataset, h, g, folder, config["train"]["batch_size"]
        )
        for g in GAPS
    }
    write_json(
        out / "result.json",
        {
            "status": "complete",
            "model": "OTARF",
            "variant": a.variant,
            "dataset": a.dataset,
            "horizon": h,
            "seed": a.seed,
            "parameters": sum(p.numel() for p in model.parameters()),
            "training_seconds": fitted["training_seconds"],
            "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
            "best_epoch": fitted["best_epoch"],
            "metrics": metrics,
            "masked": masked,
        },
    )
    print(
        "done",
        out,
        "test sMAE",
        metrics["best"]["test"]["standardized_pollutant_macro_mae"],
    )


if __name__ == "__main__":
    main()
