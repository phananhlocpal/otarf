#!/usr/bin/env python3
"""Train an UNCHANGED official iTransformer or CRIB under the OTARF training recipe (extended comparison).

With --no-window-norm, any of SOFTS, TimeFilter, iTransformer and CRIB is trained with its window normalization
switched off (otarf.nonorm; mechanism test of the paper, H = 24) and written to runs/<model>-nonorm/.

iTransformer receives the mask and log(1+age) of every channel as extra variates, like SOFTS and TimeFilter.
CRIB keeps the authors' input convention (series multiplied by the mask, no extra variates, no age) and its two
auxiliary official loss terms; it uses a 96-hour window instead of the authors' 24 hours. Every run evaluates
natural histories and the controlled 6/24/96-hour outage of one target.

Clone the official repositories first: bash scripts/fetch_baselines.sh
Example: python scripts/train_extended_baseline.py --model CRIB --dataset kdd2018_london --horizon 24 --seed 2021
         python scripts/train_extended_baseline.py --model SOFTS --dataset kdd2018_london --horizon 24 --seed 2021 --no-window-norm
"""
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from otarf.baselines import build_recipe_baseline, verify_official_source
from otarf.data import CIRFData, EvaluationAdapter
from otarf.evaluate import evaluate
from otarf.io import read_json, write_json
from otarf.masking import GAPS, evaluate_masked
from otarf.training import fit, set_seed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=("iTransformer", "CRIB", "SOFTS", "TimeFilter"))
    p.add_argument("--dataset", required=True)
    p.add_argument("--horizon", type=int, required=True, choices=(12, 24, 48))
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--config", type=Path, default=ROOT / "configs/baselines.json")
    p.add_argument("--otarf-config", type=Path, default=ROOT / "configs/otarf.json")
    p.add_argument("--lock", type=Path, default=ROOT / "configs/baselines.lock.json")
    p.add_argument("--data-root", type=Path, default=ROOT / "data/processed")
    p.add_argument("--runs", type=Path, default=ROOT / "runs")
    p.add_argument("--no-window-norm", action="store_true", help="switch off the window normalization (mechanism test)")
    a = p.parse_args()
    if a.model in ("SOFTS", "TimeFilter") and not a.no_window_norm:
        raise SystemExit("SOFTS and TimeFilter with normalization are trained by scripts/train_baseline.py --recipe indicator")
    for key in ("config", "otarf_config", "lock", "data_root", "runs"):
        setattr(a, key, getattr(a, key).resolve())
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required.")
    name, h = a.model, a.horizon
    config = read_json(a.config)
    repo = (ROOT / "external" / name).resolve()
    commit = verify_official_source(repo, name, a.lock)
    directory = (a.data_root / a.dataset).resolve()
    meta = read_json(directory / "metadata.json")
    tag = f"{name}-nonorm" if a.no_window_norm else f"{name}-indicator"
    out = (a.runs / tag / a.dataset / f"h{h}" / f"s{a.seed}").resolve()
    out.mkdir(parents=True, exist_ok=True)
    set_seed(a.seed)
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda:0")
    data = CIRFData(directory)
    model = build_recipe_baseline(name, repo, config, meta, directory, out, h, device, window_norm=not a.no_window_norm)
    write_json(
        out / "run.json",
        {
            "model": name,
            "recipe": "authors-input (zero-filled, no age)" if name == "CRIB" else "indicator",
            "window_normalization": "off" if a.no_window_norm else "official",
            "dataset": a.dataset,
            "horizon": h,
            "seed": a.seed,
            "official_commit": commit,
            "official_arguments": config["model_args"][name],
            "gpu": torch.cuda.get_device_name(0),
        },
    )
    otarf_cfg = read_json(a.otarf_config)
    mc = config["model_args"][name]
    train_cfg = {
        "epochs": otarf_cfg["train"]["epochs"],
        "batch_size": mc["batch_size"],
        "learning_rate": mc["learning_rate"],
        "weight_decay": 0.0,
        "clip_grad_norm": otarf_cfg["train"]["clip_grad_norm"],
        "auxiliary_weight": otarf_cfg["train"]["auxiliary_weight"],
    }
    rcfg = {"seq_len": 96, "label_len": 48, "train": train_cfg, "augmentation": otarf_cfg["augmentation"]}
    fitted = fit(model, data, rcfg, h, device, out, mode="baseline", moe_weight={"CRIB": 1.0, "TimeFilter": 0.05}.get(name, 0.0))
    eargs = SimpleNamespace(pred_len=h, seq_len=96, label_len=48, batch_size=train_cfg["batch_size"], num_workers=0)
    adapter = EvaluationAdapter(model, data, eargs, rcfg, device)
    metrics = {}
    for selection in ("best", "last"):
        model.load_state_dict(
            torch.load(out / "checkpoints" / f"{selection}.pth", map_location=device, weights_only=True)
        )
        metrics[selection] = {
            s: evaluate(adapter, "OTARF", meta, directory, s, out / selection) for s in ("val", "test")
        }
    model.load_state_dict(torch.load(out / "checkpoints" / "best.pth", map_location=device, weights_only=True))
    model.eval()

    def predict(x, mark, origins):
        with torch.no_grad():
            return model(torch.from_numpy(x).float().to(device), torch.from_numpy(mark).float().to(device)).cpu().numpy()

    folder = out / "masked"
    folder.mkdir(exist_ok=True)
    masked = {str(g): evaluate_masked(predict, data, a.dataset, h, g, folder, train_cfg["batch_size"]) for g in GAPS}
    write_json(
        out / "result.json",
        {
            "status": "complete",
            "model": name,
            "recipe": "indicator",
            "window_normalization": "off" if a.no_window_norm else "official",
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
    print("done", out, "test nMAE", metrics["best"]["test"]["standardized_pollutant_macro_mae"])


if __name__ == "__main__":
    main()
