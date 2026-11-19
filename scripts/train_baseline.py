#!/usr/bin/env python3
"""Train an UNCHANGED official SOFTS or TimeFilter.

--recipe indicator  the comparison used in the paper: the OTARF training recipe (observed-label MAE,
                    masked-history view, checkpoint on validation sMAE) with the mask and log(1+age)
                    of every channel appended as extra variates.
--recipe official   the authors' own trainer and checkpoint rule (optional; not used in the paper).
Every run evaluates natural histories and the controlled 6/24/96-hour outage of one target.

Clone the official repositories first: bash scripts/fetch_baselines.sh
"""
import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from otarf.baselines import OfficialRecipeModel, make_arguments, verify_official_source
from otarf.data import CIRFData, EvaluationAdapter
from otarf.evaluate import evaluate
from otarf.io import read_json, write_json
from otarf.masking import GAPS, evaluate_masked
from otarf.training import fit, set_seed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=("SOFTS", "TimeFilter"))
    p.add_argument(
        "--recipe", default="indicator", choices=("indicator", "official")
    )
    p.add_argument("--dataset", required=True)
    p.add_argument("--horizon", type=int, required=True, choices=(12, 24, 48))
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--config", type=Path, default=ROOT / "configs/baselines.json")
    p.add_argument("--otarf-config", type=Path, default=ROOT / "configs/otarf.json")
    p.add_argument("--lock", type=Path, default=ROOT / "configs/baselines.lock.json")
    p.add_argument("--data-root", type=Path, default=ROOT / "data/processed")
    p.add_argument("--runs", type=Path, default=ROOT / "runs")
    a = p.parse_args()
    for key in ("config", "otarf_config", "lock", "data_root", "runs"):
        setattr(
            a, key, getattr(a, key).resolve()
        )  # the official code is run from its own folder
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required.")
    name, h = a.model, a.horizon
    config = read_json(a.config)
    repo = (ROOT / "external" / name).resolve()
    commit = verify_official_source(repo, name, a.lock)
    directory = (a.data_root / a.dataset).resolve()
    meta = read_json(directory / "metadata.json")
    out = (a.runs / f"{name}-{a.recipe}" / a.dataset / f"h{h}" / f"s{a.seed}").resolve()
    out.mkdir(parents=True, exist_ok=True)
    set_seed(a.seed)
    # TimeFilter's official noisy top-k gate uses a CUDA cumsum without a deterministic kernel; warn_only keeps the run going.
    torch.cuda.reset_peak_memory_stats()
    args = make_arguments(
        name,
        f"{a.dataset}_{name}_h{h}_s{a.seed}",
        h,
        config,
        meta,
        directory,
        out,
        repo,
    )
    indicators = a.recipe == "indicator"
    if indicators:
        args.enc_in = args.dec_in = args.c_out = 3 * meta["channels"]
    args.use_gpu, args.use_multi_gpu, args.gpu = True, False, 0
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast

    exp = Exp_Long_Term_Forecast(args)
    data = CIRFData(directory)
    device = exp.device
    write_json(
        out / "run.json",
        {
            "model": name,
            "recipe": a.recipe,
            "dataset": a.dataset,
            "horizon": h,
            "seed": a.seed,
            "official_commit": commit,
            "official_arguments": {k: str(v) for k, v in vars(args).items()},
            "gpu": torch.cuda.get_device_name(0),
        },
    )
    if a.recipe == "official":
        import time

        torch.cuda.synchronize()
        begin = time.perf_counter()
        exp.train("official")
        torch.cuda.synchronize()
        seconds = time.perf_counter() - begin
        # SOFTS keeps the minimum-validation-MSE checkpoint; TimeFilter's EarlyStopping saves every epoch (last epoch).
        metrics = {
            "best": {
                s: evaluate(exp, name, meta, directory, s, out / "best")
                for s in ("val", "test")
            }
        }
        best_epoch = None
        parameters = sum(p.numel() for p in exp.model.parameters())
        model = exp.model

        def predict(x, mark, origins):
            xt = torch.from_numpy(x).float().to(device)
            with torch.no_grad():
                if name == "TimeFilter":
                    return (
                        model(xt, exp.masks, is_training=False)[0][:, -h:].cpu().numpy()
                    )
                c = meta["channels"]
                calendar = torch.from_numpy(mark[..., 4 * c :]).float().to(device)
                dec = torch.cat(
                    [
                        xt[:, -args.label_len :],
                        torch.zeros((len(x), h, c), device=device),
                    ],
                    dim=1,
                )
                return model(xt, calendar, dec, None)[:, -h:].cpu().numpy()

    else:
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
        rcfg = {
            "seq_len": 96,
            "label_len": 48,
            "train": train_cfg,
            "augmentation": otarf_cfg["augmentation"],
        }
        model = OfficialRecipeModel(
            name,
            exp.model,
            getattr(exp, "masks", None),
            meta["channels"],
            rcfg["label_len"],
            h,
            indicators,
        ).to(device)
        fitted = fit(
            model,
            data,
            rcfg,
            h,
            device,
            out,
            mode="baseline",
            moe_weight=0.05 if name == "TimeFilter" else 0.0,
        )
        seconds, best_epoch = fitted["training_seconds"], fitted["best_epoch"]
        eargs = SimpleNamespace(
            pred_len=h,
            seq_len=96,
            label_len=48,
            batch_size=train_cfg["batch_size"],
            num_workers=0,
        )
        adapter = EvaluationAdapter(model, data, eargs, rcfg, device)
        metrics = {}
        for selection in ("best", "last"):
            model.load_state_dict(
                torch.load(
                    out / "checkpoints" / f"{selection}.pth",
                    map_location=device,
                    weights_only=True,
                )
            )
            # Generic evaluator branch: the model is called as model(x, mark, ...).
            metrics[selection] = {
                s: evaluate(adapter, "OTARF", meta, directory, s, out / selection)
                for s in ("val", "test")
            }
        model.load_state_dict(
            torch.load(
                out / "checkpoints" / "best.pth", map_location=device, weights_only=True
            )
        )
        parameters = sum(p.numel() for p in model.parameters())

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

    model.eval()
    folder = out / "masked"
    folder.mkdir(exist_ok=True)
    masked = {
        str(g): evaluate_masked(predict, data, a.dataset, h, g, folder, args.batch_size)
        for g in GAPS
    }
    write_json(
        out / "result.json",
        {
            "status": "complete",
            "model": name,
            "recipe": a.recipe,
            "dataset": a.dataset,
            "horizon": h,
            "seed": a.seed,
            "parameters": parameters,
            "training_seconds": seconds,
            "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
            "best_epoch": best_epoch,
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
