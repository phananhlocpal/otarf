#!/usr/bin/env python3
"""Inference only: where the cross-variable branch takes its information from, and the 96-hour case study.

For one dataset, using the trained runs in runs/ (OTARF H = 24 seeds 2021-2023; SOFTS / TimeFilter indicator
seed 2021 for the case study):
  * attention mass per source group (same station / other pollutant, same pollutant / other station,
    other station / other pollutant, meteorology) and the null weight, on test, for natural histories and
    the 24 h / 96 h outage of the hashed target;
  * knockout: the sources of one group (or of all groups) are removed from the valid set and the test sMAE
    is recomputed;
  * the case study at the origin listed in configs/case_selection.json.
Outputs go to runs/inspect/<dataset>/; scripts/summarize.py and scripts/make_figures.py read them.

  python scripts/inspect_otarf.py --dataset kdd2018_london
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from otarf.baselines import OfficialRecipeModel, make_arguments, verify_official_source
from otarf.core import OTARFCore
from otarf.data import CIRFData
from otarf.inspect import OTARFCoreInspect
from otarf.io import read_json, write_json
from otarf.masking import corrupt, selected_targets

POLLUTANTS = {"PM2.5", "PM10", "SO2", "NO2", "CO", "O3"}
GROUPS = ["same_station_other_pollutant", "same_pollutant_other_station", "other_station_other_pollutant", "meteorology"]


def parse(name):
    if "__" in name:
        return tuple(name.split("__", 1))
    if name.startswith("PM_"):
        return name[3:], "PM2.5"
    return None, name


def group_matrix(columns):
    info = [parse(c) for c in columns]
    C = len(columns)
    g = np.full((C, C), -1, np.int64)
    for i, (si, vi) in enumerate(info):
        for j, (sj, vj) in enumerate(info):
            if i == j:
                continue
            g[i, j] = 3 if vj not in POLLUTANTS else 0 if sj == si else 1 if vj == vi else 2
    return g


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--config", type=Path, default=ROOT / "configs/otarf.json")
    p.add_argument("--baseline-config", type=Path, default=ROOT / "configs/baselines.json")
    p.add_argument("--lock", type=Path, default=ROOT / "configs/baselines.lock.json")
    p.add_argument("--cases", type=Path, default=ROOT / "configs/case_selection.json")
    p.add_argument("--data-root", type=Path, default=ROOT / "data/processed")
    p.add_argument("--runs", type=Path, default=ROOT / "runs")
    p.add_argument("--batch-size", type=int, default=64)
    a = p.parse_args()
    for key in ("config", "baseline_config", "lock", "cases", "data_root", "runs"):
        setattr(a, key, getattr(a, key).resolve())
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required.")
    device = torch.device("cuda:0")
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = read_json(a.config)
    directory = a.data_root / a.dataset
    data = CIRFData(directory)
    meta = data.meta
    C, h = meta["channels"], 24
    ids = np.array([meta["columns"].index(t) for t in meta["targets"]])
    with np.load(directory / "observations.npz") as f:
        raw, observed, dates = f["raw"].copy(), f["observed"].copy(), f["dates"].copy()
    out = a.runs / "inspect" / a.dataset
    out.mkdir(parents=True, exist_ok=True)
    gm = group_matrix(meta["columns"])
    K = len(GROUPS)
    onehot = torch.zeros((C, C, K), device=device)
    for k in range(K):
        onehot[..., k] = torch.as_tensor(gm == k, device=device).float()
    excludes = {name: torch.as_tensor(gm == k, device=device) for k, name in enumerate(GROUPS) if (gm == k).any()}
    excludes["all_sources"] = torch.as_tensor(gm >= 0, device=device)
    sets = ["none"] + list(excludes)
    origins_all = np.arange(meta["test_start"], meta["rows"] - h + 1)
    summary = {"groups": GROUPS, "excluded_sets": sets, "equality_max_abs": 0.0}
    for seed in (2021, 2022, 2023):
        model = OTARFCoreInspect(C, h, config).to(device).eval()
        ck = a.runs / "OTARF-otarf" / a.dataset / "h24" / f"s{seed}" / "checkpoints" / "best.pth"
        model.load_state_dict(torch.load(ck, map_location=device, weights_only=True), strict=True)
        for scenario in ("natural", "gap24", "gap96"):
            arrays = {s: {"abs_z": [], "count": []} for s in sets}
            att = {"mass": [], "candidates": [], "null": []}
            with torch.no_grad():
                for off in range(0, len(origins_all), a.batch_size):
                    origins = origins_all[off:off + a.batch_size]
                    sel = selected_targets(a.dataset, origins, len(ids))
                    if scenario == "natural":
                        idx = origins[:, None] - 96 + np.arange(96)[None, :]
                        x, mark = data.values[idx], data.marks[idx]
                    else:
                        x, mark, _ = corrupt(data, origins, sel, ids, int(scenario[3:]))
                    xt = torch.from_numpy(np.ascontiguousarray(x)).float().to(device)
                    mt = torch.from_numpy(np.ascontiguousarray(mark)).float().to(device)
                    pos = origins[:, None] + np.arange(h)[None, :]
                    valid = observed[pos][:, :, ids]
                    if scenario != "natural":
                        valid = valid & (np.arange(len(ids))[None, None, :] == sel[:, None, None])
                    z = (raw[pos][:, :, ids].astype(np.float64) - data.scaler.mean_[ids]) / data.scaler.scale_[ids]
                    ref = OTARFCore.forward(model, xt, mt)
                    for s in sets:
                        fc, info = model.forward_inspect(xt, mt, onehot, None if s == "none" else excludes[s])
                        if s == "none":
                            diff = float((fc - ref).abs().max().item())
                            if diff > 1e-5:
                                raise ValueError(f"Inspection forward differs from the model forward: {diff}")
                            summary["equality_max_abs"] = max(summary["equality_max_abs"], diff)
                            m, cd, nl = (info[k].cpu().numpy() for k in ("mass", "candidates", "null"))
                            if scenario == "natural":
                                att["mass"].append(m[:, ids]); att["candidates"].append(cd[:, ids]); att["null"].append(nl[:, ids])
                            else:
                                r, rows = ids[sel], np.arange(len(origins))
                                att["mass"].append(m[rows, r][:, None]); att["candidates"].append(cd[rows, r][:, None]); att["null"].append(nl[rows, r][:, None])
                        err = np.where(valid, fc.cpu().numpy().astype(np.float64)[:, :, ids] - z, 0.0)
                        arrays[s]["abs_z"].append(np.abs(err).sum(1))
                        arrays[s]["count"].append(valid.sum(1))
            payload = {f"{s}__{k}": np.concatenate(v) for s, d in arrays.items() for k, v in d.items()}
            payload.update({f"att_{k}": np.concatenate(v) for k, v in att.items()})
            np.savez_compressed(out / f"inspect_s{seed}_{scenario}.npz", **payload, origin_hour=dates[origins_all],
                                target_names=np.array(meta["targets"]))
            print("done", a.dataset, seed, scenario, flush=True)
    # Case study (seed 2021).
    case = next(c for c in read_json(a.cases) if c["dataset"] == a.dataset)
    o = np.array([case["origin_index"]])
    sel = selected_targets(a.dataset, o, len(ids))
    if int(sel[0]) != case["target_position"]:
        raise ValueError("Case target does not match the hash rule")
    x, mark, _ = corrupt(data, o, sel, ids, 96)
    xt, mt = torch.from_numpy(x).float().to(device), torch.from_numpy(mark).float().to(device)
    core = OTARFCore(C, h, config, "core").to(device).eval()
    core.load_state_dict(torch.load(a.runs / "OTARF-otarf" / a.dataset / "h24" / "s2021" / "checkpoints" / "best.pth", map_location=device, weights_only=True))
    preds = {}
    with torch.no_grad():
        preds["OTARF"] = core(xt, mt).cpu().numpy()[0]
    bconfig = read_json(a.baseline_config)
    here = os.getcwd()
    for name in ("SOFTS", "TimeFilter"):
        repo = (ROOT / "external" / name).resolve()
        verify_official_source(repo, name, a.lock)
        args = make_arguments(name, f"{a.dataset}_{name}_case", h, bconfig, meta, directory, out / f"case_{name}", repo)
        args.enc_in = args.dec_in = args.c_out = 3 * C
        args.use_gpu, args.use_multi_gpu, args.gpu = True, False, 0
        for mod in [m for m in sys.modules if m == "exp" or m.startswith(("exp.", "models", "layers", "utils", "data_provider"))]:
            del sys.modules[mod]
        sys.path.insert(0, str(repo))
        os.chdir(repo)
        from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
        exp = Exp_Long_Term_Forecast(args)
        model = OfficialRecipeModel(name, exp.model, getattr(exp, "masks", None), C, 48, h, True).to(device).eval()
        ck = a.runs / f"{name}-indicator" / a.dataset / "h24" / "s2021" / "checkpoints" / "best.pth"
        model.load_state_dict(torch.load(ck, map_location=device, weights_only=True), strict=True)
        with torch.no_grad():
            preds[name] = model(xt, mt).cpu().numpy()[0]
        os.chdir(here)
        sys.path.remove(str(repo))
    j = int(ids[sel[0]])
    pos = o[0] + np.arange(h)
    truth = (raw[pos, j].astype(np.float64) - data.scaler.mean_[j]) / data.scaler.scale_[j]
    summary["case_check"] = {m: float(np.mean(np.abs(p[:, j] - truth)[observed[pos, j]])) for m, p in preds.items()}
    hist = np.arange(o[0] - 240, o[0])
    scale, mean = data.scaler.scale_[j], data.scaler.mean_[j]
    np.savez_compressed(out / "case_96h.npz", history_raw=raw[hist, j], history_observed=observed[hist, j],
                        input_raw=x[0, :, j].astype(np.float64) * scale + mean, future_raw=raw[pos, j],
                        future_observed=observed[pos, j], **{f"pred_{m}": v[:, j].astype(np.float64) * scale + mean for m, v in preds.items()})
    write_json(out / "summary.json", summary)
    print("case sMAE", summary["case_check"], "expected", case["stored_smae"])


if __name__ == "__main__":
    main()
