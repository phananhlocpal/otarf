"""Streaming evaluation outside the models and their trainers.

Copied from the research code; the only addition is the per-origin q95 exceedance
hit/miss counts (tail_tp, tail_fn) used for the tail recall in the paper."""

import time

import numpy as np
import pandas as pd
import torch

from .io import write_json


def evaluate(exp, model_name, meta, directory, split, destination):
    dataset, loader = exp._get_data(split)
    # Official validation loaders shuffle. External origin-indexed exports must not.
    # This does not alter the loader/trainer used for official training or selection.
    from torch.utils.data import DataLoader, SequentialSampler

    loader = DataLoader(
        dataset,
        batch_size=exp.args.batch_size,
        shuffle=False,
        num_workers=exp.args.num_workers,
        drop_last=False,
    )
    assert isinstance(loader.sampler, SequentialSampler)
    with np.load(directory / "observations.npz") as values:
        raw, observed, dates = values["raw"], values["observed"], values["dates"]
    columns = meta["columns"]
    channels = [columns.index(c) for c in meta["targets"]]
    mean, scale = dataset.scaler.mean_, dataset.scaler.scale_
    filled = pd.read_csv(directory / "data.csv")[columns].to_numpy()
    start = meta["test_start"] if split == "test" else meta["train_end"]
    end = meta["rows"] if split == "test" else meta["test_start"]
    horizon, context = exp.args.pred_len, exp.args.seq_len
    expected = end - start - horizon + 1
    if len(dataset) != expected:
        raise ValueError(
            f"Official loader split disagrees: {len(dataset)} != {expected}"
        )
    thresholds = np.nanquantile(raw[: meta["train_end"], channels], 0.95, axis=0)
    count = np.zeros((horizon, len(channels)), np.int64)
    abs_z, sq_z, abs_raw, sq_raw = [
        np.zeros_like(count, dtype=np.float64) for _ in range(4)
    ]
    tail_count, tail_abs = np.zeros_like(count), np.zeros_like(abs_z)
    origin_arrays = {
        k: []
        for k in [
            "tail_tp",
            "tail_fn",
            "count",
            "abs_z",
            "sq_z",
            "abs_raw",
            "sq_raw",
            "tail_count",
            "tail_abs",
            "signed_z",
            "tail_signed_z",
            "tail_abs_z",
            "tail_sq_z",
            "ramp_count",
            "ramp_abs_z",
            "ramp_signed_z",
            "stable_count",
            "stable_abs_z",
            "context_missing",
            "origin_hour",
        ]
    }
    full_sq, full_abs, full_count, offset = 0.0, 0.0, 0, 0
    leakage = []
    exact_removed = []
    hooks = []
    if model_name == "TimeFilter" and split == "val":

        def graph_hook(_module, _inputs, result):
            logits = result[0].detach()
            # Zeroed edges before softmax receive exp(0)>0, measurable without altering code.
            mass = torch.softmax(logits, dim=-1).masked_fill(logits != 0, 0).sum(-1)
            leakage.append((float(mass.sum().item()), mass.numel()))

        for module in exp.model.modules():
            if module.__class__.__name__ == "GraphLearner":
                hooks.append(module.register_forward_hook(graph_hook))
            if module.__class__.__name__ == "mask_moe":

                def mask_hook(_module, inputs, result):
                    region_mask = result[0].detach()
                    logits = inputs[0].detach()
                    # These tensors are actual upstream values; diagnostic only, no mutation.
                    b, h, l, _ = region_mask.shape
                    final = logits.reshape(b, h, l, l) * region_mask
                    mass = (
                        torch.softmax(final, dim=-1)
                        .masked_fill(region_mask != 0, 0)
                        .sum(-1)
                    )
                    exact_removed.append(
                        (
                            float(mass.sum().item()),
                            mass.numel(),
                            int((region_mask == 0).sum().item()),
                            region_mask.numel(),
                        )
                    )

                hooks.append(module.register_forward_hook(mask_hook))
    exp.model.eval()
    torch.cuda.synchronize()
    begin = time.perf_counter()
    try:
        with torch.no_grad():
            for x, y, xmark, ymark in loader:
                x = x.float().to(exp.device)
                if model_name == "TimeFilter":
                    output, _ = exp.model(x, exp.masks, is_training=False)
                else:
                    decoder = (
                        torch.cat(
                            [
                                y[:, : exp.args.label_len],
                                torch.zeros_like(y[:, -horizon:]),
                            ],
                            dim=1,
                        )
                        .float()
                        .to(exp.device)
                    )
                    output = exp.model(
                        x,
                        xmark.float().to(exp.device),
                        decoder,
                        ymark.float().to(exp.device),
                    )
                pred_all = (
                    output[:, -horizon:].detach().cpu().numpy().astype(np.float64)
                )
                if not np.isfinite(pred_all).all():
                    raise FloatingPointError("Non-finite official model prediction")
                truth_all = y[:, -horizon:].float().numpy().astype(np.float64)
                delta_all = pred_all - truth_all
                full_sq += np.square(delta_all).sum()
                full_abs += np.abs(delta_all).sum()
                full_count += delta_all.size
                b = len(x)
                origins = np.arange(start + offset, start + offset + b)
                positions = origins[:, None] + np.arange(horizon)[None, :]
                expected_y = ((filled[positions] - mean) / scale).astype(np.float32)
                expected_x = (
                    (
                        filled[origins[:, None] - context + np.arange(context)[None, :]]
                        - mean
                    )
                    / scale
                ).astype(np.float32)
                if not np.allclose(truth_all, expected_y, rtol=1e-5, atol=1e-5):
                    raise ValueError("Forecast-origin/target alignment failed")
                if not np.allclose(
                    x.detach().cpu().numpy(), expected_x, rtol=1e-5, atol=1e-5
                ):
                    raise ValueError("Forecast-origin/input alignment failed")
                target_raw = raw[positions][:, :, channels].astype(np.float64)
                valid = observed[positions][:, :, channels]
                predicted = pred_all[:, :, channels]
                target_z = (target_raw - mean[channels]) / scale[channels]
                dz = np.where(valid, predicted - target_z, 0.0)
                dr = dz * scale[channels]
                tail = valid & (target_raw >= thresholds[None, None, :])
                last_raw = raw[origins - 1][:, channels]
                last_valid = observed[origins - 1][:, channels]
                change = (target_raw - last_raw[:, None, :]) / scale[channels]
                ramp = valid & last_valid[:, None, :] & (change >= 1.0)
                stable = valid & last_valid[:, None, :] & (np.abs(change) <= 0.25)
                current = {
                    "count": valid.sum(1),
                    "abs_z": np.abs(dz).sum(1),
                    "sq_z": np.square(dz).sum(1),
                    "abs_raw": np.abs(dr).sum(1),
                    "sq_raw": np.square(dr).sum(1),
                    "tail_count": tail.sum(1),
                    "tail_abs": (np.abs(dr) * tail).sum(1),
                    "signed_z": dz.sum(1),
                    "tail_signed_z": (dz * tail).sum(1),
                    "tail_abs_z": (np.abs(dz) * tail).sum(1),
                    "tail_sq_z": (np.square(dz) * tail).sum(1),
                    "ramp_count": ramp.sum(1),
                    "ramp_abs_z": (np.abs(dz) * ramp).sum(1),
                    "ramp_signed_z": (dz * ramp).sum(1),
                    "stable_count": stable.sum(1),
                    "stable_abs_z": (np.abs(dz) * stable).sum(1),
                    # Release addition: q95 exceedance hits/misses (tail recall), no effect on other outputs.
                    "tail_tp": (
                        tail
                        & (
                            (predicted * scale[channels] + mean[channels])
                            >= thresholds[None, None, :]
                        )
                    ).sum(1),
                    "tail_fn": (
                        tail
                        & (
                            (predicted * scale[channels] + mean[channels])
                            < thresholds[None, None, :]
                        )
                    ).sum(1),
                    "context_missing": np.array(
                        [1.0 - observed[o - context : o].mean() for o in origins]
                    ),
                    "origin_hour": dates[origins],
                }
                for key in origin_arrays:
                    origin_arrays[key].append(current[key])
                count += valid.sum(0)
                abs_z += np.abs(dz).sum(0)
                sq_z += np.square(dz).sum(0)
                abs_raw += np.abs(dr).sum(0)
                sq_raw += np.square(dr).sum(0)
                tail_count += tail.sum(0)
                tail_abs += (np.abs(dr) * tail).sum(0)
                offset += b
    finally:
        for hook in hooks:
            hook.remove()
    torch.cuda.synchronize()
    seconds = time.perf_counter() - begin
    if offset != expected:
        raise ValueError("Dropped or repeated forecast origins")
    measurable = count.sum(0) > 0
    if not measurable.any():
        raise ValueError("No observed pollutant labels")
    destination.mkdir(parents=True, exist_ok=True)
    arrays = {key: np.concatenate(value) for key, value in origin_arrays.items()}
    np.savez_compressed(
        destination / f"{split}_origin_errors.npz",
        **arrays,
        target_names=np.array(meta["targets"]),
    )
    rows = []
    for j, name in enumerate(meta["targets"]):
        for h in range(horizon):
            c = int(count[h, j])
            rows.append(
                {
                    "target": name,
                    "pollutant": meta["target_groups"][name],
                    "lead_hour": h + 1,
                    "count": c,
                    "mae": float(abs_raw[h, j] / c) if c else None,
                    "mse": float(sq_raw[h, j] / c) if c else None,
                    "rmse": float(np.sqrt(sq_raw[h, j] / c)) if c else None,
                    "standardized_mae": float(abs_z[h, j] / c) if c else None,
                    "standardized_mse": float(sq_z[h, j] / c) if c else None,
                    "tail_count": int(tail_count[h, j]),
                    "tail_mae": (
                        float(tail_abs[h, j] / tail_count[h, j])
                        if tail_count[h, j]
                        else None
                    ),
                }
            )
    pd.DataFrame(rows).to_csv(destination / f"{split}_by_target_lead.csv", index=False)
    sums = count.sum(0)
    stats = {
        "split": split,
        "forecast_origins": offset,
        "horizon": horizon,
        "standardized_pollutant_macro_mae": float(
            np.mean(abs_z.sum(0)[measurable] / sums[measurable])
        ),
        "standardized_pollutant_macro_mse": float(
            np.mean(sq_z.sum(0)[measurable] / sums[measurable])
        ),
        "standardized_pollutant_macro_rmse": float(
            np.mean(np.sqrt(sq_z.sum(0)[measurable] / sums[measurable]))
        ),
        "targets_without_observations": [
            meta["targets"][j] for j in range(len(channels)) if not measurable[j]
        ],
        "origin_alignment": "sequential sampler; every input and target window checked against train-scaled chronological CSV",
        "official_all_imputed_channels_mse": float(full_sq / full_count),
        "official_all_imputed_channels_mae": float(full_abs / full_count),
        "observed_labels_in_overlapping_windows": int(count.sum()),
        "evaluation_seconds_including_diagnostics": seconds,
        "end_to_end_origins_per_second": offset / seconds,
        "raw_metrics": {},
    }
    for pollutant in sorted(set(meta["target_groups"].values())):
        ids = [
            j
            for j, name in enumerate(meta["targets"])
            if meta["target_groups"][name] == pollutant
        ]
        total = int(count[:, ids].sum())
        tail_total = int(tail_count[:, ids].sum())
        stats["raw_metrics"][pollutant] = {
            "count": total,
            "mae": float(abs_raw[:, ids].sum() / total) if total else None,
            "rmse": float(np.sqrt(sq_raw[:, ids].sum() / total)) if total else None,
            "tail_count": tail_total,
            "tail_mae": (
                float(tail_abs[:, ids].sum() / tail_total) if tail_total else None
            ),
        }
    if leakage:
        stats["timefilter_zero_logit_softmax_mass"] = sum(x for x, n in leakage) / sum(
            n for x, n in leakage
        )
        stats["graph_diagnostic_scope"] = (
            "Full validation split; zero logits include removed edges and exact zeros. This is a code diagnostic, not an accuracy conclusion."
        )
    if exact_removed:
        stats["timefilter_exact_region_removed_softmax_mass"] = sum(
            v[0] for v in exact_removed
        ) / sum(v[1] for v in exact_removed)
        stats["timefilter_exact_region_removed_edge_fraction"] = sum(
            v[2] for v in exact_removed
        ) / sum(v[3] for v in exact_removed)
    write_json(destination / f"{split}_metrics.json", stats)
    return stats
