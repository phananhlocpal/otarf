"""Training recipe shared by OTARF, its variants and the retrained baselines (copied from the research runners)."""

import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .augmentation import corrupt
from .data import TrainObserved
from .io import write_json


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


class ObservedLoss:
    """Uniform-origin estimator of the target-macro MAE on observed labels (Eq. loss in the paper)."""

    def __init__(self, data, windows, train_data, horizon, device, peak=None):
        """peak = {"quantile": q, "weight": w}: observed labels at or above the train q-quantile of their own
        target (standardized, observed train labels only) get weight w; all others weight 1. Not renormalized."""
        meta = data.meta
        self.ids = np.array(
            [meta["columns"].index(n) for n in meta["targets"]], dtype=np.int64
        )
        self.target_indices = torch.as_tensor(self.ids, device=device)
        with np.load(data.directory / "observations.npz") as obs:
            self.raw = obs["raw"].copy()
            self.observed = obs["observed"].copy()
        ids, train_end = self.ids, meta["train_end"]
        train_obs = self.observed[:train_end, ids]
        train_z = (
            self.raw[:train_end, ids].astype(np.float64) - data.scaler.mean_[ids]
        ) / data.scaler.scale_[ids]
        if not np.allclose(
            data.values[:train_end, ids][train_obs],
            train_z[train_obs],
            rtol=1e-5,
            atol=1e-5,
        ):
            raise ValueError("Observed train labels disagree with original data")
        prefix = np.vstack(
            [
                np.zeros((1, len(ids)), dtype=np.int64),
                train_obs.cumsum(axis=0, dtype=np.int64),
            ]
        )
        starts = np.arange(windows.start, windows.end - horizon + 1)
        if len(starts) != len(train_data) or starts[-1] + horizon > train_end:
            raise ValueError("Train origin bounds mismatch")
        totals = (prefix[starts + horizon] - prefix[starts]).sum(axis=0)
        active = totals > 0
        if not active.any():
            raise ValueError("No observed training targets")
        self.weights = torch.as_tensor(
            np.where(active, 1.0 / np.maximum(totals, 1), 0.0),
            dtype=torch.float32,
            device=device,
        )
        self.active_count = int(active.sum())
        self.n_origins = len(starts)
        self.peak_weight = None
        if peak is not None:
            thresholds = np.array(
                [
                    np.quantile(train_z[train_obs[:, k], k], peak["quantile"])
                    if train_obs[:, k].any()
                    else np.finfo(np.float32).max
                    for k in range(len(ids))
                ],
                dtype=np.float64,
            )
            self.peak_threshold = torch.as_tensor(thresholds, dtype=torch.float32, device=device)
            self.peak_weight = float(peak["weight"])
            self.peak_thresholds_z = thresholds

    def __call__(self, pred, truth, future_mask):
        p = pred.index_select(2, self.target_indices)
        y = truth.index_select(2, self.target_indices)
        m = future_mask.index_select(2, self.target_indices)
        error = (p - y).abs() * m
        if self.peak_weight is not None:
            error = error * (1.0 + (self.peak_weight - 1.0) * (y >= self.peak_threshold).to(p.dtype))
        return (error.sum((0, 1)) * self.weights).sum() * (
            self.n_origins / (p.shape[0] * self.active_count)
        )


def validation_smae(model, loader, data, loss, horizon, config, device):
    meta, ids, raw, observed = data.meta, loss.ids, loss.raw, loss.observed
    model.eval()
    val_abs = np.zeros(len(ids))
    val_counts = np.zeros(len(ids), np.int64)
    offset = 0
    with torch.no_grad():
        for x, y, mark, _ in loader:
            b = len(x)
            pred = model(x.to(device), mark.to(device))
            origins = np.arange(
                meta["train_end"] + offset, meta["train_end"] + offset + b
            )
            positions = origins[:, None] + np.arange(horizon)[None, :]
            if positions.max() >= meta["test_start"]:
                raise ValueError("Validation target crosses test boundary")
            if not np.allclose(
                x.numpy(),
                data.values[
                    origins[:, None]
                    - config["seq_len"]
                    + np.arange(config["seq_len"])[None, :]
                ],
                rtol=1e-5,
                atol=1e-5,
            ):
                raise ValueError("Validation input alignment")
            mask = observed[positions][:, :, ids]
            target_z = (
                raw[positions][:, :, ids].astype(np.float64) - data.scaler.mean_[ids]
            ) / data.scaler.scale_[ids]
            error = np.where(
                mask,
                pred.detach().cpu().numpy().astype(np.float64)[:, :, ids] - target_z,
                0.0,
            )
            val_abs += np.abs(error).sum((0, 1))
            val_counts += mask.sum((0, 1))
            offset += b
    active = val_counts > 0
    return float(np.mean(val_abs[active] / val_counts[active]))


def fit(
    model,
    data,
    config,
    horizon,
    device,
    out,
    mode,
    natural_only=False,
    clean_first=False,
    moe_weight=0.0,
    peak=None,
):
    """mode='otarf': joint backward with correction penalty (OTARF and its variants).
    mode='baseline': per-view backward with weight 1/2 (identical gradient, half the peak memory) and the
    official TimeFilter auxiliary loss weighted by moe_weight."""
    train = config["train"]
    datasets = {s: data.dataset(s, config, horizon) for s in ("train", "val")}
    train_data = TrainObserved(datasets["train"])
    loaders = {
        "train": DataLoader(
            train_data,
            batch_size=train["batch_size"],
            shuffle=True,
            num_workers=0,
            drop_last=False,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=train["batch_size"],
            shuffle=False,
            num_workers=0,
            drop_last=False,
        ),
    }
    loss_fn = ObservedLoss(data, datasets["train"], train_data, horizon, device, peak)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train["learning_rate"],
        weight_decay=train["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train["epochs"]
    )
    checkpoint = out / "checkpoints"
    checkpoint.mkdir(parents=True, exist_ok=True)
    begin = time.perf_counter()
    best = float("inf")
    curve = []
    steps = 0
    reg = config.get("architecture", {}).get("correction_regularization", 0.0)
    for epoch in range(1, train["epochs"] + 1):
        model.train()
        total = 0.0
        num = 0
        start = time.perf_counter()
        for x, y, mark, _, prefix_values, future_observed in loaders["train"]:
            x = x.to(device)
            mark = mark.to(device)
            truth = y[:, -horizon:].to(device)
            future_observed = future_observed.to(device)
            if mode == "otarf":
                if not clean_first and not natural_only:
                    aug_x, aug_mark = corrupt(x, mark, prefix_values.to(device), config)
                optimizer.zero_grad(set_to_none=True)
                pred = model(x, mark)
                penalty = model.correction_penalty
                if natural_only:
                    aug_pred, aug_penalty = pred, penalty
                else:
                    if clean_first:
                        aug_x, aug_mark = corrupt(
                            x, mark, prefix_values.to(device), config
                        )
                    aug_pred = model(aug_x, aug_mark)
                    aug_penalty = model.correction_penalty
                observed_loss = (
                    loss_fn(pred, truth, future_observed)
                    + loss_fn(aug_pred, truth, future_observed)
                ) / 2
                auxiliary = (
                    (pred - truth).square().mean() + (aug_pred - truth).square().mean()
                ) / 2
                loss = (
                    observed_loss
                    + train["auxiliary_weight"] * auxiliary
                    + reg * (penalty + aug_penalty) / 2
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite loss")
                loss.backward()
                batch_loss = loss.item()
            else:
                optimizer.zero_grad(set_to_none=True)
                batch_loss = 0.0
                for view_x, view_mark in (
                    (x, mark),
                    corrupt(x, mark, prefix_values.to(device), config),
                ):
                    pred = model(view_x, view_mark)
                    aux = model.aux_loss
                    view_loss = (
                        loss_fn(pred, truth, future_observed)
                        + train["auxiliary_weight"] * (pred - truth).square().mean()
                        + moe_weight * aux
                    ) / 2
                    if not torch.isfinite(view_loss):
                        raise FloatingPointError("Nonfinite loss")
                    view_loss.backward()
                    batch_loss += view_loss.item()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), train["clip_grad_norm"], error_if_nonfinite=True
            )
            optimizer.step()
            steps += 1
            total += batch_loss * len(x)
            num += len(x)
        val_smae = validation_smae(
            model, loaders["val"], data, loss_fn, horizon, config, device
        )
        if not np.isfinite(val_smae):
            raise FloatingPointError("Nonfinite validation metric")
        improved = val_smae < best
        if improved:
            best = val_smae
            torch.save(model.state_dict(), checkpoint / "best.pth")
        record = {
            "epoch": epoch,
            "train_objective": total / num,
            "validation_observed_macro_smae": val_smae,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": time.perf_counter() - start,
            "saved_best": improved,
        }
        curve.append(record)
        write_json(out / "learning_curve.json", curve)
        print(record, flush=True)
        scheduler.step()
    torch.save(model.state_dict(), checkpoint / "last.pth")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if steps != train["epochs"] * len(loaders["train"]):
        raise ValueError("Unexpected optimizer step count")
    best_epoch = min(curve, key=lambda r: r["validation_observed_macro_smae"])["epoch"]
    return {
        "training_seconds": time.perf_counter() - begin,
        "best_epoch": best_epoch,
        "curve": curve,
        "optimizer_steps": steps,
    }
