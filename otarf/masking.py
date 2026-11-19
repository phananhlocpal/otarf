"""Controlled outage of one target's most recent history (6, 24 or 96 hours).

At every forecast origin one target is chosen by a fixed SHA-256 rule, so all models and
all gap lengths see the same target at the same origin. Inside the gap the input value is
held at the last valid observation before the gap; the mask is zero and the age keeps
increasing. Only labels of the selected target are scored.
"""

import hashlib

import numpy as np

GAPS = (6, 24, 96)


def selected_targets(dataset, origins, n):
    return np.array(
        [
            int.from_bytes(
                hashlib.sha256(
                    f"otarf-missing24-v1|{dataset}|{int(t)}".encode()
                ).digest()[:8],
                "big",
            )
            % n
            for t in origins
        ],
        dtype=np.int64,
    )


def corrupt(data, origins, selected, target_ids, gap, context=96):
    if gap not in GAPS or gap > context:
        raise ValueError("Unsupported gap")
    c = data.meta["channels"]
    idx = origins[:, None] - context + np.arange(context)[None, :]
    x = data.values[idx].copy()
    marks = data.marks[idx].copy()
    rows = np.arange(len(origins))
    channels = target_ids[selected]
    prev = origins - gap - 1
    if np.any(prev < 0):
        raise ValueError("Insufficient past before the gap")
    span = np.arange(context - gap, context)[None, :]
    removed = (
        data.mask[origins[:, None] - gap + np.arange(gap)[None, :], channels[:, None]]
        .sum(1)
        .astype(np.int64)
    )
    x[rows[:, None], span, channels[:, None]] = data.values[prev, channels][:, None]
    marks[rows[:, None], span, channels[:, None]] = 0
    marks[rows[:, None], span, (c + channels)[:, None]] = (
        data.age[prev, channels][:, None] + np.arange(1, gap + 1)[None, :]
    )
    for offset in (2 * c, 3 * c):
        marks[rows[:, None], span, (offset + channels)[:, None]] = data.marks[
            prev, offset + channels
        ][:, None]
    return x, marks, removed


def evaluate_masked(
    predict, data, dataset, horizon, gap, destination, batch_size, context=96
):
    """predict(x, mark, origins) -> array [B, horizon, channels] in train-standardized units."""
    meta = data.meta
    ids = np.array([meta["columns"].index(t) for t in meta["targets"]])
    with np.load(data.directory / "observations.npz") as f:
        raw, observed, dates = f["raw"], f["observed"], f["dates"]
    summary = {}
    for split in ("val", "test"):
        start = meta["train_end"] if split == "val" else meta["test_start"]
        end = meta["test_start"] if split == "val" else meta["rows"]
        origins_all = np.arange(start, end - horizon + 1)
        arrays = {k: [] for k in ("count", "abs_z", "signed_z", "selected", "removed")}
        digest = hashlib.sha256()
        for offset in range(0, len(origins_all), batch_size):
            origins = origins_all[offset : offset + batch_size]
            selected = selected_targets(dataset, origins, len(ids))
            x, mark, removed = corrupt(data, origins, selected, ids, gap, context)
            digest.update(x.tobytes(order="C"))
            pred = np.asarray(predict(x, mark, origins), dtype=np.float64)[:, :, ids]
            if (
                pred.shape != (len(origins), horizon, len(ids))
                or not np.isfinite(pred).all()
            ):
                raise ValueError("Invalid masked-history predictions")
            pos = origins[:, None] + np.arange(horizon)[None, :]
            valid = observed[pos][:, :, ids] & (
                np.arange(len(ids))[None, None, :] == selected[:, None, None]
            )
            truth = (
                raw[pos][:, :, ids].astype(np.float64) - data.scaler.mean_[ids]
            ) / data.scaler.scale_[ids]
            err = np.where(valid, pred - truth, 0.0)
            for k, v in [
                ("count", valid.sum(1)),
                ("abs_z", np.abs(err).sum(1)),
                ("signed_z", err.sum(1)),
                ("selected", selected),
                ("removed", removed),
            ]:
                arrays[k].append(v)
        arrays = {k: np.concatenate(v) for k, v in arrays.items()}
        np.savez_compressed(
            destination / f"{split}_gap{gap}.npz",
            **arrays,
            origin_hour=dates[origins_all],
            target_names=np.array(meta["targets"]),
        )
        total, count = arrays["abs_z"].sum(0), arrays["count"].sum(0)
        use = count > 0
        summary[split] = {
            "origins": len(origins_all),
            "smae": float((total[use] / count[use]).mean()),
            "input_sha256": digest.hexdigest(),
            "removed_observations": int(arrays["removed"].sum()),
        }
    return summary
