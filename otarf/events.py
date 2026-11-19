"""Exceedance events on the 24-hour mean (paper, Sect. 5.3 and 6.5).

For every test origin at H = 24 and every PM2.5 / PM10 target, the forecast daily mean is the mean of the 24 hourly
forecasts and the observed daily mean is the mean of the observed hours in the same 24 hours. An interval is scored
only when at least MIN_HOURS of its 24 hours are observed. Scenarios: natural history, and a 96-hour outage of one
target (same corruption and target selection as otarf.masking). Per-origin daily means are stored, so thresholds can
be applied without running the model again.
"""
import numpy as np

from .masking import corrupt, selected_targets
from .outages import parse

HORIZON = 24
VARIABLES = ("PM2.5", "PM10")
MIN_HOURS = {"beijing_multisite": 20, "shanghai_confirmation": 20, "kdd2018_london": 18}
# PM2.5: GB 3095-2012 Grade II 24-hour limit (Beijing, Shanghai); start of the UK DAQI "Moderate" band (London).
THRESHOLDS = {"beijing_multisite": {"PM2.5": 75.0, "PM10": 150.0}, "shanghai_confirmation": {"PM2.5": 75.0, "PM10": 150.0},
              "kdd2018_london": {"PM2.5": 36.0, "PM10": 51.0}}


def daily_means(predict, data, dataset, scenario, batch_size, context=96):
    """Returns per-origin forecast and observed 24-hour means of the PM targets for scenario 'natural' or 'single96'."""
    meta, h = data.meta, HORIZON
    ids = np.array([meta["columns"].index(t) for t in meta["targets"]])
    n_targets = len(ids)
    variables = [parse(n)[1] for n in meta["targets"]]
    keep = np.array([k for k, v in enumerate(variables) if v in VARIABLES], dtype=np.int64)
    if not len(keep):
        raise ValueError("No PM2.5 or PM10 target")
    with np.load(data.directory / "observations.npz") as f:
        raw, observed = f["raw"], f["observed"]
    mean, scale = data.scaler.mean_[ids], data.scaler.scale_[ids]
    origins_all = np.arange(meta["test_start"], meta["rows"] - h + 1)
    pred_mean, true_mean, hours, scored = [], [], [], []
    for start in range(0, len(origins_all), batch_size):
        origins = origins_all[start:start + batch_size]
        if scenario == "natural":
            idx = origins[:, None] - context + np.arange(context)[None, :]
            x, mark = data.values[idx].copy(), data.marks[idx].copy()
            evaluated = np.ones((len(origins), n_targets), bool)
        elif scenario == "single96":
            selected = selected_targets(dataset, origins, n_targets)
            x, mark, _ = corrupt(data, origins, selected, ids, 96, context)
            evaluated = np.arange(n_targets)[None, :] == selected[:, None]
        else:
            raise ValueError("Unsupported scenario " + scenario)
        pred = np.asarray(predict(x, mark, origins), dtype=np.float64)[:, :, ids]
        if pred.shape != (len(origins), h, n_targets) or not np.isfinite(pred).all():
            raise FloatingPointError("Invalid prediction in " + scenario)
        pos = origins[:, None] + np.arange(h)[None, :]
        obs = observed[pos][:, :, ids]
        y_raw = raw[pos][:, :, ids].astype(np.float64)
        n_obs = obs.sum(1)
        t_mean = np.where(n_obs > 0, np.where(obs, y_raw, 0.0).sum(1) / np.maximum(n_obs, 1), np.nan)
        pred_mean.append((pred * scale + mean).mean(1)[:, keep])
        true_mean.append(t_mean[:, keep])
        hours.append(n_obs[:, keep].astype(np.int16))
        scored.append((evaluated & (n_obs >= MIN_HOURS[dataset]))[:, keep])
    return {"pred_daily_mean": np.concatenate(pred_mean).astype(np.float32),
            "true_daily_mean": np.concatenate(true_mean).astype(np.float32),
            "observed_hours": np.concatenate(hours), "scored": np.concatenate(scored), "origin_index": origins_all,
            "target_names": np.array([meta["targets"][k] for k in keep]),
            "variables": np.array([variables[k] for k in keep])}


def f1_scores(arrays, dataset, variable):
    """(precision, recall, F1, positives) of exceedance of the 24-hour-mean threshold for one variable."""
    use = (arrays["variables"] == variable)[None, :] & arrays["scored"]
    threshold = THRESHOLDS[dataset][variable]
    truth = arrays["true_daily_mean"][use] >= threshold
    pred = arrays["pred_daily_mean"][use] >= threshold
    tp, fp, fn = int((truth & pred).sum()), int((~truth & pred).sum()), int((truth & ~pred).sum())
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return precision, recall, 2 * precision * recall / max(precision + recall, 1e-12), int(truth.sum())
