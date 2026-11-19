"""Outages of several channels at once, hourly threshold events, and parameter-free reference forecasts.

Scenarios (all at the end of the 96-hour history, test split):
  natural            no outage
  single{6,24,96}    one target per origin (otarf.masking; the paper's controlled outage)
  rand{25,50}_g{24,96}     25% or 50% of the targets, chosen per origin by a SHA-256 rule
  station{25,50}_g{24,96}  every channel (pollutant and meteorology) of 25% or 50% of the stations
                           (datasets with station-prefixed channel names only)
Only labels of the removed targets are scored. Every model sees the same removed channels.
"""

import hashlib
import math

import numpy as np

from .masking import corrupt as corrupt_single
from .masking import selected_targets

# Hourly-value thresholds (ug/m3) used for the event analysis. Beijing and Shanghai: grade II 24-hour limit of
# the Chinese national standard GB 3095-2012 (PM2.5 75, PM10 150). London: first value of the 'Moderate' and
# 'High' bands of the UK Daily Air Quality Index (PM2.5 36 and 54, PM10 51 and 76). Applied to hourly values,
# so these are exceedance events, not the official AQI.
THRESHOLDS = {
    "beijing_multisite": {"PM2.5": [75], "PM10": [150]},
    "shanghai_confirmation": {"PM2.5": [75]},
    "kdd2018_london": {"PM2.5": [36, 54], "PM10": [51, 76]},
}


def parse(name):
    """channel name -> (station or None, variable)."""
    if "__" in name:
        station, var = name.split("__", 1)
    elif name.startswith("PM_"):
        station, var = name[3:], "PM2.5"
    else:
        station, var = None, name
    return station, var


def _rng(dataset, origin, kind):
    digest = hashlib.sha256(f"revision-multi-v1|{dataset}|{int(origin)}|{kind}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def multi_selection(dataset, origins, kind, pct, columns, target_names, ids):
    """Per origin: the channels to remove, and a boolean [n_origins, n_targets] of removed targets."""
    n_t = len(target_names)
    eval_mask = np.zeros((len(origins), n_t), bool)
    chans = []
    if kind == "rand":
        k = math.ceil(pct / 100 * n_t)
        for i, o in enumerate(origins):
            chosen = _rng(dataset, o, "rand").permutation(n_t)[:k]
            eval_mask[i, chosen] = True
            chans.append(ids[chosen])
        return chans, eval_mask
    tgt_station = [parse(t)[0] for t in target_names]
    col_station = [parse(c)[0] for c in columns]
    stations = sorted({s for s in tgt_station if s is not None})
    if not stations:
        raise ValueError("No station structure in this dataset")
    k = math.ceil(pct / 100 * len(stations))
    for i, o in enumerate(origins):
        chosen = {stations[j] for j in _rng(dataset, o, "station").permutation(len(stations))[:k]}
        chans.append(np.array([c for c, s in enumerate(col_station) if s in chosen], dtype=np.int64))
        eval_mask[i] = [s in chosen for s in tgt_station]
    return chans, eval_mask


def corrupt_multi(data, origins, chan_lists, gap, context=96):
    """Same construction as otarf.masking.corrupt, applied to every removed channel of every origin."""
    c = data.meta["channels"]
    idx = origins[:, None] - context + np.arange(context)[None, :]
    x = data.values[idx].copy()
    marks = data.marks[idx].copy()
    rows = np.concatenate([np.full(len(ch), i, dtype=np.int64) for i, ch in enumerate(chan_lists)])
    chs = np.concatenate(chan_lists).astype(np.int64)
    prev = origins[rows] - gap - 1
    if np.any(prev < 0) or gap not in (24, 96):
        raise ValueError("Unsupported gap or insufficient past")
    t = np.arange(context - gap, context)[None, :]
    x[rows[:, None], t, chs[:, None]] = data.values[prev, chs][:, None]
    marks[rows[:, None], t, chs[:, None]] = 0
    marks[rows[:, None], t, (c + chs)[:, None]] = data.age[prev, chs][:, None] + np.arange(1, gap + 1)[None, :]
    for offset in (2 * c, 3 * c):
        marks[rows[:, None], t, (offset + chs)[:, None]] = data.marks[prev, offset + chs][:, None]
    return x, marks


def scenarios_for(dataset, include_single6=False):
    names = (["natural", "single6"] if include_single6 else ["natural"]) + ["single24", "single96"]
    names += [f"rand{p}_g{g}" for p in (25, 50) for g in (24, 96)]
    if dataset != "shanghai_confirmation":
        names += [f"station{p}_g{g}" for p in (25, 50) for g in (24, 96)]
    return names


def events_update(store, dataset, target_names, pred_raw, y_raw, valid):
    """Accumulate [TP, FP, FN, TN] per (variable, threshold)."""
    var = [parse(n)[1] for n in target_names]
    for v, levels in THRESHOLDS[dataset].items():
        pos = np.array([j for j, x in enumerate(var) if x == v], dtype=np.int64)
        if not len(pos):
            continue
        p, y, m = pred_raw[:, :, pos], y_raw[:, :, pos], valid[:, :, pos]
        for level in levels:
            s = store.setdefault(f"{v}_{level}", [0, 0, 0, 0])
            yp, pp = y >= level, p >= level
            s[0] += int((m & yp & pp).sum())
            s[1] += int((m & ~yp & pp).sum())
            s[2] += int((m & yp & ~pp).sum())
            s[3] += int((m & ~yp & ~pp).sum())


def run_scenario(predict, data, dataset, horizon, scenario, batch_size, context=96):
    """predict(x, mark, origins) -> array [B, horizon, channels] in train-standardized units.
    Returns ({'count', 'abs_z'} per origin and target, events) for the test origins."""
    meta = data.meta
    ids = np.array([meta["columns"].index(t) for t in meta["targets"]])
    n_t = len(ids)
    with np.load(data.directory / "observations.npz") as f:
        raw, observed = f["raw"], f["observed"]
    mean, scale = data.scaler.mean_[ids], data.scaler.scale_[ids]
    origins_all = np.arange(meta["test_start"], meta["rows"] - horizon + 1)
    kind = scenario.split("_")[0]
    if kind.startswith(("rand", "station")):
        pct = int("".join(ch for ch in kind if ch.isdigit()))
        gap = int(scenario.split("_g")[1])
        chans, eval_all = multi_selection(
            dataset, origins_all, "rand" if kind.startswith("rand") else "station", pct, meta["columns"], meta["targets"], ids
        )
    counts, abs_z, events = [], [], {}
    for start in range(0, len(origins_all), batch_size):
        origins = origins_all[start : start + batch_size]
        idx = origins[:, None] - context + np.arange(context)[None, :]
        if scenario == "natural":
            x, mark, ev = data.values[idx].copy(), data.marks[idx].copy(), np.ones((len(origins), n_t), bool)
        elif scenario.startswith("single"):
            selected = selected_targets(dataset, origins, n_t)
            x, mark, _ = corrupt_single(data, origins, selected, ids, int(scenario[6:]), context)
            ev = np.arange(n_t)[None, :] == selected[:, None]
        else:
            sl = slice(start, start + len(origins))
            x, mark = corrupt_multi(data, origins, chans[sl], gap, context)
            ev = eval_all[sl]
        pred = np.asarray(predict(x, mark, origins), dtype=np.float64)[:, :, ids]
        if pred.shape != (len(origins), horizon, n_t) or not np.isfinite(pred).all():
            raise ValueError("Invalid predictions in " + scenario)
        pos = origins[:, None] + np.arange(horizon)[None, :]
        valid = observed[pos][:, :, ids] & ev[:, None, :]
        y_raw = raw[pos][:, :, ids].astype(np.float64)
        err = np.where(valid, pred - (y_raw - mean) / scale, 0.0)
        counts.append(valid.sum(1).astype(np.int16))
        abs_z.append(np.abs(err).sum(1).astype(np.float32))
        events_update(events, dataset, meta["targets"], pred * scale + mean, y_raw, valid)
    return {"count": np.concatenate(counts), "abs_z": np.concatenate(abs_z)}, events


def naive_predictor(kind, horizon):
    """Parameter-free references on the filled series: 'persistence' repeats the last value; 'seasonal'
    repeats the value 24 h earlier (48 h earlier for steps 25-48)."""
    if kind == "persistence":
        return lambda x, mark, origins: np.repeat(x[:, -1:, :], horizon, axis=1)
    if kind == "seasonal":
        index = [95 + k - 24 * math.ceil(k / 24) for k in range(1, horizon + 1)]
        return lambda x, mark, origins: x[:, index, :]
    raise ValueError("Unknown naive kind " + kind)
