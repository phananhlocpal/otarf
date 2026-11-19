#!/usr/bin/env python3
"""Build the three hourly benchmarks used in the paper (Beijing, London, Shanghai).

Splits are chronological 70/10/20 %. Channels are kept if train coverage >= 20 % and train
variance > 0; missing values are forward filled (initial gaps: train median); the original
values and the observation mask are stored separately in observations.npz.

Expected raw files (see scripts/download_data.sh):
  data/raw/beijing_multisite.zip
  data/raw/kdd2018.zip
  data/raw/ShanghaiPM20100101_20151231.csv   (from the UCI "PM2.5 Data of Five Chinese Cities" archive)
"""
import argparse
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from otarf.io import sha256, write_json

RAW = ROOT / "data/raw"
OUT = ROOT / "data/processed"
POLLUTANTS = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
SOURCES = {
    "beijing_multisite": {
        "archive": "beijing_multisite.zip",
        "url": "https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip",
        "citation": "https://doi.org/10.24432/C5RK5G",
        "license": "CC BY 4.0",
    },
    "kdd2018_london": {
        "archive": "kdd2018.zip",
        "url": "https://zenodo.org/records/4656719/files/kdd_cup_2018_dataset_with_missing_values.zip?download=1",
        "citation": "https://doi.org/10.5281/zenodo.4656719",
        "license": "See deposited record and original KDD Cup terms",
    },
}
SHANGHAI_CSV = "ShanghaiPM20100101_20151231.csv"
SHANGHAI_SHA256 = "49258afcca08ca1a5c76379c31a439cc0179f53371842f14fd6ee6127ebffc14"
SH_TARGETS = ["PM_Jingan", "PM_US Post", "PM_Xuhui"]
SH_NUMERIC = ["DEWP", "HUMI", "PRES", "TEMP", "Iws", "precipitation", "Iprec"]
SH_WIND = ["NE", "NW", "SE", "SW", "cv"]


def hourly(frame):
    frame = frame.sort_index()
    if frame.index.has_duplicates:
        raise ValueError("Duplicate hours: an explicit dataset policy is required")
    return frame.reindex(pd.date_range(frame.index.min(), frame.index.max(), freq="h"))


def beijing():
    with zipfile.ZipFile(RAW / "beijing_multisite.zip") as outer:
        inner = zipfile.ZipFile(
            io.BytesIO(outer.read("PRSA2017_Data_20130301-20170228.zip"))
        )
    frames, targets, groups = [], [], {}
    compass = [
        "N",
        "NNE",
        "NE",
        "ENE",
        "E",
        "ESE",
        "SE",
        "SSE",
        "S",
        "SSW",
        "SW",
        "WSW",
        "W",
        "WNW",
        "NW",
        "NNW",
    ]
    angles = {name: i * np.pi / 8 for i, name in enumerate(compass)}
    stations = []
    for name in sorted(inner.namelist()):
        if not name.endswith(".csv"):
            continue
        raw = pd.read_csv(inner.open(name))
        station = str(raw.station.iloc[0])
        stations.append(station)
        frame = raw[POLLUTANTS + ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"]].copy()
        angle = raw.wd.map(angles)
        frame["wind_sin"], frame["wind_cos"] = np.sin(angle), np.cos(angle)
        frame.index = pd.to_datetime(raw[["year", "month", "day", "hour"]])
        frame.columns = [station + "__" + c for c in frame]
        targets += [station + "__" + c for c in POLLUTANTS]
        groups.update({station + "__" + c: c for c in POLLUTANTS})
        frames.append(hourly(frame))
    if len(frames) != 12:
        raise ValueError(f"Expected 12 stations, found {len(frames)}")
    return (
        pd.concat(frames, axis=1),
        targets,
        groups,
        {"station_count": 12, "stations": stations},
    )


def kdd():
    # Parse the original Monash TSF attributes instead of assuming a fixed field order.
    attributes, series = [], []
    with zipfile.ZipFile(RAW / "kdd2018.zip") as archive:
        names = [n for n in archive.namelist() if n.endswith(".tsf")]
        if len(names) != 1:
            raise ValueError("Expected one TSF file")
        for line in archive.read(names[0]).decode("cp1252").splitlines():
            line = line.strip()
            if line.startswith("@attribute"):
                attributes.append(line.split()[1])
            elif line and not line.startswith(("@", "#")):
                fields = line.split(":")
                if len(fields) != len(attributes) + 1:
                    raise ValueError("Unexpected TSF field count")
                record = dict(zip(attributes, fields[:-1]))
                # London only: all city stations, no cherry picking by model accuracy.
                city = record.get("city", record.get("city_name", ""))
                if city.lower() not in ("london", "ld"):
                    continue
                values = np.array(
                    [float(x) if x != "?" else np.nan for x in fields[-1].split(",")]
                )
                start = pd.to_datetime(
                    record["start_timestamp"], format="%Y-%m-%d %H-%M-%S"
                )
                pollutant = record.get(
                    "air_quality_measurement",
                    record.get("air_quality", record.get("measurement", "")),
                )
                station = record.get(
                    "station", record.get("station_name", record.get("series_name", ""))
                )
                if not pollutant:
                    raise ValueError(f"Unknown pollutant field: {attributes}")
                name = station + "__" + pollutant
                series.append(
                    (
                        name,
                        pollutant,
                        pd.Series(
                            values,
                            index=pd.date_range(start, periods=len(values), freq="h"),
                        ),
                    )
                )
    if not series:
        raise ValueError(f"No London series; TSF attributes={attributes}")
    frame = pd.concat([s.rename(n) for n, _, s in series], axis=1)
    if frame.columns.has_duplicates:
        raise ValueError("Duplicate KDD channel names")
    frame = hourly(frame)
    return (
        frame,
        list(frame.columns),
        {n: p for n, p, _ in series},
        {
            "station_count": len({n.split("__")[0] for n, _, _ in series}),
            "selection": "all London stations and pollutants in original deposit",
        },
    )


def save(name):
    frame, targets, groups, extra = {
        "beijing_multisite": beijing,
        "kdd2018_london": kdd,
    }[name]()
    n = len(frame)
    train_end, test_start = int(n * 0.7), n - int(n * 0.2)
    observed = frame.notna()
    # Selection uses training only, including genuinely unusable/constant channels.
    coverage = observed.iloc[:train_end].mean()
    spread = frame.iloc[:train_end].std()
    keep = list(frame.columns[(coverage >= 0.2) & (spread > 1e-8)])
    dropped = [str(c) for c in frame if c not in keep]
    frame, observed = frame[keep], observed[keep]
    targets = [c for c in targets if c in keep]
    if not targets:
        raise ValueError(f"No measurable pollutants in {name}")
    # Official loaders put --target last even for M; make this explicit in all artifacts.
    final_target = targets[0]
    columns = [c for c in keep if c != final_target] + [final_target]
    frame, observed = frame[columns], observed[columns]
    medians = frame.iloc[:train_end].median()
    filled = frame.ffill().fillna(medians)
    if not np.isfinite(filled.to_numpy()).all():
        raise ValueError("Non-finite value after causal imputation")
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    csv = out / "data.csv"
    filled.rename_axis("date").reset_index().to_csv(
        csv, index=False, float_format="%.8g"
    )
    np.savez_compressed(
        out / "observations.npz",
        raw=frame.to_numpy(np.float32),
        observed=observed.to_numpy(bool),
        dates=frame.index.to_numpy(dtype="datetime64[ns]").astype(np.int64),
    )
    # Count actual labels by split; do not use these counts to select model hyperparameters.
    counts = {
        split: {c: int(observed[c].iloc[a:b].sum()) for c in targets}
        for split, a, b in [
            ("train", 0, train_end),
            ("val", train_end, test_start),
            ("test", test_start, n),
        ]
    }
    unevaluable = {
        s: [c for c in targets if counts[s][c] == 0] for s in ("val", "test")
    }
    if any(sum(counts[s].values()) == 0 for s in ("val", "test")):
        raise ValueError("No observed pollutant labels in an entire evaluation split")
    source = dict(SOURCES[name])
    archive = RAW / source["archive"]
    source.update(sha256=sha256(archive), bytes=archive.stat().st_size)
    meta = {
        "name": name,
        "rows": n,
        "channels": len(columns),
        "columns": columns,
        "targets": targets,
        "target_groups": {c: groups[c] for c in targets},
        "official_target": final_target,
        "train_end": train_end,
        "test_start": test_start,
        "first_hour": str(frame.index[0]),
        "last_hour": str(frame.index[-1]),
        "dropped_training_only": dropped,
        "missing_fraction": float(1 - observed.to_numpy().mean()),
        "observed_target_counts": counts,
        "targets_without_observations": unevaluable,
        "imputation": "past-only forward fill; initial gaps use training medians; no backfill/interpolation",
        "train_medians": medians.to_dict(),
        "csv_sha256": sha256(csv),
        "observations_sha256": sha256(out / "observations.npz"),
        "source": source,
        **extra,
    }
    write_json(out / "metadata.json", meta)
    print(
        f"{name}: {n} hours x {len(columns)} channels; {len(targets)} pollutant targets; missing={meta['missing_fraction']:.2%}",
        flush=True,
    )
    return meta


def shanghai():
    """Shanghai city file of UCI 394: three PM2.5 sites, seven meteorological series, five wind categories."""
    path = RAW / SHANGHAI_CSV
    if sha256(path) != SHANGHAI_SHA256:
        raise ValueError("Unexpected Shanghai source file (SHA-256 differs)")
    raw = pd.read_csv(path)
    rows = len(raw)
    train = int(rows * 0.7)
    dates = pd.to_datetime(raw[["year", "month", "day", "hour"]], errors="raise")
    if not dates.equals(pd.Series(pd.date_range("2010-01-01", periods=rows, freq="h"))):
        raise ValueError("Timestamps are not a complete hourly grid")
    if set(raw.cbwd.dropna().unique()) - set(SH_WIND):
        raise ValueError("Unknown wind direction")
    frame = raw[SH_TARGETS + SH_NUMERIC].apply(pd.to_numeric, errors="raise").copy()
    for target in SH_TARGETS:
        frame.loc[frame[target] < 0, target] = np.nan
    for category in SH_WIND:
        frame["wind_" + category] = np.where(
            raw.cbwd.notna(), (raw.cbwd == category).astype(float), np.nan
        )
    frame.index = pd.DatetimeIndex(dates)
    observed = frame.notna()
    final_target = SH_TARGETS[0]
    columns = [c for c in frame.columns if c != final_target] + [final_target]
    frame, observed = frame[columns], observed[columns]
    medians = frame.iloc[:train].median()
    filled = frame.ffill().fillna(medians)
    test_start = rows - int(0.2 * rows)
    out = OUT / "shanghai_confirmation"
    out.mkdir(parents=True, exist_ok=True)
    filled.rename_axis("date").reset_index().to_csv(
        out / "data.csv", index=False, float_format="%.8g"
    )
    np.savez_compressed(
        out / "observations.npz",
        raw=frame.to_numpy(np.float32),
        observed=observed.to_numpy(bool),
        dates=frame.index.to_numpy(dtype="datetime64[ns]").astype(np.int64),
    )
    counts = {
        s: {c: int(observed[c].iloc[a:b].sum()) for c in SH_TARGETS}
        for s, a, b in [
            ("train", 0, train),
            ("val", train, test_start),
            ("test", test_start, rows),
        ]
    }
    meta = {
        "name": "shanghai_confirmation",
        "rows": rows,
        "channels": len(columns),
        "columns": columns,
        "targets": SH_TARGETS,
        "target_groups": {c: "PM2.5" for c in SH_TARGETS},
        "official_target": final_target,
        "train_end": train,
        "test_start": test_start,
        "first_hour": str(frame.index[0]),
        "last_hour": str(frame.index[-1]),
        "station_count": 3,
        "train_medians": medians.to_dict(),
        "missing_fraction": float(1 - observed.to_numpy().mean()),
        "observed_target_counts": counts,
        "targets_without_observations": {
            s: [c for c in SH_TARGETS if not counts[s][c]] for s in ("val", "test")
        },
        "imputation": "past-only forward fill; initial gaps use train medians",
        "csv_sha256": sha256(out / "data.csv"),
        "observations_sha256": sha256(out / "observations.npz"),
        "source": {
            "dataset": "UCI 394 PM2.5 Data of Five Chinese Cities, Shanghai",
            "doi": "10.24432/C52K58",
            "license": "CC BY 4.0",
            "sha256": SHANGHAI_SHA256,
        },
    }
    write_json(out / "metadata.json", meta)
    print(
        f"shanghai_confirmation: {rows} hours x {len(columns)} channels; {len(SH_TARGETS)} targets; missing={meta['missing_fraction']:.2%}",
        flush=True,
    )
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["beijing_multisite", "kdd2018_london", "shanghai_confirmation"],
        default=["beijing_multisite", "kdd2018_london", "shanghai_confirmation"],
    )
    parser.add_argument("--raw-root", type=Path, default=RAW)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    RAW, OUT = args.raw_root, args.out
    for name in args.datasets:
        shanghai() if name == "shanghai_confirmation" else save(name)
