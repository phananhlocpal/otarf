#!/usr/bin/env python3
"""Draw the result figures of the paper (run scripts/summarize.py first).

masking_curves   sMAE versus hours of masked target history (OTARF, retrained baselines, last-value reference)
gap_lengths      share of missing hours that lie in outages of at least L hours (needs data/processed)
"""
import argparse
import csv
from pathlib import Path

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CITIES = ["Beijing", "London", "Shanghai"]
COLORS = {"OTARF": "#2a78d6", "SOFTS": "#eb6834", "TimeFilter": "#1baf7a"}
MARKERS = {"OTARF": "o", "SOFTS": "s", "TimeFilter": "^"}
CITY_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, MUTED = "#0b0b0b", "#52514e"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    }
)


def rows(path):
    return list(csv.DictReader(open(path)))


def masking_curves(results, out):
    """Figure: test sMAE at H = 24 versus hours of masked target history, with the last-value reference."""
    data = [r for r in rows(results / "table_masked.csv") if r["horizon"] == "24"]
    natural = {(r["city"], r["model"]): float(r["natural_mean"]) for r in rows(results / "table_natural.csv") if r["horizon"] == "24"}
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    for ax, city in zip(axes, CITIES):
        for m in ("OTARF", "SOFTS", "TimeFilter"):
            r = next(r for r in data if r["city"] == city and r["model"] == m)
            ys = [natural[(city, m)]] + [float(r[f"gap{g}_mean"]) for g in (6, 24, 96)]
            ax.plot(range(4), ys, color=COLORS[m], marker=MARKERS[m], lw=1.4, ms=4, label=m)
        ref = [next(float(r[f"gap{g}_mean"]) for r in data if r["city"] == city and r["model"].startswith("last") and r.get(f"gap{g}_mean")) for g in (6, 24, 96)]
        ax.plot(range(1, 4), ref, ls=":", lw=1.2, color="#9a9893", label="Last value (reference)")
        ax.set_title(city, fontsize=9, color=INK)
        ax.set_xticks(range(4), ["0", "6", "24", "96"])
        ax.set_xlabel("Most recent hours masked")
        ax.grid(axis="y", color="#e6e5e1", lw=0.6)
    axes[0].set_ylabel("Test sMAE, H = 24")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.tight_layout(w_pad=1.2, rect=(0, 0.1, 1, 1))
    fig.savefig(out / "masking_curves.pdf")
    fig.savefig(out / "masking_curves.png", dpi=200)


def gap_lengths(data_root, out):
    """Share of the missing hours of the target series that lie in outages of at least L hours."""
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    lengths = np.unique(np.round(np.logspace(0, 3.4, 120)).astype(int))
    names = {
        "beijing_multisite": "Beijing",
        "kdd2018_london": "London",
        "shanghai_confirmation": "Shanghai",
    }
    for (dataset, name), color in zip(names.items(), CITY_COLORS):
        meta = json.load(open(data_root / dataset / "metadata.json"))
        with np.load(data_root / dataset / "observations.npz") as z:
            observed = z["observed"]
        runs = []
        for target in meta["targets"]:
            o = observed[:, meta["columns"].index(target)]
            if not o.any():
                continue
            first, last = int(np.argmax(o)), len(o) - 1 - int(np.argmax(o[::-1]))
            gap = ~o[
                first : last + 1
            ]  # between the first and the last valid observation
            d = np.diff(np.concatenate([[0], gap.astype(int), [0]]))
            runs += list(np.where(d == -1)[0] - np.where(d == 1)[0])
        runs = np.array(runs)
        ax.plot(
            lengths,
            [runs[runs >= L].sum() / runs.sum() for L in lengths],
            color=color,
            lw=1.8,
            label=name,
        )
    for L in (6, 24, 96):
        ax.axvline(L, color=MUTED, lw=0.7, ls=":")
        ax.text(L * 1.06, 1.02, f"{L} h", fontsize=7, color=MUTED, va="bottom")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.0)
    ax.set_xlim(1, 2500)
    ax.set_xlabel("Length L of a continuous outage (hours)")
    ax.set_ylabel("Share of missing hours in\noutages of length >= L")
    ax.grid(color="#e6e5e1", lw=0.6)
    ax.legend(
        frameon=False, fontsize=7.5, loc="upper right", bbox_to_anchor=(1.0, 0.92)
    )
    fig.tight_layout()
    fig.savefig(out / "gap_lengths.pdf")
    fig.savefig(out / "gap_lengths.png", dpi=200)


def case_study(runs, out):
    """Figure 4: the 96-hour case study (needs runs/inspect/<dataset>/case_96h.npz)."""
    names = {"beijing_multisite": "Beijing", "kdd2018_london": "London", "shanghai_confirmation": "Shanghai"}
    styles = {"OTARF": dict(color=COLORS["OTARF"], ls="-", lw=1.6), "TimeFilter": dict(color=COLORS["TimeFilter"], ls="-", lw=1.6),
              "SOFTS": dict(color=COLORS["SOFTS"], ls=(0, (3, 2)), lw=1.6)}
    cases = {c["dataset"]: c for c in json.load(open(ROOT / "configs/case_selection.json"))}
    if not all((runs / "inspect" / d / "case_96h.npz").exists() for d in names):
        return
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))
    for ax, d in zip(axes, names):
        z = dict(np.load(runs / "inspect" / d / "case_96h.npz"))
        t_hist = np.arange(-len(z["history_raw"]), 0); t_fut = np.arange(len(z["future_raw"]))
        keep, hidden, obs = t_hist >= -144, t_hist >= -96, z["history_observed"].astype(bool)
        ax.axvspan(-96, 0, color="#ececea", lw=0)
        ax.plot(t_hist[keep & obs & ~hidden], z["history_raw"][keep & obs & ~hidden], ".", ms=2.5, color="#52514e", label="Observed, seen by the model")
        ax.plot(t_hist[obs & hidden], z["history_raw"][obs & hidden], ".", ms=2.5, color="#b4b2ad", label="Observed, masked")
        ax.plot(np.arange(-96, 0), z["input_raw"], "-", lw=1, color="#52514e", label="Filled input")
        fm = z["future_observed"].astype(bool)
        ax.plot(t_fut[fm], z["future_raw"][fm], "k.", ms=3.5, label="Future observations")
        for m in ("OTARF", "TimeFilter", "SOFTS"):
            ax.plot(t_fut, z["pred_" + m], label=m, **styles[m])
        ax.set_title(f"{names[d]}: {cases[d]['target'].replace('__', ', ').replace('PM_', 'PM2.5, ')}", fontsize=8.5)
        ax.set_xlim(-146, 26); ax.set_xticks([-144, -96, -48, 0, 24]); ax.axvline(0, color="#9a9893", lw=0.6)
    axes[0].set_ylabel("Concentration")
    axes[1].set_xlabel("Hours relative to the forecast origin")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(out / "case_96h.pdf")
    fig.savefig(out / "case_96h.png", dpi=200)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=ROOT / "results")
    p.add_argument("--data-root", type=Path, default=ROOT / "data/processed")
    p.add_argument("--runs", type=Path, default=ROOT / "runs")
    a = p.parse_args()
    out = a.results / "figures"
    out.mkdir(parents=True, exist_ok=True)
    masking_curves(a.results, out)
    case_study(a.runs, out)
    if (a.data_root / "beijing_multisite" / "observations.npz").exists():
        gap_lengths(a.data_root, out)
    print("Figures written to", out)
