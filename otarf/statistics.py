"""Paired temporal block estimates for frozen, already fitted models."""

import numpy as np


def macro(numerator, count):
    a, n = np.asarray(numerator).sum(0), np.asarray(count).sum(0)
    use = n > 0
    return float((a[use] / n[use]).mean()) if use.any() else float("nan")


def paired_intervals(first, second, counts, length, reps, rng, family):
    active = counts.sum(0) > 0
    first, second, counts = first[:, active], second[:, active], counts[:, active]
    n, c = counts.shape
    length = min(length, n)
    whole, remainder = divmod(n, length)
    values = np.concatenate([first, second, counts], axis=1)
    prefix = np.vstack([np.zeros((1, 3 * c)), values.cumsum(0)])
    blocks = prefix[length:] - prefix[:-length]
    residual = prefix[remainder:] - prefix[:-remainder] if remainder else None
    samples = []
    invalid = 0
    for begin in range(0, reps, 100):
        size = min(100, reps - begin)
        draw = blocks[rng.integers(len(blocks), size=(size, whole))].sum(1)
        if remainder:
            draw += residual[rng.integers(len(residual), size=size)]
        den = draw[:, 2 * c :]
        valid = (den > 0).all(1)
        invalid += int((~valid).sum())
        draw, den = draw[valid], den[valid]
        samples.append((draw[:, :c] / den).mean(1) - (draw[:, c : 2 * c] / den).mean(1))
    samples = np.concatenate(samples)
    if len(samples) < reps / 2:
        return {
            "ci95_low": None,
            "ci95_high": None,
            "family_low": None,
            "family_high": None,
            "invalid_fraction": invalid / reps,
            "reason": "Insufficient target coverage in temporal resamples",
        }
    limits = np.quantile(
        samples, [0.025, 0.975, 0.05 / (2 * family), 1 - 0.05 / (2 * family)]
    )
    return {
        "ci95_low": float(limits[0]),
        "ci95_high": float(limits[1]),
        "family_low": float(limits[2]),
        "family_high": float(limits[3]),
        "invalid_fraction": invalid / reps,
        "reason": "Coverage-conditioned percentile bootstrap; fitted-seed uncertainty reported separately",
    }
