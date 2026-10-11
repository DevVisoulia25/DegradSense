"""Engine-level uncertainty summaries; never bootstrap overlapping windows as IID."""

import math

import numpy as np


def conformal_radius(actual, predicted, alpha=0.1):
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError("Calibration targets and predictions must have matching shapes")
    scores = np.abs(actual - predicted)
    if scores.ndim != 1 or len(scores) == 0 or not np.isfinite(scores).all() or not 0 < alpha < 1:
        raise ValueError("Require finite one-dimensional calibration scores and 0 < alpha < 1")
    rank = math.ceil((len(scores) + 1) * (1 - alpha))
    if rank > len(scores):
        raise ValueError("Too few calibration engines for the requested coverage")
    return float(np.sort(scores)[rank - 1])


def interval_bounds(predicted, radius, cap):
    if radius < 0 or not np.isfinite(radius) or cap <= 0:
        raise ValueError("Invalid interval radius or target cap")
    predicted = np.asarray(predicted, dtype=float)
    return np.clip(predicted - radius, 0, cap), np.clip(predicted + radius, 0, cap)


def bootstrap_rmse(actual, predicted, seed=2026, repetitions=2000):
    """Input must have one observation per independent engine."""
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError("Bootstrap targets and predictions must have matching shapes")
    errors = predicted - actual
    if errors.ndim != 1 or len(errors) < 2 or not np.isfinite(errors).all():
        raise ValueError("Require at least two finite engine-level errors")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(errors), size=(repetitions, len(errors)))
    distribution = np.sqrt(np.mean(errors[indices] ** 2, axis=1))
    low, high = np.quantile(distribution, [0.025, 0.975])
    return {
        "rmse_ci_low": float(low),
        "rmse_ci_high": float(high),
        "bootstrap_repetitions": repetitions,
        "bootstrap_seed": seed,
    }
