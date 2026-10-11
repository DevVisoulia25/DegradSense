"""RUL metrics and explicit warning definitions on complete held-out lifetimes."""

import numpy as np
import pandas as pd


def regression_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if actual.ndim != 1 or actual.shape != predicted.shape:
        raise ValueError("Targets and predictions must be matching one-dimensional arrays")
    error = predicted - actual
    if error.size == 0 or not np.isfinite(error).all():
        raise ValueError("Metrics require finite, nonempty predictions and targets")
    # Positive error overestimates available life and receives a larger penalty.
    score = np.expm1(np.where(error < 0, -error / 13.0, error / 10.0))
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "nasa_score_sum": float(score.sum()),
        "nasa_score_mean": float(score.mean()),
        "n_predictions": int(error.size),
    }


def warning_metrics(frame, alerts, horizon=30, persistence=3):
    """A warning is actionable after persistence consecutive cycle-level alerts.

    False-positive cycle rate: fraction of observed cycles with raw RUL > horizon
    that carry an alert. A missed engine has no persistent warning inside the
    final horizon cycles. Lead time is raw RUL at its first confirmed warning
    inside that horizon; this avoids rewarding premature alerts as early success.
    """
    if persistence < 1 or horizon < 1:
        raise ValueError("Horizon and persistence must be positive")
    working = frame.copy()
    working["alert"] = np.asarray(alerts, dtype=bool)
    leads, missed, records = [], 0, []
    for engine, group in working.groupby("engine", sort=True):
        group = group.sort_values("cycle").copy()
        if (np.diff(group.cycle) != 1).any():
            raise ValueError("Warning replay requires consecutive cycles (stride=1)")
        confirmed = group.alert.rolling(persistence, min_periods=persistence).sum() == persistence
        eligible = group[confirmed & (group.rul_raw <= horizon)]
        if eligible.empty:
            missed += 1
            lead = None
        else:
            lead = float(eligible.iloc[0].rul_raw)
            leads.append(lead)
        records.append({"engine": int(engine), "lead_cycles": lead, "missed": eligible.empty})
    outside = working.rul_raw > horizon
    summary = {
        "horizon_cycles": horizon,
        "persistence_cycles": persistence,
        "n_engines": len(records),
        "missed_engines": missed,
        "missed_engine_rate": missed / len(records),
        "false_positive_cycle_rate": float(working.loc[outside, "alert"].mean()) if outside.any() else None,
        "mean_confirmed_lead_cycles": float(np.mean(leads)) if leads else None,
        "median_confirmed_lead_cycles": float(np.median(leads)) if leads else None,
    }
    return summary, pd.DataFrame(records)
