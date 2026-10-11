"""Prespecified development, calibration, assessment and official-test study."""

import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from rul.data import (
    ROOT,
    Preprocessor,
    label_test,
    label_training,
    load_trajectories,
    make_windows,
    split_engines,
    window_features,
)
from rul.experiment import prediction_frame, validation_endpoints, write_json
from rul.metrics import regression_metrics, warning_metrics
from rul.models import classical_model, fit_sequence, sequence_predict
from rul.statistics import bootstrap_rmse, conformal_radius, interval_bounds

MODELS = ["mean", "rf", "xgb", "lstm", "gru", "rf_latest", "rf_single"]


def cohorts(ids, seed=2026):
    """Reserve calibration and assessment before any development model fitting."""
    development, reserved = train_test_split(sorted(ids), test_size=0.4, random_state=seed)
    calibration, assessment = train_test_split(reserved, test_size=0.5, random_state=seed + 1)
    return {
        "development": sorted(development),
        "calibration": sorted(calibration),
        "assessment": sorted(assessment),
    }


def inputs(x, name, sensor_index):
    if name == "rf_latest":
        return x[:, -1]
    if name == "rf_single":
        return window_features(x[:, :, [sensor_index]])
    return window_features(x)


def select_single(train, sensors):
    # Chosen with training-engine data only; selection is repeated per split.
    return sensors.index(train[sensors].corrwith(train.rul_raw).abs().idxmax())


def fit_candidate(name, training, validation, settings, seed, single_index):
    started = time.perf_counter()
    history = []
    if name in ("lstm", "gru"):
        model, history = fit_sequence(name, training, validation, settings["cap"], settings["epochs"], seed)
    else:
        model = classical_model("rf" if name.startswith("rf_") else name, seed, settings["trees"])
        model.fit(inputs(training.x, name, single_index), training.y)
    return model, history, time.perf_counter() - started


def predict_candidate(name, model, windows, settings, single_index):
    if name in ("lstm", "gru"):
        return sequence_predict(model, windows.x, settings["cap"])
    return np.clip(model.predict(inputs(windows.x, name, single_index)), 0, settings["cap"])


def development_run(data, subset, seed, output, settings):
    if (output / "complete.json").exists():
        return pd.read_csv(output / "metrics.csv").to_dict("records")
    output.mkdir(parents=True, exist_ok=True)
    train, val = split_engines(data, 0.25, seed)
    prep = Preprocessor.fit(train)
    single = select_single(train, prep.sensors)
    training = make_windows(prep.transform(train), prep.sensors, settings["window"], settings["stride"])
    endpoint = make_windows(
        prep.transform(validation_endpoints(val, settings["window"], 31415)),
        prep.sensors,
        settings["window"],
        last_only=True,
    )
    write_json(
        output / "split.json",
        {
            "train": sorted(train.engine.unique().tolist()),
            "validation": sorted(val.engine.unique().tolist()),
            "sensors": prep.sensors,
            "single_sensor": prep.sensors[single],
        },
    )
    rows = []
    for name in MODELS:
        model, history, elapsed = fit_candidate(name, training, endpoint, settings, seed, single)
        pred = predict_candidate(name, model, endpoint, settings, single)
        row = {
            "subset": subset,
            "seed": seed,
            "model": name,
            "train_seconds": elapsed,
            "single_sensor": prep.sensors[single],
            **regression_metrics(endpoint.y, pred),
        }
        rows.append(row)
        prediction_frame(endpoint, pred).to_csv(output / f"{name}_predictions.csv", index=False)
        if history:
            pd.DataFrame(history).to_csv(output / f"{name}_history.csv", index=False)
        pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False)
        print(f"{subset} seed {seed} {name}: validation RMSE {row['rmse']:.2f}", flush=True)
    write_json(output / "complete.json", {"complete": True})
    return rows


def evaluate_endpoints(name, model, frame, prep, settings, single, radius, output, split):
    windows = make_windows(prep.transform(frame), prep.sensors, settings["window"], last_only=True)
    pred = predict_candidate(name, model, windows, settings, single)
    result = prediction_frame(windows, pred)
    result["lower"], result["upper"] = interval_bounds(pred, radius, settings["cap"])
    result["error"] = result.predicted_rul - result.rul
    result["absolute_error"] = result.error.abs()
    result["covered"] = (result.rul >= result.lower) & (result.rul <= result.upper)
    result.to_csv(output / f"{split}_predictions.csv", index=False)
    result.nlargest(10, "absolute_error").to_csv(output / f"{split}_worst_engines.csv", index=False)
    result["stage"] = pd.cut(
        result.rul_raw, [-1, 30, 80, float("inf")], labels=["late_0_30", "middle_31_80", "early_81_plus"]
    )
    stage_rows = [
        {
            "stage": str(stage),
            **regression_metrics(group.rul, group.predicted_rul),
            "mean_signed_error": float(group.error.mean()),
            "interval_coverage": float(group.covered.mean()),
        }
        for stage, group in result.groupby("stage", observed=True)
    ]
    pd.DataFrame(stage_rows).to_csv(output / f"{split}_stage_errors.csv", index=False)
    return {
        "model": name,
        "split": split,
        **regression_metrics(windows.y, pred),
        **bootstrap_rmse(windows.y, pred),
        "interval_coverage": float(result.covered.mean()),
        "mean_interval_width": float((result.upper - result.lower).mean()),
    }


def final_evaluation(data, subset, ids, output, settings, summary):
    if (output / "complete.json").exists():
        return
    output.mkdir(parents=True, exist_ok=True)
    ranking = summary[summary.subset == subset].sort_values(["rmse_mean", "model"])
    selected = ranking.iloc[0].model
    final_settings = dict(settings)
    if selected in ("lstm", "gru"):
        epochs = []
        for seed in settings["seeds"]:
            history = pd.read_csv(
                output.parent / "development" / subset / f"seed_{seed}" / f"{selected}_history.csv"
            )
            epochs.append(int(history.loc[history.validation_rmse.idxmin(), "epoch"]))
        final_settings["epochs"] = max(1, int(np.median(epochs)))
    fit_frame = data[data.engine.isin(ids["development"])]
    prep = Preprocessor.fit(fit_frame)
    single = select_single(fit_frame, prep.sensors)
    # All choices below are frozen before reading calibration/assessment/test targets.
    frozen = {
        "model": selected,
        "settings": final_settings,
        "engine_cohorts": ids,
        "sensors": prep.sensors,
        "single_sensor": prep.sensors[single],
        "selection_rule": "lowest mean validation endpoint RMSE across five development seeds",
        "interval_target": "capped RUL at one endpoint per engine",
    }
    write_json(output / "frozen_protocol.json", frozen)
    training = make_windows(prep.transform(fit_frame), prep.sensors, settings["window"], settings["stride"])
    model, history, elapsed = fit_candidate(selected, training, None, final_settings, 2026, single)
    joblib.dump(prep, output / "preprocessor.joblib")
    if selected in ("lstm", "gru"):
        torch.save(
            {"state_dict": model.state_dict(), "kind": selected, "sensors": len(prep.sensors), "hidden": 48},
            output / f"{selected}.pt",
        )
    else:
        joblib.dump(model, output / f"{selected}.joblib")
    if history:
        pd.DataFrame(history).to_csv(output / "refit_history.csv", index=False)
    calibration_frame = validation_endpoints(
        data[data.engine.isin(ids["calibration"])], settings["window"], 31415
    )
    calibration = make_windows(
        prep.transform(calibration_frame), prep.sensors, settings["window"], last_only=True
    )
    calibration_pred = predict_candidate(selected, model, calibration, settings, single)
    radius = conformal_radius(calibration.y, calibration_pred, settings["alpha"])
    prediction_frame(calibration, calibration_pred).to_csv(
        output / "calibration_predictions.csv", index=False
    )
    write_json(
        output / "calibration.json",
        {
            "radius_cycles": radius,
            "alpha": settings["alpha"],
            "n_engines": len(calibration.y),
            "target": "capped RUL",
            "assumption": "exchangeable engine endpoints; empirical official-test coverage is reported separately",
            "reference": "https://arxiv.org/abs/2107.07511",
        },
    )
    assessment_frame = data[data.engine.isin(ids["assessment"])]
    truncated = validation_endpoints(assessment_frame, settings["window"], 31415)
    metrics = [
        evaluate_endpoints(selected, model, truncated, prep, settings, single, radius, output, "assessment")
    ]
    replay = make_windows(prep.transform(assessment_frame), prep.sensors, settings["window"])
    replay_pred = predict_candidate(selected, model, replay, settings, single)
    replay_result = prediction_frame(replay, replay_pred)
    # No intervals on replay: endpoint calibration does not establish simultaneous coverage over time.
    replay_result.to_csv(output / "assessment_replay_predictions.csv", index=False)
    warning_rows = []
    warning, details = warning_metrics(
        replay.metadata, replay_pred <= settings["horizon"], settings["horizon"], settings["persistence"]
    )
    warning_rows.append({"model": selected, **warning})
    details.to_csv(output / "assessment_warning_engines.csv", index=False)
    healthy = fit_frame[fit_frame.cycle <= 20].sensor_2
    observed = (
        assessment_frame.set_index(["engine", "cycle"])
        .loc[pd.MultiIndex.from_frame(replay.metadata[["engine", "cycle"]]), "sensor_2"]
        .to_numpy()
    )
    alarms = np.abs((observed - healthy.mean()) / max(float(healthy.std()), 1e-8)) >= 3
    warning, details = warning_metrics(replay.metadata, alarms, settings["horizon"], settings["persistence"])
    warning_rows.append({"model": "sensor_2_3sigma", **warning})
    details.to_csv(output / "threshold_warning_engines.csv", index=False)
    pd.DataFrame(warning_rows).to_csv(output / "assessment_warning_metrics.csv", index=False)
    # Official tests are evaluated once after all development choices and calibration are fixed.
    test_path = ROOT / "data" / "raw" / f"test_{subset}.txt"
    rul_path = ROOT / "data" / "raw" / f"RUL_{subset}.txt"
    test = label_test(load_trajectories(test_path), rul_path, settings["cap"])
    metrics.append(
        evaluate_endpoints(selected, model, test, prep, settings, single, radius, output, "official_test")
    )
    # Raw targets are provided as an additional convention; capped metrics remain the prespecified primary metric.
    test_predictions = pd.read_csv(output / "official_test_predictions.csv")
    write_json(
        output / "official_test_uncapped_metrics.json",
        regression_metrics(test_predictions.rul_raw, test_predictions.predicted_rul),
    )
    write_json(
        output / "test_source_hashes.json",
        {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (test_path, rul_path)},
    )
    pd.DataFrame(metrics).to_csv(output / "final_metrics.csv", index=False)
    write_json(output / "complete.json", {"complete": True, "refit_seconds": elapsed})
    print(
        f"{subset}: frozen {selected}; official test RMSE {metrics[-1]['rmse']:.2f}; interval coverage {metrics[-1]['interval_coverage']:.1%}",
        flush=True,
    )


def study(args):
    output = Path(args.output)
    settings = {
        "seeds": args.seeds,
        "subsets": args.subsets,
        "epochs": args.epochs,
        "trees": args.trees,
        "window": 30,
        "stride": 2,
        "cap": 125,
        "horizon": 30,
        "persistence": 3,
        "alpha": 0.1,
    }
    if len(set(args.seeds)) != len(args.seeds) or len(args.seeds) < 3 or args.epochs < 1 or args.trees < 1:
        raise ValueError("Require at least three distinct seeds and positive epochs and trees")
    datasets = {
        subset: label_training(load_trajectories(ROOT / "data" / "raw" / f"train_{subset}.txt"))
        for subset in args.subsets
    }
    assignments = {subset: cohorts(data.engine.unique().tolist()) for subset, data in datasets.items()}
    protocol = {
        "settings": settings,
        "cohorts": assignments,
        "model_candidates": MODELS,
        "data_hashes": {
            s: hashlib.sha256((ROOT / "data" / "raw" / f"train_{s}.txt").read_bytes()).hexdigest()
            for s in args.subsets
        },
        "fd001_prior_exposure": "Previously used in v1 exploration; assessment is retrospective, not globally pristine.",
        "fd003_prior_exposure": "No FD003 models or outcome exploration before this protocol.",
        "endpoint_sampling_seed": 31415,
        "cohort_seed": 2026,
        "warning_rule": "predicted RUL <=30 for 3 consecutive cycles; no threshold tuning on assessment",
        "version": 2,
    }
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("Cannot change an existing study protocol; use a new folder")
    else:
        write_json(protocol_path, protocol)
    if (output / "complete.json").exists():
        print("Study already completed; saved results preserved.")
        return
    all_rows = []
    for subset, data in datasets.items():
        development = data[data.engine.isin(assignments[subset]["development"])]
        for seed in args.seeds:
            all_rows.extend(
                development_run(
                    development, subset, seed, output / "development" / subset / f"seed_{seed}", settings
                )
            )
            pd.DataFrame(all_rows).to_csv(output / "development_metrics.csv", index=False)
    records = pd.DataFrame(all_rows)
    summary = records.groupby(["subset", "model"], as_index=False).agg(
        rmse_mean=("rmse", "mean"),
        rmse_std=("rmse", "std"),
        mae_mean=("mae", "mean"),
        mae_std=("mae", "std"),
        train_seconds_mean=("train_seconds", "mean"),
        n_seeds=("seed", "nunique"),
    )
    summary.to_csv(output / "development_summary.csv", index=False)
    for subset, data in datasets.items():
        final_evaluation(data, subset, assignments[subset], output / subset, settings, summary)
    write_json(output / "complete.json", {"complete": True})
    print(f"Research study completed: {output.resolve()}", flush=True)
