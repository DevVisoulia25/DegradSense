"""Reproducible FD001 training, held-out replay, and locked test evaluation."""

import hashlib
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from rul.data import (
    ROOT,
    Preprocessor,
    feature_names,
    label_test,
    label_training,
    load_trajectories,
    make_windows,
    split_engines,
    window_features,
)
from rul.metrics import regression_metrics, warning_metrics
from rul.models import SequenceRegressor, classical_model, fit_sequence, sequence_predict


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def validation_endpoints(frame, window, seed):
    """One reproducible truncated observation per validation engine."""
    parts = []
    for engine_id, group in frame.groupby("engine", sort=True):
        rng = np.random.default_rng(np.random.SeedSequence([seed, int(engine_id)]))
        end = int(rng.integers(window, len(group))) if len(group) > window else len(group)
        parts.append(group.iloc[:end])
    return pd.concat(parts, ignore_index=True)


def predict(name, model, windows, cap):
    if name in ("lstm", "gru"):
        return sequence_predict(model, windows.x, cap)
    return np.clip(model.predict(window_features(windows.x)), 0, cap)


def prediction_frame(windows, predictions):
    frame = windows.metadata.copy()
    frame["predicted_rul"] = predictions
    return frame


def train(args):
    output = (
        Path(args.output)
        if args.output
        else ROOT / "artifacts" / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    )
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty; use a new run name")
    if (
        args.epochs < 1
        or args.trees < 1
        or args.cap <= args.horizon
        or args.stride < 1
        or args.window < 2
        or args.persistence < 1
        or args.horizon < 1
    ):
        raise ValueError(
            "Require positive epochs/trees/stride/horizon/persistence, window >=2, and cap > horizon"
        )
    if not 0 < args.validation_fraction < 1 or len(set(args.models)) != len(args.models):
        raise ValueError("Validation fraction must be between 0 and 1; model names must be unique")
    raw_path = ROOT / "data" / "raw" / "train_FD001.txt"
    data = label_training(load_trajectories(raw_path), args.cap)
    train_frame, val_frame = split_engines(data, args.validation_fraction, args.seed)
    prep = Preprocessor.fit(train_frame)
    training = make_windows(prep.transform(train_frame), prep.sensors, args.window, args.stride)
    # Replay uses stride 1, regardless of training stride, for warning persistence.
    validation = make_windows(prep.transform(val_frame), prep.sensors, args.window)
    endpoints = make_windows(
        prep.transform(validation_endpoints(val_frame, args.window, args.seed)),
        prep.sensors,
        args.window,
        last_only=True,
    )
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(prep, output / "preprocessor.joblib")
    config = {
        "subset": "FD001",
        "window": args.window,
        "stride": args.stride,
        "cap": args.cap,
        "seed": args.seed,
        "epochs": args.epochs,
        "trees": args.trees,
        "validation_fraction": args.validation_fraction,
        "horizon": args.horizon,
        "persistence": args.persistence,
        "train_engines": sorted(train_frame.engine.unique().tolist()),
        "validation_engines": sorted(val_frame.engine.unique().tolist()),
        "sensors": prep.sensors,
        "models": args.models,
        "selection_metric": "validation_endpoint_rmse",
        "test_used_during_training": False,
        "train_data_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "versions": {
            p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scikit-learn", "torch", "xgboost")
        },
    }
    write_json(output / "config.json", config)
    endpoints.metadata.to_csv(output / "validation_endpoints.csv", index=False)
    results, warning_results = [], []
    for name in args.models:
        print(f"Training {name} on {len(training.x)} windows / {len(prep.sensors)} sensors", flush=True)
        if name in ("lstm", "gru"):
            model, history = fit_sequence(name, training, endpoints, args.cap, args.epochs, args.seed)
            torch.save(
                {"state_dict": model.state_dict(), "kind": name, "sensors": len(prep.sensors), "hidden": 48},
                output / f"{name}.pt",
            )
            pd.DataFrame(history).to_csv(output / f"{name}_history.csv", index=False)
        else:
            model = classical_model(name, args.seed, args.trees)
            model.fit(window_features(training.x), training.y)
            joblib.dump(model, output / f"{name}.joblib")
            if name == "linear":
                pd.DataFrame(
                    {"feature": feature_names(prep.sensors), "coefficient": model[-1].coef_}
                ).to_csv(output / "linear_coefficients.csv", index=False)
            if hasattr(model, "feature_importances_"):
                pd.DataFrame(
                    {"feature": feature_names(prep.sensors), "importance": model.feature_importances_}
                ).to_csv(output / f"{name}_feature_importance.csv", index=False)
        for split, windows in (("validation_endpoint", endpoints), ("validation_replay", validation)):
            predictions = predict(name, model, windows, args.cap)
            frame = prediction_frame(windows, predictions)
            frame.to_csv(output / f"{name}_{split}_predictions.csv", index=False)
            results.append({"model": name, "split": split, **regression_metrics(windows.y, predictions)})
            if split == "validation_replay":
                summary, details = warning_metrics(
                    frame, predictions <= args.horizon, args.horizon, args.persistence
                )
                warning_results.append({"model": name, **summary})
                details.to_csv(output / f"{name}_warning_engines.csv", index=False)
        pd.DataFrame(results).to_csv(output / "metrics.csv", index=False)
        print(f"{name}: endpoint RMSE {results[-2]['rmse']:.2f}", flush=True)
    # A simple sensor alarm uses training-only nominal statistics. It is an
    # illustrative fixed rule; select/tune its threshold only on validation.
    sensor = "sensor_2" if "sensor_2" in prep.sensors else prep.sensors[0]
    healthy = train_frame[train_frame.cycle <= 20][sensor]
    mean, std = float(healthy.mean()), max(float(healthy.std()), 1e-8)
    observed = (
        val_frame.set_index(["engine", "cycle"])
        .loc[pd.MultiIndex.from_frame(validation.metadata[["engine", "cycle"]]), sensor]
        .to_numpy()
    )
    alerts = np.abs((observed - mean) / std) >= 3
    summary, details = warning_metrics(validation.metadata, alerts, args.horizon, args.persistence)
    warning_results.append({"model": f"threshold_{sensor}_3sigma", **summary})
    details.to_csv(output / "threshold_warning_engines.csv", index=False)
    write_json(
        output / "threshold_rule.json",
        {"sensor": sensor, "nominal_mean": mean, "nominal_std": std, "z_threshold": 3},
    )
    pd.DataFrame(warning_results).to_csv(output / "warning_metrics.csv", index=False)
    write_json(output / "complete.json", {"training_completed": True, "test_evaluated": False})
    print(f"Run saved to {output.resolve()}. Official test data was not evaluated.", flush=True)


def load_run(path):
    path = Path(path)
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    prep = joblib.load(path / "preprocessor.joblib")
    return path, config, prep


def load_model(path, name):
    if name in ("lstm", "gru"):
        state = torch.load(path / f"{name}.pt", map_location="cpu", weights_only=True)
        model = SequenceRegressor(state["sensors"], state["kind"], state["hidden"])
        model.load_state_dict(state["state_dict"])
        return model
    return joblib.load(path / f"{name}.joblib")


def evaluate_test(args):
    path, config, prep = load_run(args.run)
    if (path / "test_metrics.csv").exists():
        raise ValueError("This run was already test-evaluated. Results are in test_metrics.csv.")
    metrics = pd.read_csv(path / "metrics.csv")
    endpoints = metrics[metrics.split == "validation_endpoint"]
    best = endpoints.sort_values("rmse").iloc[0].model
    write_json(path / "test_selection.json", {"model": best, "rule": "Lowest validation endpoint RMSE"})
    data_dir = ROOT / "data" / "raw"
    frame = label_test(
        load_trajectories(data_dir / "test_FD001.txt"), data_dir / "RUL_FD001.txt", config["cap"]
    )
    windows = make_windows(prep.transform(frame), prep.sensors, config["window"], last_only=True)
    pred = predict(best, load_model(path, best), windows, config["cap"])
    prediction_frame(windows, pred).to_csv(path / "test_predictions.csv", index=False)
    result = {"model": best, "split": "official_test_endpoint_capped", **regression_metrics(windows.y, pred)}
    pd.DataFrame([result]).to_csv(path / "test_metrics.csv", index=False)
    write_json(
        path / "test_data_hashes.json",
        {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (data_dir / "test_FD001.txt", data_dir / "RUL_FD001.txt")
        },
    )
    print(json.dumps(result, indent=2), flush=True)


def fusion_study(args):
    """Decision Tree single-sensor and leave-one-sensor-out study on fixed validation only."""
    path, config, prep = load_run(args.run)
    destination = path / "fusion"
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Fusion study already exists; keep it as a reproducible result")
    raw_path = ROOT / "data" / "raw" / "train_FD001.txt"
    if hashlib.sha256(raw_path.read_bytes()).hexdigest() != config["train_data_sha256"]:
        raise ValueError("Training dataset has changed since this run")
    data = label_training(load_trajectories(raw_path), config["cap"])
    training_frame = data[data.engine.isin(config["train_engines"])]
    validation_frame = data[data.engine.isin(config["validation_engines"])]
    train_windows = make_windows(
        prep.transform(training_frame), prep.sensors, config["window"], config["stride"]
    )
    endpoint_frame = validation_endpoints(validation_frame, config["window"], config["seed"])
    val_windows = make_windows(prep.transform(endpoint_frame), prep.sensors, config["window"], last_only=True)
    replay_windows = make_windows(prep.transform(validation_frame), prep.sensors, config["window"])
    destination.mkdir(exist_ok=True)
    experiments = [("all_sensors", list(range(len(prep.sensors))))]
    experiments += [(f"only_{s}", [i]) for i, s in enumerate(prep.sensors)]
    experiments += [
        (f"without_{s}", [j for j in range(len(prep.sensors)) if j != i]) for i, s in enumerate(prep.sensors)
    ]
    rows, warnings = [], []
    for name, indices in experiments:
        model = classical_model("dt", config["seed"])
        model.fit(window_features(train_windows.x[:, :, indices]), train_windows.y)
        pred = np.clip(model.predict(window_features(val_windows.x[:, :, indices])), 0, config["cap"])
        row = {"experiment": name, "model": "dt", "sensor_count": len(indices), **regression_metrics(val_windows.y, pred)}
        rows.append(row)
        replay_predictions = np.clip(
            model.predict(window_features(replay_windows.x[:, :, indices])), 0, config["cap"]
        )
        summary, _ = warning_metrics(
            replay_windows.metadata,
            replay_predictions <= config["horizon"],
            config["horizon"],
            config["persistence"],
        )
        warnings.append({"experiment": name, **summary})
        print(f"{name}: validation endpoint RMSE {row['rmse']:.2f}", flush=True)
        pd.DataFrame(rows).to_csv(destination / "metrics.csv", index=False)
        pd.DataFrame(warnings).to_csv(destination / "warning_metrics.csv", index=False)
    write_json(
        destination / "protocol.json",
        {
            "model": "dt",
            "max_depth": 14,
            "min_samples_leaf": 3,
            "seed": config["seed"],
            "split": "validation_endpoint",
            "test_used": False,
            "warning_horizon": config["horizon"],
            "warning_persistence": config["persistence"],
        },
    )


def explain(args):
    import shap

    path, config, prep = load_run(args.run)
    if args.model not in ("dt", "xgb") or args.model not in config["models"]:
        raise ValueError("Choose a Decision Tree or XGBoost model that was trained in this run")
    frame = label_training(load_trajectories(ROOT / "data" / "raw" / "train_FD001.txt"), config["cap"])
    frame = frame[frame.engine.isin(config["validation_engines"])]
    windows = make_windows(prep.transform(frame), prep.sensors, config["window"])
    rng = np.random.default_rng(config["seed"])
    indices = rng.choice(len(windows.x), min(100, len(windows.x)), replace=False)
    x = window_features(windows.x[indices])
    values = shap.TreeExplainer(
        load_model(path, args.model), feature_perturbation="tree_path_dependent"
    ).shap_values(x)
    features = pd.DataFrame(
        {"feature": feature_names(prep.sensors), "mean_absolute_shap": np.abs(values).mean(0)}
    )
    features.to_csv(path / f"{args.model}_shap_features.csv", index=False)
    features["sensor"] = features.feature.str.split("__").str[0]
    features.groupby("sensor").mean_absolute_shap.sum().sort_values(ascending=False).to_csv(
        path / f"{args.model}_shap_sensors.csv"
    )
    write_json(
        path / f"{args.model}_shap_protocol.json",
        {
            "n_samples": len(indices),
            "seed": config["seed"],
            "split": "validation",
            "method": "tree_path_dependent",
            "interpretation": "Model attribution under correlated features; not causal evidence.",
        },
    )
    print("SHAP feature and sensor summaries saved.", flush=True)
