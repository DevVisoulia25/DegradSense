"""Prediction on unlabeled engine histories using a frozen research model."""

import json
from pathlib import Path

import joblib

from rul.data import load_trajectories, make_windows, validate_trajectories
from rul.experiment import load_model
from rul.statistics import interval_bounds
from rul.study import predict_candidate


def predict_history(frame, study_path, subset):
    path = Path(study_path) / subset
    if not (path / "complete.json").exists():
        raise ValueError("The selected research model is not finalized yet")
    protocol = json.loads((path / "frozen_protocol.json").read_text())
    calibration = json.loads((path / "calibration.json").read_text())
    prep = joblib.load(path / "preprocessor.joblib")
    validated = validate_trajectories(frame)
    # Dummy labels satisfy the shared window container; predictions never use them.
    validated["rul"] = 0
    validated["rul_raw"] = 0
    windows = make_windows(
        prep.transform(validated), prep.sensors, protocol["settings"]["window"], last_only=True
    )
    name = protocol["model"]
    single = prep.sensors.index(protocol["single_sensor"])
    predictions = predict_candidate(name, load_model(path, name), windows, protocol["settings"], single)
    result = windows.metadata[["engine", "cycle"]].copy()
    result["predicted_rul"] = predictions
    result["lower"], result["upper"] = interval_bounds(
        predictions, calibration["radius_cycles"], protocol["settings"]["cap"]
    )
    result["input_cycles"] = result.engine.map(validated.groupby("engine").size())
    result["history_padded"] = result.input_cycles < protocol["settings"]["window"]
    result["model"] = name
    return result


def predict_file(args):
    result = predict_history(load_trajectories(args.input), args.study, args.subset)
    if args.output:
        target = Path(args.output)
        if target.exists():
            raise ValueError("Prediction output already exists; choose another filename")
        result.to_csv(target, index=False)
    print(result.to_string(index=False))
