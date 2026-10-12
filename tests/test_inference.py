import json

import joblib
import numpy as np
import pandas as pd

from rul.data import COLUMNS, Preprocessor, label_training, make_windows, window_features
from rul.inference import predict_history
from rul.models import classical_model


def test_unlabeled_inference_ignores_supplied_targets_and_retains_only_outputs(tmp_path):
    rows = [[1, cycle, 0, 0, 0, *[cycle + sensor for sensor in range(21)]] for cycle in range(1, 41)]
    raw = pd.DataFrame(rows, columns=COLUMNS)
    training = label_training(raw)
    prep = Preprocessor.fit(training)
    windows = make_windows(prep.transform(training), prep.sensors, 30)
    model = classical_model("dt", trees=2)
    model.fit(window_features(windows.x), windows.y)
    path = tmp_path / "FD001"
    path.mkdir()
    (path / "complete.json").write_text('{"complete": true}')
    (path / "frozen_protocol.json").write_text(
        json.dumps({"model": "dt", "settings": {"cap": 125, "window": 30}, "single_sensor": prep.sensors[0]})
    )
    (path / "calibration.json").write_text('{"radius_cycles": 10}')
    joblib.dump(prep, path / "preprocessor.joblib")
    joblib.dump(model, path / "dt.joblib")
    original = predict_history(raw, tmp_path, "FD001")
    altered = raw.copy()
    altered["rul"] = -999999
    altered["rul_raw"] = 999999
    repeated = predict_history(altered, tmp_path, "FD001")
    np.testing.assert_array_equal(original.predicted_rul, repeated.predicted_rul)
    assert "rul" not in original and "rul_raw" not in original
    assert original.iloc[0].cycle == 40
    assert not original.iloc[0].history_padded
    assert original.iloc[0].lower <= original.iloc[0].predicted_rul <= original.iloc[0].upper
