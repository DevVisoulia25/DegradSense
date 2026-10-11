import numpy as np
import pandas as pd
import pytest

from rul.data import (
    COLUMNS,
    SENSORS,
    Preprocessor,
    label_test,
    label_training,
    load_trajectories,
    make_windows,
    split_engines,
    window_features,
)
from rul.metrics import regression_metrics, warning_metrics


def sample(engines=10, cycles=6):
    rows = []
    for engine in range(1, engines + 1):
        for cycle in range(1, cycles + 1):
            rows.append([engine, cycle, 0, 0, 0, *[engine * 100 + cycle + s for s in range(21)]])
    return pd.DataFrame(rows, columns=COLUMNS)


def test_split_has_no_shared_engines_and_scaler_uses_only_train():
    train, val = split_engines(sample())
    assert set(train.engine).isdisjoint(val.engine)
    prep = Preprocessor.fit(train)
    np.testing.assert_allclose(prep.scaler.mean_, train[prep.sensors].mean())
    transformed = prep.transform(train)
    np.testing.assert_allclose(transformed[prep.sensors].mean(), 0, atol=1e-6)
    shifted = val.copy()
    shifted[SENSORS] += 10000
    before = prep.scaler.mean_.copy()
    prep.transform(shifted)
    np.testing.assert_array_equal(before, prep.scaler.mean_)


def test_windows_never_cross_engines_or_include_future():
    data = label_training(sample(engines=2), cap=3)
    windows = make_windows(data, ["sensor_1"], length=3)
    for x, y, row in zip(windows.x, windows.y, windows.metadata.itertuples()):
        np.testing.assert_array_equal(x[:, 0], row.engine * 100 + np.arange(row.cycle - 2, row.cycle + 1))
        assert y == min(6 - row.cycle, 3)
    altered = data.copy()
    altered.loc[altered.cycle == 6, "sensor_1"] = -999
    changed = make_windows(altered, ["sensor_1"], length=3)
    np.testing.assert_array_equal(
        windows.x[windows.metadata.cycle < 6], changed.x[changed.metadata.cycle < 6]
    )


def test_short_test_windows_pad_only_existing_history(tmp_path):
    data = sample(engines=2, cycles=2)
    targets = tmp_path / "RUL.txt"
    targets.write_text("5\n10\n")
    labeled = label_test(data, targets)
    windows = make_windows(labeled, ["sensor_1"], length=4, last_only=True)
    np.testing.assert_array_equal(windows.x[0, :, 0], [101, 101, 101, 102])
    np.testing.assert_array_equal(windows.y, [5, 10])
    assert labeled.iloc[0].rul_raw == 6


def test_labels_cap_only_targets():
    data = label_training(sample(engines=1), cap=3)
    np.testing.assert_array_equal(data.rul, [3, 3, 3, 2, 1, 0])
    np.testing.assert_array_equal(data.rul_raw, [5, 4, 3, 2, 1, 0])


def test_feature_slope_is_per_cycle():
    x = np.array([[[1], [3], [5]]], dtype=np.float32)
    np.testing.assert_allclose(window_features(x)[0, -1], 2)


def test_nasa_score_penalizes_overestimation_more():
    assert regression_metrics([50], [60])["nasa_score_sum"] > regression_metrics([50], [40])["nasa_score_sum"]
    assert regression_metrics([50], [50])["rmse"] == 0


def test_warning_persistence_false_alarms_and_missed_engines():
    frame = pd.DataFrame(
        {"engine": [1] * 6 + [2] * 6, "cycle": list(range(1, 7)) * 2, "rul_raw": [5, 4, 3, 2, 1, 0] * 2}
    )
    alerts = [True, False, True, True, False, False] + [False] * 6
    summary, engines = warning_metrics(frame, alerts, horizon=3, persistence=2)
    assert summary["missed_engines"] == 1
    assert summary["mean_confirmed_lead_cycles"] == 2
    assert summary["false_positive_cycle_rate"] == 0.25
    assert engines.iloc[0].lead_cycles == 2


def test_warning_rejects_sparse_replay():
    frame = pd.DataFrame({"engine": [1, 1], "cycle": [1, 3], "rul_raw": [4, 2]})
    with pytest.raises(ValueError, match="consecutive"):
        warning_metrics(frame, [True, True])


def test_loader_rejects_duplicate_cycles(tmp_path):
    path = tmp_path / "invalid.txt"
    frame = sample(engines=1)
    pd.concat([frame, frame.iloc[[0]]]).to_csv(path, sep=" ", header=False, index=False)
    with pytest.raises(ValueError, match="Duplicate"):
        load_trajectories(path)
