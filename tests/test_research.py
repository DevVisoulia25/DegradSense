import numpy as np
import pandas as pd
import pytest

from rul.data import label_training
from rul.experiment import validation_endpoints
from rul.statistics import bootstrap_rmse, conformal_radius, interval_bounds
from rul.study import cohorts, inputs, select_single
from rul.metrics import regression_metrics


def test_three_cohorts_are_disjoint_reproducible_and_cover_all_engines():
    result = cohorts(list(range(1, 101)))
    assert {key: len(value) for key, value in result.items()} == {
        "development": 60,
        "calibration": 20,
        "assessment": 20,
    }
    groups = [set(value) for value in result.values()]
    assert not (groups[0] & groups[1] or groups[1] & groups[2] or groups[0] & groups[2])
    assert set.union(*groups) == set(range(1, 101))
    assert cohorts(list(range(1, 101))) == result


def test_finite_sample_conformal_quantile_is_exact_order_statistic():
    scores = np.arange(1, 21)
    assert conformal_radius(np.zeros(20), scores, alpha=0.1) == 19
    with pytest.raises(ValueError, match="Too few"):
        conformal_radius([0], [1], alpha=0.1)
    lower, upper = interval_bounds([2, 120], 10, 125)
    np.testing.assert_array_equal(lower, [0, 110])
    np.testing.assert_array_equal(upper, [12, 125])


def test_engine_bootstrap_is_reproducible_and_contains_rmse_for_constant_error():
    result = bootstrap_rmse([1, 2, 3], [3, 4, 5])
    assert result["rmse_ci_low"] == result["rmse_ci_high"] == 2
    assert result == bootstrap_rmse([1, 2, 3], [3, 4, 5])
    with pytest.raises(ValueError):
        bootstrap_rmse([[1, 2]], [[3, 4]])


def test_metrics_reject_silent_broadcasting():
    with pytest.raises(ValueError, match="matching"):
        regression_metrics([1, 2], [[1], [2]])
    with pytest.raises(ValueError, match="matching"):
        conformal_radius([1, 2], [1])


def test_sensor_selection_uses_only_supplied_training_frame():
    train = pd.DataFrame({"sensor_1": [0, 1, 2, 3], "sensor_2": [2, 0, 3, 1], "rul_raw": [0, 1, 2, 3]})
    assert select_single(train, ["sensor_1", "sensor_2"]) == 0
    x = np.ones((2, 30, 3), dtype=np.float32)
    assert inputs(x, "rf_latest", 0).shape == (2, 3)
    assert inputs(x, "rf_single", 0).shape == (2, 6)


def test_endpoint_cutoffs_are_stable_for_each_engine_across_seed_splits():
    frame = pd.DataFrame({"engine": np.repeat([1, 2], 100), "cycle": np.tile(np.arange(1, 101), 2)})
    frame = label_training(frame)
    first = validation_endpoints(frame, 30, 31415)
    second = validation_endpoints(frame[frame.engine == 2], 30, 31415)
    # Engine-specific random streams keep paired comparisons on the same cycle.
    assert first[first.engine == 2].cycle.max() == second.cycle.max()
