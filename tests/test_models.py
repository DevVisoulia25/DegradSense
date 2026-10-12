import numpy as np
import pytest
import torch

from rul.models import SequenceRegressor, classical_model


def test_linear_regression_fits_a_linear_relationship():
    x = np.arange(20, dtype=float).reshape(-1, 1)
    model = classical_model("linear")
    model.fit(x, 2 * x[:, 0] + 3)
    np.testing.assert_allclose(model.predict([[20], [25]]), [43, 53], atol=1e-8)


def test_decision_tree_is_reproducible_and_removed_baselines_are_rejected():
    from sklearn.tree import DecisionTreeRegressor

    first = classical_model("dt", seed=42)
    second = classical_model("dt", seed=42)
    assert isinstance(first, DecisionTreeRegressor)
    x = np.arange(20, dtype=float).reshape(-1, 1)
    y = np.where(x[:, 0] < 10, 5, 50)
    first.fit(x, y)
    second.fit(x, y)
    np.testing.assert_array_equal(first.predict(x), second.predict(x))
    for removed in ("mean", "rf"):
        with pytest.raises(ValueError, match="Unknown classical"):
            classical_model(removed)


@pytest.mark.parametrize("kind", ["lstm", "gru"])
def test_sequence_prediction_has_one_output_per_engine_window(kind):
    model = SequenceRegressor(3, kind)
    x = torch.from_numpy(np.ones((4, 30, 3), dtype=np.float32))
    assert model(x).shape == (4,)
    model(x).sum().backward()
    assert all(p.grad is not None for p in model.parameters())
