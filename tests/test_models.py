import numpy as np
import pytest
import torch

from rul.models import SequenceRegressor


@pytest.mark.parametrize("kind", ["lstm", "gru"])
def test_sequence_prediction_has_one_output_per_engine_window(kind):
    model = SequenceRegressor(3, kind)
    x = torch.from_numpy(np.ones((4, 30, 3), dtype=np.float32))
    assert model(x).shape == (4,)
    model(x).sum().backward()
    assert all(p.grad is not None for p in model.parameters())
