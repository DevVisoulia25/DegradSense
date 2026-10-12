import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from rul.data import COLUMNS
from rul.inference import predict_history

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "deployment" / "studies" / "research_v3_college"


def test_packaged_models_match_manifest_and_predict_without_raw_data():
    manifest = json.loads((STUDY / "model_manifest.json").read_text())
    for name, expected in manifest.items():
        assert hashlib.sha256((STUDY / name).read_bytes()).hexdigest() == expected
    # Shape-valid synthetic readings exercise loading and inference, not model accuracy.
    rows = [[1, cycle, 0, 0, 0, *[float(sensor) for sensor in range(21)]] for cycle in range(1, 31)]
    frame = pd.DataFrame(rows, columns=COLUMNS)
    for subset in ("FD001", "FD003"):
        result = predict_history(frame, STUDY, subset)
        assert len(result) == 1
        assert np.isfinite(result[["predicted_rul", "lower", "upper"]].to_numpy()).all()
        row = result.iloc[0]
        assert 0 <= row.lower <= row.predicted_rul <= row.upper <= 125


def test_cloud_entrypoint_renders_both_packaged_datasets():
    app = AppTest.from_file(str(ROOT / "deployment" / "app.py")).run(timeout=60)
    assert not app.exception
    assert app.title[0].value == "DegradSense"
    assert next(m for m in app.metric if m.label == "Official test RMSE").value == "16.28 cycles"
    app.sidebar.selectbox[1].select("FD003").run(timeout=60)
    assert not app.exception
    assert next(m for m in app.metric if m.label == "Official test RMSE").value == "14.94 cycles"
    app.slider[0].set_value(app.slider[0].min).run(timeout=60)
    assert not app.exception
