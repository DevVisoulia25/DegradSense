from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_dashboard_renders_without_exceptions():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
    app.run(timeout=60)
    app.sidebar.radio[0].set_value("Single-split experiment").run(timeout=60)
    assert not app.exception


def test_dashboard_model_and_replay_controls():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
    app.run(timeout=60)
    app.sidebar.radio[0].set_value("Single-split experiment").run(timeout=60)
    if not app.slider:
        return  # Empty-state dashboard before a first training run.
    app.sidebar.selectbox[1].select("dt").run(timeout=60)
    assert not app.exception
    slider = app.slider[0]
    slider.set_value(slider.min).run(timeout=60)
    assert not app.exception
    assert next(m for m in app.metric if m.label == "Current operating cycle").value == str(int(slider.min))


def test_research_dashboard_switches_datasets_and_replays_assessment():
    from rul.data import ROOT

    if not (ROOT / "artifacts" / "research_v3_college" / "complete.json").exists():
        import pytest

        pytest.skip("Research integration requires a completed local study")
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
    app.run(timeout=60)
    assert not app.exception
    assert app.sidebar.radio[0].value == "Research evidence"
    assert any(metric.label == "Official test RMSE" for metric in app.metric)
    app.sidebar.selectbox[1].select("FD003").run(timeout=60)
    assert not app.exception
    app.slider[0].set_value(app.slider[0].min).run(timeout=60)
    assert not app.exception
