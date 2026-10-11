"""Completed research evidence, final engine replay and unlabeled inference."""

import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from rul.data import COLUMNS, ROOT
from rul.inference import predict_history


def render_research(study_root=None):
    study_root = ROOT / "artifacts" if study_root is None else study_root
    studies = sorted(
        [
            p
            for p in study_root.iterdir()
            if p.is_dir() and (p / "protocol.json").exists() and (p / "complete.json").exists()
        ],
        reverse=True,
    )
    if not studies:
        st.info("The prespecified research study is running or has not been started yet.")
        st.code(r".\.venv\Scripts\python.exe -m rul study")
        return
    study = st.sidebar.selectbox("Research study", studies, format_func=lambda p: p.name)
    protocol = json.loads((study / "protocol.json").read_text())
    subset = st.sidebar.selectbox("Dataset", protocol["settings"]["subsets"])
    path = study / subset
    frozen = json.loads((path / "frozen_protocol.json").read_text())
    calibration = json.loads((path / "calibration.json").read_text())
    summary = pd.read_csv(study / "development_summary.csv")
    summary = summary[summary.subset == subset].sort_values("rmse_mean")
    final = pd.read_csv(path / "final_metrics.csv")
    official = final[final.split == "official_test"].iloc[0]
    st.sidebar.caption(f"Frozen final model: {frozen['model']}")
    st.sidebar.caption("60 development · 20 calibration · 20 assessment engines")
    st.subheader("Evidence from repeated and final evaluations")
    cols = st.columns(3)
    cols[0].metric("Official test RMSE", f"{official.rmse:.2f} cycles")
    cols[1].metric("Official test MAE", f"{official.mae:.2f} cycles")
    cols[2].metric("Observed endpoint coverage", f"{official.interval_coverage:.0%}")
    st.caption(
        f"100 official test engines · capped RUL · 95% engine-bootstrap RMSE interval "
        f"{official.rmse_ci_low:.2f}–{official.rmse_ci_high:.2f} cycles. "
        "Interval coverage is empirical; nominal target is 90%."
    )
    evidence, warning, errors, inference, methods = st.tabs(
        [
            "Repeated experiments",
            "Final warning replay",
            "Failure analysis",
            "Predict from a file",
            "Frozen protocol",
        ]
    )
    with evidence:
        st.write(
            "Five development splits compare model families and controlled ablations. "
            "Bars show mean validation endpoint RMSE; error bars show variability across seeds."
        )
        st.plotly_chart(
            px.bar(
                summary,
                x="model",
                y="rmse_mean",
                error_y="rmse_std",
                labels={"rmse_mean": "Mean RMSE (cycles)", "model": "Candidate"},
            ),
            width="stretch",
        )
        st.dataframe(summary.drop(columns="subset"), hide_index=True, width="stretch")
        st.caption(
            "rf_latest uses current readings without window statistics. rf_single uses one sensor "
            "chosen by training-only correlation. Seed splits overlap; their standard deviation is "
            "a stability summary, not a confidence interval or significance test."
        )
        st.write("Final assessment and official test metrics")
        st.dataframe(final, hide_index=True, width="stretch")
        st.download_button(
            "Download research comparison",
            summary.to_csv(index=False),
            f"{subset}_comparison.csv",
            "text/csv",
        )
    with warning:
        st.write(
            "A fixed warning rule is evaluated on 20 assessment-engine lifetimes, excluded from the "
            "new study's development fitting and calibration."
        )
        st.dataframe(pd.read_csv(path / "assessment_warning_metrics.csv"), hide_index=True, width="stretch")
        if subset == "FD001":
            st.caption(
                "FD001 assessment is retrospective because version one used this dataset. "
                "FD003 was first modeled under the frozen research protocol."
            )
        replay = pd.read_csv(path / "assessment_replay_predictions.csv")
        engine = st.selectbox("Assessment engine", sorted(replay.engine.unique()))
        history = replay[replay.engine == engine].sort_values("cycle")
        cutoff = st.slider(
            "Assessment cycle", int(history.cycle.min()), int(history.cycle.max()), int(history.cycle.max())
        )
        visible = history[history.cycle <= cutoff]
        fig = px.line(visible, x="cycle", y=["predicted_rul", "rul"], labels={"value": "Remaining cycles"})
        fig.add_hline(y=30, line_dash="dot", annotation_text="Warning horizon")
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Warning confirmed after predicted RUL ≤30 for three consecutive cycles. "
            "No interval band is shown here: endpoint calibration does not establish coverage over a whole trajectory."
        )
    with errors:
        predictions = pd.read_csv(path / "official_test_predictions.csv")
        predictions = predictions.sort_values("rul")
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=predictions.rul, y=predictions.upper, mode="lines", line={"width": 0}, showlegend=False
            )
        )
        fig.add_trace(
            go.Scatter(
                x=predictions.rul,
                y=predictions.lower,
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                name="Calibrated endpoint interval",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=predictions.rul,
                y=predictions.predicted_rul,
                mode="markers",
                name="Predicted RUL",
                text=predictions.engine,
                hovertemplate="Engine %{text}<br>Actual %{x}<br>Predicted %{y}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(x=[0, 125], y=[0, 125], mode="lines", name="Perfect prediction", line={"dash": "dash"})
        )
        fig.update_layout(xaxis_title="Actual capped RUL", yaxis_title="Predicted RUL (cycles)")
        st.plotly_chart(fig, width="stretch")
        st.write("Error by true remaining-life stage")
        st.dataframe(pd.read_csv(path / "official_test_stage_errors.csv"), hide_index=True)
        st.write("Engines with the largest errors")
        st.dataframe(pd.read_csv(path / "official_test_worst_engines.csv"), hide_index=True, width="stretch")
        st.caption(
            "Positive signed error overestimates life. A stage table localizes failures; it does not establish their physical cause."
        )
    with inference:
        st.write(
            "Upload an engine history to estimate remaining cycles with the frozen model. "
            "The file must use the selected dataset's sensor definitions and units."
        )
        st.caption(
            "Accepted: NASA whitespace text with 26 columns, or CSV with the headers below. "
            "Cycles must begin at 1 and be consecutive within each engine."
        )
        st.code(",".join(COLUMNS))
        uploaded = st.file_uploader("Sensor history", type=["txt", "csv"])
        if uploaded is not None:
            try:
                if uploaded.name.lower().endswith(".csv"):
                    frame = pd.read_csv(uploaded)
                else:
                    frame = pd.read_csv(uploaded, sep=r"\s+", header=None)
                    if frame.shape[1] != 26:
                        raise ValueError("NASA text requires exactly 26 columns")
                    frame.columns = COLUMNS
                results = predict_history(frame, study, subset)
                st.dataframe(results, hide_index=True, width="stretch")
                if results.history_padded.any():
                    st.info(
                        "Some histories have fewer than 30 cycles and were padded with their first observation."
                    )
                st.download_button(
                    "Download predictions", results.to_csv(index=False), "rul_predictions.csv", "text/csv"
                )
            except (ValueError, KeyError, TypeError) as error:
                st.error(str(error))
        st.caption(
            "Intervals concern capped RUL (maximum 125 cycles) at a single observation. "
            "They do not express failure probabilities. Coverage may change for external equipment or operating conditions."
        )
    with methods:
        st.write(
            "Model family is selected by mean development RMSE across five seeds. The selected model "
            "is refitted on all 60 development engines. Twenty separate engine endpoints calibrate "
            "absolute-residual intervals. Assessment outcomes and official test labels are read only "
            "after these choices are frozen. Official test results are not used to retune this release."
        )
        st.write(
            "Error intervals use 2,000 bootstrap samples of engine endpoints. "
            "The two datasets are modeled separately; this is a replication, not cross-dataset transfer."
        )
        st.json(frozen)
        st.json(calibration)
        st.download_button(
            "Download study protocol",
            json.dumps(protocol, indent=2),
            "study_protocol.json",
            "application/json",
        )
