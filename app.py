"""Run with: .venv/Scripts/python.exe -m streamlit run app.py"""

import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from rul.data import ROOT, load_trajectories
from rul.experiment import load_run
from rul.research_dashboard import render_research
from rul.study import MODELS
from rul.models import MODEL_LABELS

st.set_page_config(page_title="DegradSense | Turbofan Prognostics", page_icon="⚙️", layout="wide")
st.title("DegradSense")
st.caption("Multisensor Turbofan Remaining Useful Life Prediction and Early Warning • NASA C-MAPSS")
completed_research = any(
    (p / "protocol.json").exists() and (p / "complete.json").exists()
    for p in (ROOT / "artifacts").iterdir()
    if p.is_dir()
)
view = st.sidebar.radio(
    "View", ["Research evidence", "Single-split experiment"], index=0 if completed_research else 1
)
if view == "Research evidence":
    render_research()
    st.stop()

runs = sorted(
    [
        p
        for p in (ROOT / "artifacts").glob("*")
        if p.is_dir() and (p / "complete.json").exists() and (p / "config.json").exists()
        and "linear" in json.loads((p / "config.json").read_text()).get("models", [])
    ],
    reverse=True,
)
if not runs:
    st.info("Train your first experiment to explore engine predictions here.")
    st.code(r".\.venv\Scripts\python.exe -m rul train --output artifacts/first_run")
    st.stop()

selected = st.sidebar.selectbox("Experiment", runs, format_func=lambda p: p.name)
run, config, prep = load_run(selected)
metrics = pd.read_csv(run / "metrics.csv")
metrics = metrics[metrics.model.isin(MODELS + ["svr"])]
available = metrics[metrics.split == "validation_endpoint"].model.tolist()
model = st.sidebar.selectbox(
    "Model",
    available,
    format_func=lambda name: MODEL_LABELS.get(name, name),
    index=available.index(metrics[metrics.split == "validation_endpoint"].sort_values("rmse").iloc[0].model),
)
st.sidebar.caption(
    f"{len(config['train_engines'])} training engines · "
    f"{len(config['validation_engines'])} validation engines"
)
st.sidebar.caption(f"{config['window']} cycles per window · {len(prep.sensors)} sensor channels")
st.sidebar.info(
    "Replay uses held-out validation engines. Predictions use sensor history through the displayed cycle."
)

overview, replay, fusion, explain, protocol = st.tabs(
    ["Model comparison", "Engine replay", "Sensor fusion", "Explanations", "Experiment details"]
)

with overview:
    st.subheader("Predict remaining operating cycles")
    st.write(
        "Compare models at one reproducible truncated observation per validation engine. "
        "Lower RMSE and MAE indicate better RUL estimates."
    )
    endpoints = metrics[metrics.split == "validation_endpoint"].sort_values("rmse")
    chosen = endpoints[endpoints.model == model].iloc[0]
    col1, col2, col3 = st.columns(3)
    col1.metric("Validation RMSE", f"{chosen.rmse:.2f} cycles")
    col2.metric("Validation MAE", f"{chosen.mae:.2f} cycles")
    col3.metric("Engines evaluated", int(chosen.n_predictions))
    st.plotly_chart(
        px.bar(
            endpoints.assign(model=endpoints.model.map(MODEL_LABELS)), x="model", y="rmse", color="model", labels={"rmse": "RMSE (cycles)", "model": "Model"}
        ),
        width="stretch",
    )
    st.dataframe(endpoints.drop(columns="split").assign(model=endpoints.model.map(MODEL_LABELS)), hide_index=True, width="stretch")
    predictions = pd.read_csv(run / f"{model}_validation_endpoint_predictions.csv")
    fig = px.scatter(
        predictions,
        x="rul",
        y="predicted_rul",
        hover_data=["engine", "cycle"],
        labels={"rul": "Actual capped RUL", "predicted_rul": "Predicted RUL"},
    )
    fig.add_trace(
        go.Scatter(
            x=[0, config["cap"]],
            y=[0, config["cap"]],
            mode="lines",
            name="Perfect prediction",
            line={"dash": "dash", "color": "gray"},
        )
    )
    st.plotly_chart(fig, width="stretch")
    if (run / "test_metrics.csv").exists():
        st.write("Official test result for the model selected using validation")
        st.dataframe(pd.read_csv(run / "test_metrics.csv"), hide_index=True)
    else:
        st.caption("Official test results are reserved for a separate final evaluation.")
    st.download_button("Download comparison", metrics.to_csv(index=False), "metrics.csv", "text/csv")

with replay:
    prediction_data = pd.read_csv(run / f"{model}_validation_replay_predictions.csv")
    engine = st.selectbox("Validation engine", sorted(prediction_data.engine.unique()))
    engine_predictions = prediction_data[prediction_data.engine == engine].sort_values("cycle")
    cutoff = st.slider(
        "Replay through operating cycle",
        int(engine_predictions.cycle.min()),
        int(engine_predictions.cycle.max()),
        int(engine_predictions.cycle.max()),
    )
    visible = engine_predictions[engine_predictions.cycle <= cutoff]
    current = visible.iloc[-1]
    cols = st.columns(3)
    cols[0].metric("Predicted remaining life", f"{current.predicted_rul:.1f} cycles")
    cols[1].metric("Current operating cycle", int(current.cycle))
    run_length = int((visible.predicted_rul <= config["horizon"]).iloc[-config["persistence"] :].sum())
    confirmed = len(visible) >= config["persistence"] and run_length == config["persistence"]
    cols[2].metric("Persistent warning", "Active" if confirmed else "Inactive")
    st.caption(
        f"Warning: predicted RUL ≤ {config['horizon']} for {config['persistence']} consecutive cycles. "
        "This is a benchmark demonstration; no uncertainty interval is estimated."
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=visible.cycle, y=visible.predicted_rul, name="Predicted RUL"))
    if st.checkbox("Show actual remaining life for evaluation", value=True):
        fig.add_trace(
            go.Scatter(x=visible.cycle, y=visible.rul, name="Actual capped RUL", line={"dash": "dash"})
        )
        fig.add_trace(
            go.Scatter(x=visible.cycle, y=visible.rul_raw, name="Actual uncapped RUL", line={"dash": "dot"})
        )
    fig.add_hline(y=config["horizon"], line_dash="dot", annotation_text="Warning horizon")
    fig.update_layout(xaxis_title="Operating cycle", yaxis_title="Remaining cycles")
    st.plotly_chart(fig, width="stretch")
    sensor_selection = st.multiselect("Sensor histories", prep.sensors, default=prep.sensors[:3])
    if sensor_selection:
        raw = load_trajectories(ROOT / "data" / "raw" / "train_FD001.txt")
        raw = raw[(raw.engine == engine) & (raw.cycle <= cutoff)]
        normalized = prep.transform(raw)
        long = normalized.melt(
            id_vars="cycle", value_vars=sensor_selection, var_name="sensor", value_name="standardized_reading"
        )
        st.plotly_chart(px.line(long, x="cycle", y="standardized_reading", color="sensor"), width="stretch")
        st.caption("Sensor readings standardized using training-engine statistics.")
    warnings = pd.read_csv(run / "warning_metrics.csv")
    st.write("Warning performance across validation engine lifetimes")
    st.dataframe(warnings, hide_index=True, width="stretch")
    st.caption(
        "Lead time counts the first persistent warning inside the last 30 cycles, "
        "or the configured horizon. Premature alerts contribute to the false-positive cycle rate. "
        "Replay starts after a full input window is available."
    )

with fusion:
    st.subheader("Does combining sensors help?")
    fusion_path = run / "fusion" / "metrics.csv"
    if fusion_path.exists():
        results = pd.read_csv(fusion_path)
        st.plotly_chart(
            px.bar(
                results.sort_values("rmse"),
                x="rmse",
                y="experiment",
                orientation="h",
                height=850,
                labels={"rmse": "Validation endpoint RMSE (cycles)"},
            ),
            width="stretch",
        )
        st.dataframe(results, hide_index=True, width="stretch")
        warning_path = run / "fusion" / "warning_metrics.csv"
        if warning_path.exists():
            st.write("Warning performance for the same sensor variants")
            st.dataframe(pd.read_csv(warning_path), hide_index=True, width="stretch")
        st.caption(
            "Every variant is retrained with the same Decision Tree settings, engine split, "
            "and validation observation points. Results describe model dependence, not causal interactions."
        )
    else:
        st.info(
            "Run the sensor fusion study to compare all sensors, individual sensors, and removed sensors."
        )
        st.code(f'.\\.venv\\Scripts\\python.exe -m rul fusion --run "{run.relative_to(ROOT)}"')

with explain:
    st.subheader("Which inputs influence the model?")
    shap_path = run / f"{model}_shap_sensors.csv"
    importance_path = run / f"{model}_feature_importance.csv"
    coefficient_path = run / "linear_coefficients.csv"
    if model == "linear" and coefficient_path.exists():
        coefficients = pd.read_csv(coefficient_path)
        coefficients["magnitude"] = coefficients.coefficient.abs()
        coefficients = coefficients.nlargest(20, "magnitude").sort_values("magnitude")
        st.plotly_chart(px.bar(coefficients, x="coefficient", y="feature", orientation="h"), width="stretch")
        st.caption("Linear coefficients for standardized window statistics. Correlated inputs affect coefficients; this is not causal evidence.")
    elif shap_path.exists():
        values = pd.read_csv(shap_path)
        st.plotly_chart(
            px.bar(
                values.sort_values("mean_absolute_shap"),
                x="mean_absolute_shap",
                y="sensor",
                orientation="h",
                labels={"mean_absolute_shap": "Summed mean absolute SHAP"},
            ),
            width="stretch",
        )
        st.caption(
            "SHAP values aggregated across each sensor's window-statistic features. "
            "Correlated sensors can share attribution; this does not establish causation."
        )
    elif importance_path.exists():
        values = pd.read_csv(importance_path).nlargest(20, "importance")
        st.plotly_chart(
            px.bar(values.sort_values("importance"), x="importance", y="feature", orientation="h"),
            width="stretch",
        )
        st.caption("Built-in tree importance. Run SHAP for a separate attribution view.")
    else:
        st.info(
            "Tree explanations are available for Decision Tree and XGBoost. "
            "Use the fusion study to investigate sensor dependence."
        )

with protocol:
    st.write(
        "Training fits sensor selection and normalization using training engines only. "
        "Windows contain past and current readings from one engine. "
        "The model sees sensors; engine IDs and target RUL are excluded from its inputs."
    )
    st.write(
        "Targets are capped at the configured RUL limit. RMSE, MAE, and NASA score here use those "
        "capped targets. Warning evaluation uses uncapped true RUL. "
        "The NASA score penalizes overestimating life more strongly than underestimating it."
    )
    st.write(
        "This implementation supports FD001. Other subsets require additional consideration of "
        "operating conditions. A single validation split cannot establish statistical significance; "
        "repeat the study across seeds before drawing final conclusions."
    )
    history_path = run / f"{model}_history.csv"
    if history_path.exists():
        history = pd.read_csv(history_path)
        st.plotly_chart(px.line(history, x="epoch", y="validation_rmse"), width="stretch")
    st.json(config)
    st.download_button(
        "Download experiment configuration", json.dumps(config, indent=2), "config.json", "application/json"
    )
