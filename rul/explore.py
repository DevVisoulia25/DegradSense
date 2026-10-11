"""Training-only dataset exploration saved as CSVs and interactive HTML charts."""

from pathlib import Path

import plotly.express as px

from rul.data import ROOT, SENSORS, label_training, load_trajectories, split_engines
from rul.experiment import write_json


def explore(args):
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new or empty EDA directory")
    data = label_training(load_trajectories(ROOT / "data" / "raw" / "train_FD001.txt"))
    train, _ = split_engines(data, seed=args.seed)
    output.mkdir(parents=True, exist_ok=True)
    lifetimes = train.groupby("engine").cycle.max().rename("lifetime_cycles").reset_index()
    lifetimes.to_csv(output / "lifetimes.csv", index=False)
    summary = train[SENSORS].describe().T
    summary["near_constant"] = summary["std"] <= 1e-8
    informative = summary.index[~summary.near_constant].tolist()
    summary["correlation_with_raw_rul"] = train[informative].corrwith(train.rul_raw)
    summary.to_csv(output / "sensor_summary.csv")
    correlation = train[informative].corr()
    correlation.to_csv(output / "sensor_correlations.csv")
    px.histogram(lifetimes, x="lifetime_cycles", nbins=20, title="Training engine lifetimes").write_html(
        output / "lifetimes.html", include_plotlyjs=True
    )
    px.imshow(
        correlation, zmin=-1, zmax=1, color_continuous_scale="RdBu_r", title="Training sensor correlation"
    ).write_html(output / "correlations.html", include_plotlyjs=True)
    examples = train[train.engine.isin(sorted(train.engine.unique())[:4])]
    chosen = [s for s in ("sensor_2", "sensor_7", "sensor_11") if s in informative]
    long = examples.melt(
        id_vars=["engine", "cycle"], value_vars=chosen, var_name="sensor", value_name="reading"
    )
    long["engine"] = long.engine.astype(str)
    px.line(
        long,
        x="cycle",
        y="reading",
        color="engine",
        facet_row="sensor",
        title="Sensor histories for four training engines",
    ).update_yaxes(matches=None).write_html(output / "sensor_trends.html", include_plotlyjs=True)
    write_json(
        output / "protocol.json",
        {
            "seed": args.seed,
            "subset": "FD001",
            "split": "training_only",
            "engines": sorted(train.engine.unique().tolist()),
        },
    )
    print(f"Exploration saved to {output.resolve()}", flush=True)
