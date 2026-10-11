import argparse

from rul.experiment import evaluate_test, explain, fusion_study, train
from rul.explore import explore
from rul.study import study
from rul.inference import predict_file


def main():
    parser = argparse.ArgumentParser(description="FD001 sensor fusion and RUL experiments")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="Train and validate without accessing official test labels")
    training.add_argument(
        "--models",
        nargs="+",
        choices=["mean", "rf", "xgb", "svr", "lstm", "gru"],
        default=["mean", "rf", "xgb", "lstm", "gru"],
    )
    training.add_argument("--output")
    training.add_argument("--epochs", type=int, default=20)
    training.add_argument("--trees", type=int, default=120)
    training.add_argument("--window", type=int, default=30)
    training.add_argument("--stride", type=int, default=2)
    training.add_argument("--cap", type=float, default=125)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--validation-fraction", type=float, default=0.2)
    training.add_argument("--horizon", type=int, default=30)
    training.add_argument("--persistence", type=int, default=3)
    training.set_defaults(func=train)
    evaluation = commands.add_parser("evaluate-test", help="Evaluate validation-selected model once")
    evaluation.add_argument("--run", required=True)
    evaluation.set_defaults(func=evaluate_test)
    fusion = commands.add_parser("fusion", help="Compare single-sensor and removed-sensor RF models")
    fusion.add_argument("--run", required=True)
    fusion.add_argument("--trees", type=int, default=80)
    fusion.set_defaults(func=fusion_study)
    explanation = commands.add_parser("explain", help="SHAP on held-out validation observations")
    explanation.add_argument("--run", required=True)
    explanation.add_argument("--model", choices=["rf", "xgb"], default="xgb")
    explanation.set_defaults(func=explain)
    exploration = commands.add_parser("explore", help="Training-only sensor and lifetime exploration")
    exploration.add_argument("--output", default="artifacts/eda")
    exploration.add_argument("--seed", type=int, default=42)
    exploration.set_defaults(func=explore)
    research = commands.add_parser(
        "study", help="Prespecified multi-seed study, calibration and final evaluation"
    )
    research.add_argument("--output", default="artifacts/research_v2")
    research.add_argument("--seeds", nargs="+", type=int, default=[7, 21, 42, 84, 123])
    research.add_argument("--subsets", nargs="+", choices=["FD001", "FD003"], default=["FD001", "FD003"])
    research.add_argument("--epochs", type=int, default=16)
    research.add_argument("--trees", type=int, default=100)
    research.set_defaults(func=study)
    inference = commands.add_parser(
        "predict", help="Predict final RUL and interval from unlabeled NASA-format histories"
    )
    inference.add_argument("--study", default="artifacts/research_v2")
    inference.add_argument("--subset", choices=["FD001", "FD003"], default="FD001")
    inference.add_argument("--input", required=True)
    inference.add_argument("--output")
    inference.set_defaults(func=predict_file)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
