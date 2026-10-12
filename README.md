# DegradSense

Multisensor Turbofan Remaining Useful Life Prediction and Early Warning

DegradSense estimates remaining operating cycles from simulated NASA C-MAPSS sensor histories. The current college presentation compares **Linear Regression, Decision Tree, XGBoost, LSTM, and GRU**. Linear Regression and Decision Tree replace the earlier Random Forest and mean baseline. Historical records remain archived; old results are never relabeled as results for a different model.

## Run locally

Use PowerShell from the project folder:

```powershell
.\launch_dashboard.ps1
```

Open `http://127.0.0.1:8501/` without trailing asterisks. Keep the terminal open. If the script is blocked:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

The dashboard has **Research evidence** for repeated experiments and **Single-split experiment** for selecting individual models, replaying predictions, examining linear coefficients and tree explanations, and comparing sensor combinations. Use `linear` for Linear Regression and `dt` for Decision Tree.

## Install and reproduce

Python 3.12 is recommended. On a fresh Windows checkout:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe scripts/download_data.py
.\.venv\Scripts\python.exe -m rul train --output artifacts/college_run --models linear dt xgb lstm gru --epochs 16 --trees 100
.\.venv\Scripts\python.exe -m rul fusion --run artifacts/college_run
.\.venv\Scripts\python.exe -m rul explain --run artifacts/college_run --model dt
.\.venv\Scripts\python.exe -m rul study --output artifacts/research_v3_college --epochs 16 --trees 100
```

Existing experiment folders cannot be overwritten. Choose a new output folder if a completed run already exists. `--trees` controls XGBoost's estimator count; the Decision Tree uses fixed depth 14 and a minimum of three samples per leaf.

## Models and inputs

- **Linear Regression:** standardized window statistics followed by ordinary least squares. This is the simple linear comparison model, not Ridge or an ensemble.
- **Decision Tree:** one regression tree with maximum depth 14 and minimum leaf size 3; the seed fixes reproducibility.
- **XGBoost:** boosted trees retained as an advanced comparison.
- **LSTM and GRU:** recurrent networks retained as advanced sequence comparisons. SVR remains an optional CLI model.

All models use the same training-engine preprocessing and causal 30-cycle sensor windows. Classical regressors receive mean, standard deviation, minimum, maximum, latest reading, and slope per sensor. These feature means are input statistics, not a mean-predictor model. Models never receive engine IDs, cycle IDs, future readings, or labels as predictors. RUL targets and predictions are capped at 125 cycles.

## Evaluation

The research revision covers FD001 and FD003 with 60 development, 20 calibration, and 20 assessment engines per dataset. Seven configurations are compared on five development splits: the five model families, plus Decision Tree with latest readings only and Decision Tree with one sensor selected using training data only. Each development split has 45 fitting and 15 validation engines.

Scaling and sensor selection use fitting engines only. Model selection uses mean development endpoint RMSE. Calibration estimates a nominal 90% endpoint interval; assessment replays lifetimes using the fixed 30-cycle, three-consecutive-cycle warning rule. Report RMSE, MAE, NASA score, uncertainty width/coverage, missed engines, false-positive cycle rate, and warning lead time together.

The curriculum revision is retrospective: outcomes for both datasets were previously inspected in version two. Its official test evaluation documents this release and is not a fresh untouched benchmark. Identical XGBoost/LSTM/GRU development fits can be reused with `--reuse-development artifacts/research_v2`; the command checks settings, cohorts, source hashes, endpoint sampling, and per-seed preprocessing before reuse. New candidates are actually fitted.

See [current results](docs/RESULTS.md), [model card](docs/MODEL_CARD.md), and [interview guide](docs/INTERVIEW_GUIDE.md). [Version-two documentation](docs/history/version2/) and the original evidence retain the earlier comparisons for provenance. The current evidence is in `docs/evidence/research_v3_college`.

## Prediction and deployment package

```powershell
.\.venv\Scripts\python.exe -m rul predict --study artifacts/research_v3_college --subset FD001 --input data/raw/test_FD001.txt
```

The inference interface accepts unlabeled sensor histories and reports capped RUL with endpoint intervals. Arbitrary real equipment is outside the evaluated scope. The prepared `deployment/app.py` uses bundled frozen models and results without raw-data downloads or retraining. **No hosted deployment is active**; the project is demonstrated locally. See [deployment preparation notes](deployment/README.md).

## Data and verification

Dataset: A. Saxena and K. Goebel (2008), Turbofan Engine Degradation Simulation Data Set, NASA Ames Prognostics Data Repository. The download script uses the archive linked from NASA's [dataset repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) and records file hashes. Raw datasets and local environments remain excluded from Git.

```powershell
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

GitHub Actions verifies the complete project on Windows and packaged inference on Linux. The simulation results are educational research evidence, not industrial safety certification.
