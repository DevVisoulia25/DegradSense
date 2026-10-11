# Deploy DegradSense

The cloud entrypoint serves the completed research evidence and frozen-model inference. It contains no training controls and requires no NASA data download at startup.

## Streamlit Community Cloud

1. Sign in at https://share.streamlit.io/ and connect the GitHub account that owns the repository.
2. Create an app from `DevVisoulia25/DegradSense` on branch `main`.
3. Set the entrypoint to `deployment/app.py`.
4. In Advanced settings, choose Python **3.12**. No app secrets are needed.
5. Optionally request the subdomain `degradsense` if available, then deploy.

Community Cloud uses `deployment/requirements.txt` beside this entrypoint. It pins the model-compatible runtime versions and CPU builds of PyTorch and XGBoost. The theme configuration stays in the repository-root `.streamlit/config.toml`, without a localhost binding. The local launcher explicitly binds to localhost.

## Included files

`studies/research_v2` contains the frozen FD001 LSTM and FD003 XGBoost, their fitted preprocessors, saved calibration and protocol settings, and research evidence. These are the original fitted models, not retrained deployment replacements. `model_manifest.json` records model-file SHA256 hashes for comparing the bundle with local study output.

Only repository-controlled model files are loaded. Uploaded inputs are sensor histories, not serialized models. Uploaded histories and predictions are processed in memory; the application does not write them to disk. Hosting-provider processing policies still apply.

The dashboard's remaining-life estimates and intervals retain the limitations documented in the [model card](../docs/MODEL_CARD.md). The model targets simulated C-MAPSS engines and capped RUL up to 125 cycles; arbitrary equipment with different sensor definitions is outside the evaluated scope.

## Local cloud-entrypoint check

Using the existing development environment, run from the repository root:

```powershell
.\.venv\Scripts\python.exe -m streamlit run deployment/app.py --server.address 127.0.0.1
```

This checks the packaged app independently of `artifacts/first_run`, `artifacts/research_v2`, and raw training data. A successful local check does not establish that the hosted Linux environment has passed its own startup and inference checks.
