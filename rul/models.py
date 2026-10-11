"""Classical window-statistic regressors and small recurrent regressors."""

import copy
import random

import numpy as np
import torch
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def classical_model(name, seed=42, trees=120):
    if name == "mean":
        return DummyRegressor(strategy="mean")
    if name == "rf":
        return RandomForestRegressor(
            n_estimators=trees, max_depth=14, min_samples_leaf=3, n_jobs=4, random_state=seed
        )
    if name == "xgb":
        return XGBRegressor(
            n_estimators=trees * 2,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            n_jobs=4,
            random_state=seed,
            objective="reg:squarederror",
        )
    if name == "svr":
        return make_pipeline(StandardScaler(), SVR(C=50, epsilon=2))
    raise ValueError(f"Unknown classical model {name}")


class SequenceRegressor(nn.Module):
    def __init__(self, sensors, kind="lstm", hidden=48):
        super().__init__()
        recurrent = {"lstm": nn.LSTM, "gru": nn.GRU}[kind]
        self.recurrent = recurrent(sensors, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 24), nn.ReLU(), nn.Linear(24, 1))

    def forward(self, x):
        sequence, _ = self.recurrent(x)
        return self.head(sequence[:, -1]).squeeze(-1)


def sequence_predict(model, x, cap, batch_size=512):
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        values = [
            model(torch.from_numpy(x[i : i + batch_size]).to(device)).cpu().numpy()
            for i in range(0, len(x), batch_size)
        ]
    return np.clip(np.concatenate(values) * cap, 0, cap)


def fit_sequence(kind, train, validation, cap=125, epochs=20, seed=42, hidden=48):
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SequenceRegressor(train.x.shape[-1], kind, hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train.x), torch.from_numpy(train.y / cap)),
        batch_size=128,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    best_loss, best_state, stale, history = float("inf"), None, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for x, y in loader:
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(x.to(device)), y.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(x)
        if validation is None:
            history.append({"epoch": epoch, "train_mse_cycles": total / len(train.x) * cap**2})
            print(f"{kind}: frozen refit epoch {epoch}/{epochs}", flush=True)
            continue
        pred = sequence_predict(model, validation.x, cap)
        val_loss = float(np.mean((pred - validation.y) ** 2))
        history.append(
            {
                "epoch": epoch,
                "train_mse_cycles": total / len(train.x) * cap**2,
                "validation_rmse": float(np.sqrt(val_loss)),
            }
        )
        print(f"{kind}: epoch {epoch}/{epochs}, validation RMSE {np.sqrt(val_loss):.2f}", flush=True)
        if val_loss < best_loss:
            best_loss, best_state, stale = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= 5:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.cpu(), history
