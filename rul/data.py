"""Validated C-MAPSS loading and causal, engine-local window construction."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SENSORS = [f"sensor_{i}" for i in range(1, 22)]
COLUMNS = ["engine", "cycle", "setting_1", "setting_2", "setting_3", *SENSORS]
ROOT = Path(__file__).resolve().parents[1]


def load_trajectories(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing {path.name}. Run scripts/download_data.py first.")
    frame = pd.read_csv(path, sep=r"\s+", header=None)
    if frame.shape[1] != len(COLUMNS):
        raise ValueError(f"Expected 26 columns; got {frame.shape[1]}")
    frame.columns = COLUMNS
    return validate_trajectories(frame)


def validate_trajectories(frame):
    missing = set(COLUMNS) - set(frame.columns)
    if missing or frame.empty:
        raise ValueError(f"Require nonempty sensor histories with all 26 columns. Missing: {sorted(missing)}")
    frame = frame[COLUMNS].copy()
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("Sensor data contains nonfinite values")
    for column in ("engine", "cycle"):
        values = frame[column].to_numpy()
        if (values < 1).any() or (values != np.floor(values)).any():
            raise ValueError(f"{column} must contain positive integers")
        frame[column] = frame[column].astype(int)
    if frame.duplicated(["engine", "cycle"]).any():
        raise ValueError("Duplicate engine/cycle rows")
    frame = frame.sort_values(["engine", "cycle"]).reset_index(drop=True)
    for _, engine in frame.groupby("engine", sort=True):
        if not np.array_equal(engine.cycle, np.arange(1, len(engine) + 1)):
            raise ValueError("Expected consecutive cycles beginning at 1")
    return frame


def label_training(frame, cap=125):
    result = frame.copy()
    result["rul_raw"] = result.groupby("engine")["cycle"].transform("max") - result.cycle
    result["rul"] = result.rul_raw.clip(upper=cap)
    return result


def label_test(frame, target_path, cap=125):
    targets = np.loadtxt(target_path, ndmin=1)
    ids = np.sort(frame.engine.unique())
    if not np.array_equal(ids, np.arange(1, len(ids) + 1)):
        raise ValueError("Official test engine IDs must be consecutive for RUL-file alignment")
    if targets.shape != (len(ids),) or not np.isfinite(targets).all() or (targets < 0).any():
        raise ValueError("Invalid test RUL vector")
    result = frame.copy()
    offsets = dict(zip(ids, targets))
    result["rul_raw"] = (
        result.groupby("engine")["cycle"].transform("max") - result.cycle + result.engine.map(offsets)
    )
    result["rul"] = result.rul_raw.clip(upper=cap)
    return result


def split_engines(frame, fraction=0.2, seed=42):
    ids = np.sort(frame.engine.unique())
    train_ids, val_ids = train_test_split(ids, test_size=fraction, random_state=seed)
    return (frame[frame.engine.isin(train_ids)].copy(), frame[frame.engine.isin(val_ids)].copy())


@dataclass
class Preprocessor:
    sensors: list
    scaler: StandardScaler

    @classmethod
    def fit(cls, train):
        # Select only with training data. FD001 has one operating regime.
        sensors = [s for s in SENSORS if train[s].std() > 1e-8]
        if not sensors:
            raise ValueError("No varying sensors")
        return cls(sensors, StandardScaler().fit(train[sensors]))

    def transform(self, frame):
        result = frame.copy()
        result[self.sensors] = self.scaler.transform(frame[self.sensors])
        return result


@dataclass
class Windows:
    x: np.ndarray
    y: np.ndarray
    metadata: pd.DataFrame


def make_windows(frame, sensors, length=30, stride=1, last_only=False):
    if length < 2 or stride < 1:
        raise ValueError("Window length must be >=2 and stride >=1")
    xs, ys, rows = [], [], []
    for engine_id, group in frame.groupby("engine", sort=True):
        group = group.sort_values("cycle")
        values = group[sensors].to_numpy(dtype=np.float32)
        if last_only:
            ends = [len(group) - 1]
        else:
            ends = list(range(length - 1, len(group), stride))
            if len(group) >= length and ends[-1] != len(group) - 1:
                ends.append(len(group) - 1)
        for end in ends:
            window = values[max(0, end - length + 1) : end + 1]
            if len(window) < length:
                # Official test engines shorter than a window: repeat first observation.
                window = np.pad(window, ((length - len(window), 0), (0, 0)), mode="edge")
            row = group.iloc[end]
            xs.append(window)
            ys.append(row.rul)
            rows.append(
                {
                    "engine": int(engine_id),
                    "cycle": int(row.cycle),
                    "rul_raw": float(row.rul_raw),
                    "rul": float(row.rul),
                }
            )
    if not xs:
        raise ValueError("No windows; reduce window length or check data")
    return Windows(np.stack(xs), np.array(ys, dtype=np.float32), pd.DataFrame(rows))


def window_features(x):
    """Per sensor: mean, std, minimum, maximum, latest value, least-squares slope."""
    time = np.arange(x.shape[1], dtype=np.float32)
    time -= time.mean()
    slope = np.einsum("nts,t->ns", x, time) / np.square(time).sum()
    return np.concatenate([x.mean(1), x.std(1), x.min(1), x.max(1), x[:, -1], slope], axis=1)


def feature_names(sensors):
    return [f"{s}__{stat}" for stat in ("mean", "std", "min", "max", "latest", "slope") for s in sensors]
