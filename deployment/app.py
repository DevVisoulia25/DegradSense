"""Cloud entry point: frozen research evidence and prediction without training data."""

# ruff: noqa: E402
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import streamlit as st
import torch

from rul.research_dashboard import render_research

torch.set_num_threads(2)
st.set_page_config(page_title="DegradSense | Turbofan Prognostics", page_icon="⚙️", layout="wide")
st.title("DegradSense")
st.caption("Multisensor Turbofan Remaining Useful Life Prediction and Early Warning • NASA C-MAPSS")
render_research(Path(__file__).resolve().parent / "studies")
