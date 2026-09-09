"""
Streamlit demo UI for the exoplanet transit CNN project.

Lets you pick one of the cached/processed stars, see its cleaned light curve
(raw single-transit view AND phase-folded view), and get both trained
models' Planet / False Alarm prediction side by side -- this is the "does
phase folding help" question made visible on one star at a time, plus the
overall Experiment A vs B metrics from the actual smoke-test run.

Run:
    .\venv\Scripts\python.exe -m streamlit run app.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RAW_DATASET_PATH, FOLDED_DATASET_PATH, RESULTS_DIR, MODELS_DIR  # noqa: E402

st.set_page_config(page_title="Exoplanet Transit CNN", layout="wide")


@st.cache_resource
def load_models():
    import tensorflow as tf
    models = {}
    for variant in ["raw", "folded"]:
        path = MODELS_DIR / f"{variant}_best.keras"
        if path.exists():
            models[variant] = tf.keras.models.load_model(path)
    return models


@st.cache_data
def load_datasets():
    data = {}
    for variant, path in [("raw", RAW_DATASET_PATH), ("folded", FOLDED_DATASET_PATH)]:
        if path.exists():
            d = np.load(path, allow_pickle=True)
            data[variant] = {
                "X": d["X"], "y": d["y"], "kepid": d["kepid"],
                "koi_name": d["koi_name"], "split": d["split"],
            }
    return data


def load_metrics():
    out = {}
    for variant, folder in [("raw", "experiment_A_raw"), ("folded", "experiment_B_folded")]:
        path = RESULTS_DIR / folder / "metrics.json"
        if path.exists():
            with open(path) as f:
                out[variant] = json.load(f)
    return out


st.title("Exoplanet Transit Detection -- 1D CNN")
st.caption(
    "Kepler light curves -> cleaned -> phase-folded vs. raw -> resampled to 400 points "
    "-> 1D CNN -> Planet / False Alarm. This demo runs on the real (smoke-test scale) "
    "pipeline output -- no fabricated numbers."
)

models = load_models()
datasets = load_datasets()
metrics = load_metrics()

if not datasets:
    st.error("No processed dataset found yet. Run scripts 01-04 first.")
    st.stop()

# ---------------------------------------------------------------------------
# Section 1: overall Experiment A vs B results
# ---------------------------------------------------------------------------
st.header("Experiment A vs. B -- overall test-set results")
if metrics.get("raw") and metrics.get("folded"):
    cols = st.columns(2)
    for col, variant, label in zip(cols, ["raw", "folded"],
                                    ["A: Raw (no phase folding)", "B: Phase folded"]):
        m = metrics[variant]
        with col:
            st.subheader(label)
            st.metric("Accuracy", f"{m['accuracy']:.2f}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Precision", f"{m['precision']:.2f}")
            c2.metric("Recall", f"{m['recall']:.2f}")
            c3.metric("F1", f"{m['f1']:.2f}")
            st.caption(f"n_train={m['n_train']}  n_val={m['n_val']}  n_test={m['n_test']}")
    st.warning(
        "This run is a small smoke-test sample (tens of stars), used to prove the pipeline "
        "works end-to-end. Treat these specific numbers as a proof of concept, not a "
        "statistically reliable result -- that needs a much larger download."
    )
else:
    st.info("Train both variants (scripts/05_train_experiment.py --variant raw / folded) "
            "to see results here.")

comparison_png = RESULTS_DIR / "experiment_comparison.png"
if comparison_png.exists():
    st.image(str(comparison_png), caption="Experiment A vs B, all metrics")

# ---------------------------------------------------------------------------
# Section 2: pick one star, look at its light curve, get a live prediction
# ---------------------------------------------------------------------------
st.header("Inspect one star")

any_variant = "folded" if "folded" in datasets else "raw"
koi_names = datasets[any_variant]["koi_name"]
labels = datasets[any_variant]["y"]
splits = datasets[any_variant]["split"]

options = [f"{name}  (true label: {'Planet' if lab == 1 else 'False Alarm'}, split: {sp})"
           for name, lab, sp in zip(koi_names, labels, splits)]
idx = st.selectbox("Choose a KOI", range(len(options)), format_func=lambda i: options[i])

true_label = "Planet" if labels[idx] == 1 else "False Alarm"
st.write(f"**True label (from NASA KOI table):** {true_label}  |  **Split:** {splits[idx]}")

col1, col2 = st.columns(2)
for col, variant, label in zip([col1, col2], ["raw", "folded"],
                                ["Raw (single transit, unfolded)", "Phase-folded (global view)"]):
    with col:
        st.subheader(label)
        if variant in datasets:
            x = datasets[variant]["X"][idx]
            st.line_chart(pd.DataFrame({"flux": x}))
            if variant in models:
                prob = float(models[variant].predict(x[np.newaxis, :, np.newaxis],
                                                       verbose=0)[0, 0])
                pred_label = "Planet" if prob >= 0.5 else "False Alarm"
                correct = "correct" if pred_label == true_label else "incorrect"
                st.metric(f"Model prediction ({variant})", pred_label,
                          delta=f"{prob:.2f} planet-probability -- {correct}")
        else:
            st.info(f"No '{variant}' dataset found.")

st.header("Data-leakage guardrail (for reference)")
st.write(
    "The CNN never sees any NASA vetting column -- only the flux array. Columns like "
    "`koi_score`, `koi_pdisposition`, and `koi_fpflag_*` are dropped before "
    "`koi_clean_labels.csv` is even written; see `config.LEAKAGE_COLUMNS` and "
    "`scripts/02_clean_koi_table.py`."
)
