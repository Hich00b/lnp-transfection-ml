"""SHAP interpretability analysis for the trained transfection model.

Generates publication-quality SHAP plots explaining which chemical features
(Morgan fingerprint bits + physicochemical descriptors) drive the predicted
transfection efficiency.  Run as::

    python src/evaluate.py
    python src/evaluate.py --model models/xgb_transfection.pkl --features data/processed/features.csv
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Optional, Sequence

import joblib
import numpy as np
import pandas as pd

# Use a headless backend so plots save cleanly on Colab / CI without a display.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import shap  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "models/xgb_transfection.pkl"
DEFAULT_FEATURES = "data/processed/features.csv"
DEFAULT_SPLIT = "models/split_indices.csv"
DEFAULT_OUT = "models/shap_summary_transfection.png"
DEFAULT_OUT_DESC = "models/shap_descriptors_transfection.png"
DEFAULT_TARGET = "Transfection"
DESCRIPTOR_COLS = ("MolWt", "NumRotatableBonds", "TPSA", "MolLogP")
MAX_DISPLAY = 25  # top features shown on the global beeswarm


# --------------------------------------------------------------------------- #
# Plot styling
# --------------------------------------------------------------------------- #
def style_matplotlib() -> None:
    """Apply publication-quality matplotlib defaults (large fonts, clean axes)."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 14,
            "axes.titlesize": 18,
            "axes.titleweight": "bold",
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "savefig.bbox": "tight",
        }
    )


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_model(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Trained model not found: {path}")
    model = joblib.load(path)
    logger.info("Loaded model from %s", path)
    return model


def load_test_features(
    features_path: str, split_path: str
) -> tuple[pd.DataFrame, list[int]]:
    """Return the held-out test feature matrix (+ the test row indices)."""
    if not os.path.isfile(features_path):
        raise FileNotFoundError(f"Features file not found: {features_path}")
    X = pd.read_csv(features_path).select_dtypes(include=[np.number])
    test_idx: list[int] = list(range(len(X)))  # default: evaluate on everything
    if os.path.isfile(split_path):
        split = pd.read_csv(split_path)
        if "test_idx" in split.columns:
            vals = split["test_idx"].dropna().astype(int).tolist()
            if vals:
                test_idx = vals
    present = [i for i in test_idx if 0 <= i < len(X)]
    if len(present) != len(test_idx):
        logger.warning("Some test indices out of range; clamped to %d rows.", len(present))
    return X.iloc[present], present


# --------------------------------------------------------------------------- #
# SHAP plotting
# --------------------------------------------------------------------------- #
def _beeswarm(shap_values, X, out_path, target_label, max_display=MAX_DISPLAY):
    """Standard SHAP beeswarm summary plot saved at 300 DPI."""
    plt.figure()
    shap.summary_plot(
        shap_values,
        X,
        max_display=max_display,
        show=False,
    )
    ax = plt.gca()
    ax.set_title(
        f"SHAP summary: features driving predicted {target_label}", pad=14
    )
    ax.set_xlabel(f"SHAP value (impact on predicted {target_label})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved SHAP beeswarm -> %s", out_path)
    print(f"Saved SHAP summary plot -> {out_path}")


def _descriptor_bar(shap_values, X, out_path, target_label):
    """Focused bar plot of mean |SHAP| for the four interpretable descriptors."""
    present = [c for c in DESCRIPTOR_COLS if c in X.columns]
    if not present:
        logger.info("No descriptor columns present; skipping descriptor plot.")
        return
    cols = list(X.columns)
    positions = [cols.index(c) for c in present]
    means = np.abs(shap_values[:, positions]).mean(axis=0)
    order = np.argsort(means)[::-1]
    labels = [present[i] for i in order]
    values = means[order]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(range(len(labels))[::-1], values, color="#3b6fb6", edgecolor="black")
    ax.set_yticks(range(len(labels))[::-1])
    ax.set_yticklabels(labels)
    ax.set_xlabel("mean |SHAP value| (importance)")
    ax.set_title(f"{target_label}: descriptor importance")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved descriptor importance -> %s", out_path)
    print(f"Saved descriptor importance plot -> {out_path}")


def run_shap_analysis(
    model_path: str = DEFAULT_MODEL,
    features_path: str = DEFAULT_FEATURES,
    split_path: str = DEFAULT_SPLIT,
    out_path: str = DEFAULT_OUT,
    out_desc_path: str = DEFAULT_OUT_DESC,
    target: str = DEFAULT_TARGET,
) -> None:
    """Compute SHAP values with a TreeExplainer and save the summary plots."""
    style_matplotlib()
    model = load_model(model_path)
    X_test, _ = load_test_features(features_path, split_path)
    if len(X_test) == 0:
        raise RuntimeError("No test samples available for SHAP analysis.")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    logger.info("SHAP values shape: %s", np.asarray(shap_values).shape)

    _beeswarm(shap_values, X_test, out_path, target_label=target)
    _descriptor_bar(shap_values, X_test, out_desc_path, target_label=target)

    # Persist raw SHAP values + the test matrix for downstream analysis.
    np_path = out_path.replace(".png", "_values.npy")
    np.save(np_path, np.asarray(shap_values))
    logger.info("Saved raw SHAP values -> %s", np_path)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SHAP interpretability analysis for the transfection model."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--features", default=DEFAULT_FEATURES)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--out-desc", default=DEFAULT_OUT_DESC)
    parser.add_argument("--target", default=DEFAULT_TARGET,
                        help="Target label used in plot titles (default: Transfection).")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = _build_parser().parse_args(argv)
    run_shap_analysis(
        model_path=args.model,
        features_path=args.features,
        split_path=args.split,
        out_path=args.out,
        out_desc_path=args.out_desc,
        target=args.target,
    )


if __name__ == "__main__":
    main()
