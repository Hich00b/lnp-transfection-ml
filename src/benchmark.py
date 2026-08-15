"""Fair multi-algorithm benchmark for LNP transfection-efficiency prediction.

Trains and evaluates XGBoost (with early stopping fix), RandomForestRegressor,
Ridge, SVR, and MLPRegressor on the identical feature matrix, identical buffered
leave-one-group-out (LOCO) splits, and identical metrics (R², RMSE, MAE).

Outputs a single markdown table suitable for pasting directly into a paper's
results table.
"""

from __future__ import annotations

import logging
import os
from typing import Tuple, List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.ML.Cluster import Butina
from train import buffered_leave_one_group_out, fit_model
import pandas as pd

logger = logging.getLogger(__name__)

# Paths to processed data
FEATURES_PATH = "data/processed/features.csv"
TARGETS_PATH = "data/processed/targets.csv"
TARGET_COL = "Transfection"

# Default parameters for buffered splitting
DEFAULT_DISTANCE_CUTOFF = 0.1
DEFAULT_MIN_CLUSTER_SIZE = 10

# Model configurations with reasonable default hyperparameters
# These are chosen as sensible baselines, not optimized
MODELS = {
    "XGBoost": XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        # Note: early stopping is handled in the fitting loop below
    ),
    "RandomForest": RandomForestRegressor(
        n_estimators=500,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    ),
    "Ridge": Ridge(
        alpha=1.0,
        random_state=42,
    ),
    "SVR": SVR(
        kernel='rbf',
        C=1.0,
        epsilon=0.1,
    ),
    "MLP": MLPRegressor(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        learning_rate='adaptive',
        max_iter=500,
        random_state=42,
    ),
}

# Early stopping rounds for XGBoost (matched to train.py)
EARLY_STOPPING_ROUNDS = 50


def load_data() -> Tuple[np.ndarray, np.ndarray]:
    """Load feature matrix and target vector from processed CSV files."""
    if not os.path.isfile(FEATURES_PATH):
        raise FileNotFoundError(f"Features file missing: {FEATURES_PATH}")
    if not os.path.isfile(TARGETS_PATH):
        raise FileNotFoundError(f"Targets file missing: {TARGETS_PATH}")

    # Load features (all columns are numeric)
    X_df = pd.read_csv(FEATURES_PATH)
    X = X_df.values.astype(np.float32)

    # Load target
    y_df = pd.read_csv(TARGETS_PATH)
    if TARGET_COL not in y_df.columns:
        raise ValueError(
            f"Target column '{TARGET_COL}' not in {TARGETS_PATH}. "
            f"Available: {list(y_df.columns)}"
        )
    y = y_df[TARGET_COL].values.astype(np.float32)

    logger.info("Loaded X=%s, y=%d samples", X.shape, len(y))
    return X, y


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Return R^2, RMSE and MAE."""
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def fmt_metrics(metrics: dict[str, float]) -> str:
    """Format metrics as a string."""
    return f"R²={metrics['R2']:.4f}  RMSE={metrics['RMSE']:.4f}  MAE={metrics['MAE']:.4f}"



def main() -> None:
    """Run the benchmark and print results."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    # Load data
    X, y = load_data()

    # Load metadata for SMILES (needed for splitting)
    meta_path = "data/processed/metadata.csv"
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Metadata file missing: {meta_path}")
    meta = pd.read_csv(meta_path)
    if "SMILES" not in meta.columns:
        raise ValueError("Metadata must contain a 'SMILES' column for splitting.")
    smiles = meta["SMILES"].tolist()

    # Compute the buffered leave-one-group-out splits once
    logger.info("Computing buffered leave-one-group-out splits...")
    splits = buffered_leave_one_group_out(
        smiles,
        fp_radius=2,
        fp_bits=2048,
        distance_cutoff=DEFAULT_DISTANCE_CUTOFF,
        min_cluster_size=DEFAULT_MIN_CLUSTER_SIZE,
    )
    logger.info(f"Computed {len(splits)} splits for buffered LOCO.")

    # Store results for each model: we will collect per-fold metrics and out-of-fold predictions
    per_fold_metrics = {name: [] for name in MODELS.keys()}
    # For pooled metrics: collect all out-of-fold predictions and true values
    all_oof_preds = {name: [] for name in MODELS.keys()}
    all_oof_true = {name: [] for name in MODELS.keys()}

    # Evaluate each model using the same splits
    for fold, (train_indices, test_indices) in enumerate(splits, start=1):
        logger.info(f"Processing fold {fold}/{len(splits)}")
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        for name, model in MODELS.items():
            if name == "XGBoost":
                # Special handling for XGBoost with early stopping
                # Create a fresh copy of the model with the same parameters
                model_copy = XGBRegressor(**model.get_params())
                # Convert numpy arrays to pandas DataFrame/Series for fit_model
                X_train_df = pd.DataFrame(X_train) if isinstance(X_train, np.ndarray) else X_train
                y_train_series = pd.Series(y_train) if isinstance(y_train, np.ndarray) else y_train
                fitted_model = fit_model(model_copy, X_train_df, y_train_series)
                y_pred = fitted_model.predict(X_test)
            else:
                # Standard fit/predict for other models
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

            # Calculate metrics for this fold
            m = regression_metrics(y_test, y_pred)
            per_fold_metrics[name].append(m)

            # Collect out-of-fold predictions and true values for pooled metrics
            all_oof_preds[name].extend(y_pred)
            all_oof_true[name].extend(y_test)

            logger.info(
                "  %s: %s",
                name,
                fmt_metrics(m),
            )

    # Aggregate results across folds: compute pooled metrics (primary) and per-fold mean +/- std (secondary)
    print("\n## Algorithm Performance Comparison (Buffered LOCO)\n")
    print("### Pooled Metrics (PRIMARY)")
    print("| Algorithm | Pooled R² | Pooled RMSE | Pooled MAE |")
    print("|-----------|-----------|-------------|------------|")
    pooled_results = {}
    for name in MODELS.keys():
        # Compute pooled metrics from all out-of-fold predictions and true values
        pooled_m = regression_metrics(all_oof_true[name], all_oof_preds[name])
        pooled_results[name] = pooled_m
        print(
            f"| {name:<9} | {pooled_m['R2']:.4f} | {pooled_m['RMSE']:.4f} | {pooled_m['MAE']:.4f} |"
        )

    print("\n### Per-Fold Mean ± Std (SECONDARY, unstable for small groups)")
    print("| Algorithm | Mean R² ± Std | Mean RMSE ± Std | Mean MAE ± Std |")
    print("|-----------|---------------|-----------------|----------------|")
    for name in MODELS.keys():
        fold_results = per_fold_metrics[name]
        mean_r2 = np.mean([m["R2"] for m in fold_results])
        std_r2 = np.std([m["R2"] for m in fold_results])
        mean_rmse = np.mean([m["RMSE"] for m in fold_results])
        std_rmse = np.std([m["RMSE"] for m in fold_results])
        mean_mae = np.mean([m["MAE"] for m in fold_results])
        std_mae = np.std([m["MAE"] for m in fold_results])

        print(
            f"| {name:<9} | {mean_r2:.4f} ± {std_r2:.4f} | {mean_rmse:.4f} ± {std_rmse:.4f} | {mean_mae:.4f} ± {std_mae:.4f} |"
        )

    logger.info("Benchmark completed")


if __name__ == "__main__":
    main()