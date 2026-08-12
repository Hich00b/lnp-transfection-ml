"""Fair multi-algorithm benchmark for LNP transfection-efficiency prediction.

Trains and evaluates XGBoost (with early stopping fix), RandomForestRegressor,
Ridge, SVR, and MLPRegressor on the identical feature matrix, identical CV folds
(reusing the same KFold(shuffle=True, random_state=42) split indices), and
identical metrics (R², RMSE, MAE).

Outputs a single markdown table suitable for pasting directly into a paper's
results table.
"""

from __future__ import annotations

import logging
import os
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)

# Paths to processed data
FEATURES_PATH = "data/processed/features.csv"
TARGETS_PATH = "data/processed/targets.csv"
TARGET_COL = "Transfection"

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


def train_model_with_early_stopping(
    model: XGBRegressor, X_train: np.ndarray, y_train: np.ndarray
) -> XGBRegressor:
    """Train XGBoost model with early stopping on a validation split."""
    from sklearn.model_selection import train_test_split

    # Carve out validation set for early stopping
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, shuffle=True
    )

    # Set early stopping parameter
    model.set_params(early_stopping_rounds=EARLY_STOPPING_ROUNDS)

    # Fit with early stopping
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    return model


def main() -> None:
    """Run the benchmark and print results."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    # Load data
    X, y = load_data()

    # Set up CV folds (same for all models)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Store results for each model
    results = {name: [] for name in MODELS.keys()}

    # Evaluate each model using the same CV splits
    for fold, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
        logger.info("Processing fold %d/5", fold)
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        for name, model in MODELS.items():
            if name == "XGBoost":
                # Special handling for XGBoost with early stopping
                # Create a fresh copy of the model with the same parameters
                model_copy = XGBRegressor(**model.get_params())
                fitted_model = train_model_with_early_stopping(model_copy, X_train, y_train)
                y_pred = fitted_model.predict(X_test)
            else:
                # Standard fit/predict for other models
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

            # Calculate metrics
            metrics = regression_metrics(y_test, y_pred)
            results[name].append(metrics)
            logger.info(
                "  %s: %s",
                name,
                fmt_metrics(metrics),
            )

    # Aggregate results across folds
    print("\n## Algorithm Performance Comparison\n")
    print("| Algorithm | Mean R² ± Std | Mean RMSE ± Std | Mean MAE ± Std |")
    print("|-----------|---------------|-----------------|----------------|")

    for name in MODELS.keys():
        fold_results = results[name]
        mean_r2 = np.mean([r["R2"] for r in fold_results])
        std_r2 = np.std([r["R2"] for r in fold_results])
        mean_rmse = np.mean([r["RMSE"] for r in fold_results])
        std_rmse = np.std([r["RMSE"] for r in fold_results])
        mean_mae = np.mean([r["MAE"] for r in fold_results])
        std_mae = np.std([r["MAE"] for r in fold_results])

        print(
            f"| {name:<9} | {mean_r2:.4f} ± {std_r2:.4f} | {mean_rmse:.4f} ± {std_rmse:.4f} | {mean_mae:.4f} ± {std_mae:.4f} |"
        )

    logger.info("Benchmark completed")


if __name__ == "__main__":
    main()