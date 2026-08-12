"""Train an XGBoost regressor for LNP transfection-efficiency prediction.

Generalisation to unseen chemical space is tested with a **scaffold-based
train/test split**: molecules are grouped by their Murcko scaffold and whole
scaffolds are held out in the test set.  An optional K-fold cross-validation
mode is provided for reporting uncertainty across folds.

The default target is ``Transfection`` (AGILE dataset format); any other
numeric column in the processed targets file can be selected with
``--target``.

Run as::

    python src/train.py                       # single scaffold split
    python src/train.py --cv 5                 # 5-fold CV, then refit on all
    python src/train.py --cv 5 --target Transfection
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)

DEFAULT_FEATURES = "data/processed/features.csv"
DEFAULT_TARGETS = "data/processed/targets.csv"
DEFAULT_METADATA = "data/processed/metadata.csv"
DEFAULT_MODELS_DIR = "models"
DEFAULT_MODEL_NAME = "xgb_transfection.pkl"
DEFAULT_TARGET = "Transfection"
SMILES_COL = "SMILES"

# XGBoost hyper-parameters -- modest depth + learning rate for a small dataset.
XGB_PARAMS = dict(
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
)
EARLY_STOPPING_ROUNDS = 50
INTERNAL_VAL_FRAC = 0.1  # carve an eval set from the training fold for early stop


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_dataset(
    features_path: str,
    targets_path: str,
    metadata_path: str,
    target: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load the processed feature matrix, target vector, and SMILES metadata."""
    if not os.path.isfile(features_path):
        raise FileNotFoundError(f"Features file missing: {features_path}")
    if not os.path.isfile(targets_path):
        raise FileNotFoundError(f"Targets file missing: {targets_path}")

    X = pd.read_csv(features_path)
    y_df = pd.read_csv(targets_path)
    if target not in y_df.columns:
        raise ValueError(
            f"Target column '{target}' not in {targets_path}. "
            f"Available: {list(y_df.columns)}"
        )
    y = y_df[target]
    meta = (
        pd.read_csv(metadata_path)
        if os.path.isfile(metadata_path)
        else pd.DataFrame({SMILES_COL: []})
    )

    # Keep only numeric feature columns (defensive -- metadata must not leak in).
    X = X.select_dtypes(include=[np.number])
    logger.info("Loaded X=%s, y=%d samples, target=%s", X.shape, len(y), target)
    return X, y, meta


# --------------------------------------------------------------------------- #
# Scaffold splitting
# --------------------------------------------------------------------------- #
def _murcko_scaffold(smiles: str) -> str:
    """Return the canonical Murcko scaffold SMILES, or "" if unobtainable."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception:  # noqa: BLE001 -- edge-case scaffolds can fail
        return ""


def scaffold_split(
    smiles: Sequence[str],
    frac_train: float = 0.8,
    random_state: int = 42,
) -> tuple[list[int], list[int]]:
    """Group molecules by Murcko scaffold, then allocate whole scaffold groups.

    The largest scaffold groups go to the training set until the desired
    train fraction is reached; all remaining (smaller / singleton) scaffold
    groups form the test set.  This guarantees the test set contains scaffolds
    absent from training, probing generalisation to new chemical space.
    """
    rng = np.random.RandomState(random_state)
    scaffolds: dict[str, list[int]] = {}
    for i, smi in enumerate(smiles):
        scaffolds.setdefault(_murcko_scaffold(smi), []).append(i)

    # Shuffle within and across scaffold groups for reproducibility.
    scaffold_sets = [sorted(s) for s in scaffolds.values()]
    rng.shuffle(scaffold_sets)
    scaffold_sets.sort(key=len, reverse=True)

    n = len(smiles)
    train_cutoff = int(frac_train * n)
    train_idx, test_idx = [], []
    for sset in scaffold_sets:
        if len(train_idx) + len(sset) > train_cutoff:
            test_idx.extend(sset)
        else:
            train_idx.extend(sset)
    if not test_idx:  # degenerate single-scaffold input -> hold out last group
        test_idx = train_idx[train_cutoff:]
        train_idx = train_idx[:train_cutoff]
    logger.info("Scaffold split: %d train / %d test", len(train_idx), len(test_idx))
    return train_idx, test_idx


# --------------------------------------------------------------------------- #
# Model helpers
# --------------------------------------------------------------------------- #
def make_model() -> XGBRegressor:
    """Construct the XGBoost regressor with the project defaults."""
    return XGBRegressor(**XGB_PARAMS)


def fit_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> XGBRegressor:
    """Fit the model, using early stopping on an internal validation slice."""
    n_train = len(X_train)
    model = make_model()
    if n_train < 50:
        logger.info("Small training set (%d); fitting without early stopping.", n_train)
        model.fit(X_train, y_train, verbose=False)
        return model

    n_val = max(1, int(np.ceil(INTERNAL_VAL_FRAC * n_train)))
    X_tr, X_val = X_train.iloc[:-n_val], X_train.iloc[-n_val:]
    y_tr, y_val = y_train.iloc[:-n_val], y_train.iloc[-n_val:]
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    if getattr(model, "best_iteration", None) is not None:
        logger.info(
            "Early stopping at iteration %d (best=%d).",
            getattr(model, "best_iteration", 0) + 1,
            getattr(model, "best_iteration", 0) + 1,
        )
    # Refit on the full training fold with the chosen iteration count.
    final = make_model()
    final.set_params(
        n_estimators=getattr(model, "best_iteration", XGB_PARAMS["n_estimators"]) + 1
    )
    final.fit(X_train, y_train, verbose=False)
    return final


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Return R^2, RMSE and MAE."""
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def _fmt(metrics: dict[str, float]) -> str:
    return (
        f"R^2={metrics['R2']:.4f}  RMSE={metrics['RMSE']:.4f}  "
        f"MAE={metrics['MAE']:.4f}"
    )


# --------------------------------------------------------------------------- #
# Training routines
# --------------------------------------------------------------------------- #
def train_scaffold_split(X, y, meta, models_dir, model_name, target):
    """Single scaffold split: train, evaluate on held-out scaffolds, save model."""
    smiles = meta[SMILES_COL].tolist() if SMILES_COL in meta.columns else []
    if len(smiles) != len(X):
        logger.warning(
            "Metadata length mismatch (%d vs %d); falling back to random split.",
            len(smiles), len(X),
        )
        rng = np.random.RandomState(42)
        idx = rng.permutation(len(X))
        cut = int(0.8 * len(X))
        train_idx, test_idx = idx[:cut].tolist(), idx[cut:].tolist()
    else:
        train_idx, test_idx = scaffold_split(smiles)

    model = fit_model(X.iloc[train_idx], y.iloc[train_idx])
    preds = model.predict(X.iloc[test_idx])
    metrics = regression_metrics(y.iloc[test_idx], preds)
    print(f"[Scaffold split] {target} test metrics: " + _fmt(metrics))

    _save_model(model, models_dir, model_name)
    _save_split(train_idx, test_idx, models_dir)
    return model, metrics


def train_kfold(X, y, meta, k, models_dir, model_name, target):
    """K-fold CV reporting mean +/- std, then refit a final model on all data."""
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_metrics: list[dict[str, float]] = []
    for fold, (tr_idx, te_idx) in enumerate(kf.split(X), start=1):
        model = fit_model(X.iloc[tr_idx], y.iloc[tr_idx])
        preds = model.predict(X.iloc[te_idx])
        m = regression_metrics(y.iloc[te_idx], preds)
        print(f"[Fold {fold}/{k}] " + _fmt(m))
        fold_metrics.append(m)

    agg = {
        key: (
            float(np.mean([m[key] for m in fold_metrics])),
            float(np.std([m[key] for m in fold_metrics])),
        )
        for key in fold_metrics[0]
    }
    print(f"[K-Fold CV] {target} mean +/- std:")
    for key, (mean, std) in agg.items():
        print(f"    {key:<5}: {mean:.4f} +/- {std:.4f}")

    # Final model trained on the full dataset.
    final = fit_model(X, y)
    _save_model(final, models_dir, model_name)
    # Use the full-data indices for a held-out scaffold split record if possible.
    if SMILES_COL in meta.columns and len(meta) == len(X):
        train_idx, test_idx = scaffold_split(meta[SMILES_COL].tolist())
    else:
        train_idx = list(range(len(X)))
        test_idx = []
    _save_split(train_idx, test_idx, models_dir)
    return final, agg


def _save_model(model, models_dir, model_name) -> None:
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.abspath(os.path.join(models_dir, model_name))
    joblib.dump(model, path)
    logger.info("Saved trained model -> %s", path)
    print(f"Saved model -> {path}")


def _save_split(train_idx, test_idx, models_dir) -> None:
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.abspath(os.path.join(models_dir, "split_indices.csv"))
    pd.DataFrame({"train_idx": pd.Series(train_idx),
                  "test_idx": pd.Series(test_idx)}).to_csv(path, index=False)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train an XGBoost transfection-efficiency regressor with scaffold splitting."
    )
    parser.add_argument("--features", default=DEFAULT_FEATURES)
    parser.add_argument("--targets", default=DEFAULT_TARGETS)
    parser.add_argument("--metadata", default=DEFAULT_METADATA)
    parser.add_argument("--models-dir", default=DEFAULT_MODELS_DIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--target", default=DEFAULT_TARGET,
                        help="Target column to predict (default: Transfection).")
    parser.add_argument("--cv", type=int, default=None,
                        help="If set, run K-fold CV with K folds then refit on all.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = _build_parser().parse_args(argv)
    X, y, meta = load_dataset(
        args.features, args.targets, args.metadata, args.target
    )
    print(f"Training XGBoost regressor for target='{args.target}' "
          f"(model -> {os.path.join(args.models_dir, args.model_name)})")
    if args.cv:
        train_kfold(
            X, y, meta, args.cv, args.models_dir, args.model_name, args.target
        )
    else:
        train_scaffold_split(
            X, y, meta, args.models_dir, args.model_name, args.target
        )


if __name__ == "__main__":
    main()
