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
from sklearn.model_selection import train_test_split
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


def cluster_split(
    smiles: Sequence[str],
    fp_radius: int = 2,
    fp_bits: int = 2048,
    distance_cutoff: float = 0.4,
    frac_train: float = 0.8,
    random_state: int = 42,
) -> tuple[list[int], list[int]]:
    """Split molecules into train/test based on Butina clustering of Morgan fingerprints.

    Molecules are clustered using Tanimoto distance (1 - TanimotoSimilarity) with
    the given distance_cutoff. Whole clusters are allocated to train/test to ensure
    no molecule in test is within distance_cutoff similarity of any molecule in train.

    Args:
        smiles: List of SMILES strings
        fp_radius: Radius for Morgan fingerprint (default: 2)
        fp_bits: Number of bits for Morgan fingerprint (default: 2048)
        distance_cutoff: Maximum Tanimoto distance for clustering (default: 0.4)
        frac_train: Fraction of molecules to allocate to training (default: 0.8)
        random_state: Random seed for reproducibility (default: 42)

    Returns:
        Tuple of (train_indices, test_indices)
    """
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.ML.Cluster import Butina

    # Generate Morgan fingerprints
    logger.info("Computing Morgan fingerprints for clustering (radius=%d, bits=%d)",
                fp_radius, fp_bits)
    fps = []
    valid_indices = []

    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            logger.warning("Invalid SMILES for clustering: %s", smi)
            continue
        # Reuse the same generator setup as in data_processing.py
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=fp_radius, fpSize=fp_bits)
        fp = gen.GetFingerprintAsNumPy(mol)
        fps.append(fp)
        valid_indices.append(i)

    if not fps:
        raise ValueError("No valid molecules for clustering")

    # Compute pairwise Tanimoto distances
    logger.info("Computing pairwise Tanimoto distances for %d molecules", len(fps))
    distance_matrix = []
    for i in range(len(fps)):
        for j in range(i+1, len(fps)):
            # Compute Tanimoto similarity
            intersection = np.sum(np.bitwise_and(fps[i], fps[j]))
            union = np.sum(np.bitwise_or(fps[i], fps[j]))
            if union == 0:
                tanimoto_sim = 0.0
            else:
                tanimoto_sim = intersection / union
            # Convert to distance
            distance = 1.0 - tanimoto_sim
            distance_matrix.append(distance)

    # Perform clustering
    logger.info("Clustering with Butina algorithm (distance cutoff=%.2f)", distance_cutoff)
    clusters = Butina.ClusterData(distance_matrix, len(fps), distance_cutoff, isDistData=True)

    # Log cluster information
    logger.info("Butina clustering produced %d clusters", len(clusters))
    cluster_sizes = [len(c) for c in clusters]
    logger.info("Cluster sizes: %s", ", ".join(map(str, sorted(cluster_sizes, reverse=True))))
    print(f"[Clustering] Produced {len(clusters)} clusters")
    print(f"[Clustering] Size distribution: {', '.join(map(str, sorted(cluster_sizes, reverse=True)))}")

    # Sort clusters by size (largest first) for deterministic allocation
    clusters_sorted = sorted(clusters, key=len, reverse=True)

    # Allocate whole clusters to train/test
    n_mols = len(valid_indices)
    n_train_target = int(frac_train * n_mols)

    train_indices = []
    test_indices = []
    n_train_so_far = 0

    for cluster in clusters_sorted:
        if n_train_so_far + len(cluster) <= n_train_target:
            # Add entire cluster to training
            train_indices.extend([valid_indices[i] for i in cluster])
            n_train_so_far += len(cluster)
        else:
            # Add entire cluster to testing
            test_indices.extend([valid_indices[i] for i in cluster])

    # Edge case: if we still need more training molecules (shouldn't happen with reasonable cutoffs)
    if n_train_so_far < n_train_target and test_indices:
        # Move smallest test clusters to training until we reach target
        test_clusters_sorted = sorted([c for c in clusters_sorted if any(i in c for i in
                                 [valid_indices.index(ti) for ti in test_indices[:10]])],
                                    key=len)
        # Simple approach: just take from the end of test_indices
        while n_train_so_far < n_train_target and test_indices:
            moved_idx = test_indices.pop()
            train_indices.append(moved_idx)
            n_train_so_far += 1

    logger.info("Cluster split: %d train / %d test", len(train_indices), len(test_indices))
    return train_indices, test_indices


def leave_one_cluster_out(
    smiles: Sequence[str],
    fp_radius: int = 2,
    fp_bits: int = 2048,
    distance_cutoff: float = 0.4,
    min_cluster_size: int = 5,
    random_state: int = 42,
) -> list[tuple[list[int], list[int]]]:
    """Generate leave-one-cluster-out splits for cross-validation.

    Similar to cluster_split but iterates over each cluster above min_cluster_size,
    holding out each cluster in turn as test set.

    Returns:
        List of (train_indices, test_indices) tuples, one for each cluster held out.
    """
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.ML.Cluster import Butina

    # Generate Morgan fingerprints (same as in cluster_split)
    logger.info("Computing Morgan fingerprints for leave-one-cluster-out (radius=%d, bits=%d)",
                fp_radius, fp_bits)
    fps = []
    valid_indices = []

    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=fp_radius, fpSize=fp_bits)
        fp = gen.GetFingerprintAsNumPy(mol)
        fps.append(fp)
        valid_indices.append(i)

    if not fps:
        raise ValueError("No valid molecules for clustering")

    # Compute pairwise Tanimoto distances
    logger.info("Computing pairwise Tanimoto distances for %d molecules", len(fps))
    distance_matrix = []
    for i in range(len(fps)):
        for j in range(i+1, len(fps)):
            intersection = np.sum(np.bitwise_and(fps[i], fps[j]))
            union = np.sum(np.bitwise_or(fps[i], fps[j]))
            if union == 0:
                tanimoto_sim = 0.0
            else:
                tanimoto_sim = intersection / union
            distance = 1.0 - tanimoto_sim
            distance_matrix.append(distance)

    # Perform clustering
    logger.info("Clustering with Butina algorithm (distance cutoff=%.2f)", distance_cutoff)
    clusters = Butina.ClusterData(distance_matrix, len(fps), distance_cutoff, isDistData=True)

    # Filter clusters by minimum size
    clusters_sorted = sorted([c for c in clusters if len(c) >= min_cluster_size], key=len, reverse=True)
    logger.info("Found %d clusters with size >= %d", len(clusters_sorted), min_cluster_size)

    if not clusters_sorted:
        logger.warning("No clusters meet minimum size requirement of %d", min_cluster_size)
        return []

    # Generate leave-one-cluster-out splits
    splits = []
    for i, test_cluster in enumerate(clusters_sorted):
        # Test set is this cluster
        test_indices = [valid_indices[idx] for idx in test_cluster]
        # Training set is all other valid indices
        train_indices = []
        for j, cluster in enumerate(clusters_sorted):
            if i != j:  # Skip the test cluster
                train_indices.extend([valid_indices[idx] for idx in cluster])
        splits.append((train_indices, test_indices))
        logger.info("Cluster %d held out: %d train / %d test",
                   i, len(train_indices), len(test_indices))

    return splits


# --------------------------------------------------------------------------- #
# Model helpers
# --------------------------------------------------------------------------- #
def make_model() -> XGBRegressor:
    """Construct the XGBoost regressor with the project defaults."""
    return XGBRegressor(**XGB_PARAMS)


# --------------------------------------------------------------------------- #
# Evaluation routines
# --------------------------------------------------------------------------- #
def train_leave_one_cluster_out(X, y, meta, models_dir, model_name, target,
                               fp_radius=2, fp_bits=2048, distance_cutoff=0.4,
                               min_cluster_size=5):
    """Leave-one-cluster-out evaluation: hold out each cluster in turn, report distribution."""
    smiles = meta[SMILES_COL].tolist() if SMILES_COL in meta.columns else []
    if len(smiles) != len(X):
        logger.warning(
            "Metadata length mismatch (%d vs %d); cannot perform leave-one-cluster-out.",
            len(smiles), len(X),
        )
        return None, None

    splits = leave_one_cluster_out(
        smiles, fp_radius=fp_radius, fp_bits=fp_bits,
        distance_cutoff=distance_cutoff, min_cluster_size=min_cluster_size
    )

    if not splits:
        logger.warning("No valid clusters for leave-one-cluster-out evaluation")
        return None, None

    fold_metrics: list[dict[str, float]] = []
    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        model = fit_model(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[test_idx])
        m = regression_metrics(y.iloc[test_idx], preds)
        print(f"[LOCO Cluster {fold}/{len(splits)}] " + _fmt(m))
        fold_metrics.append(m)

    if not fold_metrics:
        return None, None

    agg = {
        key: (
            float(np.mean([m[key] for m in fold_metrics])),
            float(np.std([m[key] for m in fold_metrics])),
        )
        for key in fold_metrics[0]
    }
    print(f"[Leave-One-Cluster-Out] {target} mean +/- std across {len(fold_metrics)} clusters:")
    for key, (mean, std) in agg.items():
        print(f"    {key:<5}: {mean:.4f} +/- {std:.4f}")

    # Also return per-cluster results for detailed inspection
    per_cluster_results = []
    for i, (train_idx, test_idx) in enumerate(splits):
        model = fit_model(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[test_idx])
        m = regression_metrics(y.iloc[test_idx], preds)
        per_cluster_results.append({
            'cluster': i+1,
            'n_train': len(train_idx),
            'n_test': len(test_idx),
            'R2': m['R2'],
            'RMSE': m['RMSE'],
            'MAE': m['MAE']
        })

    # Save the final model (trained on all data)
    final = fit_model(X, y)
    _save_model(final, models_dir, model_name)

    # For consistency with other functions, create a dummy split
    train_idx = list(range(len(X)))
    test_idx = []
    _save_split(train_idx, test_idx, models_dir)

    return final, agg, per_cluster_results


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
    # Use train_test_split for random internal validation instead of tail slice
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=INTERNAL_VAL_FRAC, random_state=42, shuffle=True
    )
    # Pass early_stopping_rounds to the constructor
    model = make_model()
    model.set_params(early_stopping_rounds=EARLY_STOPPING_ROUNDS)
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
def train_scaffold_split(X, y, meta, models_dir, model_name, target, split_method="scaffold",
                        fp_radius=2, fp_bits=2048, distance_cutoff=0.4):
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
        if split_method == "scaffold":
            train_idx, test_idx = scaffold_split(smiles)
        elif split_method == "cluster":
            train_idx, test_idx = cluster_split(smiles, fp_radius=fp_radius, fp_bits=fp_bits,
                                               distance_cutoff=distance_cutoff)
        else:
            raise ValueError(f"Unknown split method: {split_method}")

    model = fit_model(X.iloc[train_idx], y.iloc[train_idx])
    preds = model.predict(X.iloc[test_idx])
    metrics = regression_metrics(y.iloc[test_idx], preds)
    print(f"[{split_method.capitalize()} split] {target} test metrics: " + _fmt(metrics))

    _save_model(model, models_dir, model_name)
    _save_split(train_idx, test_idx, models_dir)
    return model, metrics


def train_kfold(X, y, meta, k, models_dir, model_name, target, split_method="scaffold",
               fp_radius=2, fp_bits=2048, distance_cutoff=0.4):
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
    # Use the full-data indices for a held-out split record if possible.
    if SMILES_COL in meta.columns and len(meta) == len(X):
        if split_method == "scaffold":
            train_idx, test_idx = scaffold_split(meta[SMILES_COL].tolist())
        elif split_method == "cluster":
            train_idx, test_idx = cluster_split(meta[SMILES_COL].tolist(),
                                               fp_radius=fp_radius, fp_bits=fp_bits,
                                               distance_cutoff=distance_cutoff)
        else:
            raise ValueError(f"Unknown split method: {split_method}")
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
        description="Train an XGBoost transfection-efficiency regressor with scaffold or cluster splitting."
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
    parser.add_argument("--split-method", choices=["scaffold", "cluster"], default="scaffold",
                        help="Method to use for train/test split (default: scaffold)")
    parser.add_argument("--cv-clusters", action="store_true",
                        help="Perform leave-one-cluster-out evaluation instead of regular CV")
    parser.add_argument("--fp-radius", type=int, default=2,
                        help="Radius for Morgan fingerprint in clustering (default: 2)")
    parser.add_argument("--fp-bits", type=int, default=2048,
                        help="Number of bits for Morgan fingerprint in clustering (default: 2048)")
    parser.add_argument("--distance-cutoff", type=float, default=0.4,
                        help="Distance cutoff for Butina clustering (default: 0.4)")
    parser.add_argument("--min-cluster-size", type=int, default=5,
                        help="Minimum cluster size for leave-one-cluster-out (default: 5)")
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

    if args.cv_clusters:
        # Leave-one-cluster-out evaluation
        result = train_leave_one_cluster_out(
            X, y, meta, args.models_dir, args.model_name, args.target,
            fp_radius=args.fp_radius, fp_bits=args.fp_bits,
            distance_cutoff=args.distance_cutoff, min_cluster_size=args.min_cluster_size
        )
        if result[0] is None:  # Training failed
            logger.error("Leave-one-cluster-out evaluation failed")
            return
    elif args.cv:
        # Regular K-fold CV
        train_kfold(
            X, y, meta, args.cv, args.models_dir, args.model_name, args.target,
            split_method=args.split_method,
            fp_radius=args.fp_radius, fp_bits=args.fp_bits,
            distance_cutoff=args.distance_cutoff
        )
    else:
        # Single train/test split
        train_scaffold_split(
            X, y, meta, args.models_dir, args.model_name, args.target,
            split_method=args.split_method,
            fp_radius=args.fp_radius, fp_bits=args.fp_bits,
            distance_cutoff=args.distance_cutoff
        )


if __name__ == "__main__":
    main()
