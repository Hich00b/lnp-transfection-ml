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
from rdkit import Chem, DataStructs
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


def buffered_split(
    smiles: Sequence[str],
    fp_radius: int = 2,
    fp_bits: int = 2048,
    distance_cutoff: float = 0.1,
    target_test_frac: float = 0.2,
    min_cluster_size_for_test: int = 1,
) -> tuple[list[int], list[int]]:
    """Similarity-based split with an explicitly ENFORCED buffer.

    Unlike Butina cluster membership, this directly verifies that no
    training molecule is within distance_cutoff of any test molecule,
    by computing the actual pairwise distance for every remaining
    candidate against every test molecule, and dropping any that
    violate the cutoff instead of assuming cluster boundaries handle it.
    """
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.ML.Cluster import Butina
    from rdkit import DataStructs

    # Generate Morgan fingerprints as RDKit ExplicitBitVect objects (lower-triangular order)
    logger.info("Computing Morgan fingerprints for buffered split (radius=%d, bits=%d)",
                fp_radius, fp_bits)
    fps = []
    valid_indices = []

    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            logger.warning("Invalid SMILES for buffered split: %s", smi)
            continue
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=fp_radius, fpSize=fp_bits)
        fp = gen.GetFingerprint(mol)  # Returns ExplicitBitVect
        fps.append(fp)
        valid_indices.append(i)

    if not fps:
        raise ValueError("No valid molecules for buffered split")

    # Compute pairwise Tanimoto distances in lower-triangular order (same as fixed clustering)
    logger.info("Computing pairwise Tanimoto distances for %d molecules", len(fps))
    distance_matrix = []
    nfps = len(fps)
    for i in range(1, nfps):
        # Compute similarity of fingerprint i with all previous fingerprints 0..i-1
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        # Convert similarity to distance (1 - similarity) and extend list
        distance_matrix.extend([1.0 - s for s in sims])

    # Perform clustering (only used to pick chemically coherent candidate test region)
    logger.info("Clustering with Butina algorithm (distance cutoff=%.2f)", distance_cutoff)
    clusters = Butina.ClusterData(distance_matrix, nfps, distance_cutoff, isDistData=True)

    # Sort clusters smallest-first for deterministic accumulation into test set
    clusters_sorted = sorted(clusters, key=len)  # smallest first

    # Accumulate whole clusters into candidate test set until target_test_frac is reached
    n_mols = len(valid_indices)
    n_test_target = int(target_test_frac * n_mols)

    candidate_test_indices = []
    n_test_so_far = 0

    for cluster in clusters_sorted:
        if n_test_so_far + len(cluster) <= n_test_target:
            # Add entire cluster to candidate test set
            candidate_test_indices.extend([valid_indices[i] for i in cluster])
            n_test_so_far += len(cluster)
        else:
            # Stop at first cluster that would meet or exceed target
            break

    # If we didn't reach target and there are remaining clusters, take part of the next one
    # But per spec, we stop at first cluster that would meet/exceed target, so we don't split clusters
    # If we still need more and haven't taken any clusters, take the smallest one
    if n_test_so_far < n_test_target and not candidate_test_indices and clusters_sorted:
        # Take the smallest cluster to ensure we have something in test
        smallest_cluster = clusters_sorted[0]
        candidate_test_indices.extend([valid_indices[i] for i in smallest_cluster])
        n_test_so_far = len(smallest_cluster)
        logger.info("Taking smallest cluster (%d members) to meet minimum test size",
                    len(smallest_cluster))

    # Convert to sets for faster lookup
    candidate_test_set = set(candidate_test_indices)

    # Now apply exhaustive distance-based buffering:
    # For every molecule NOT in candidate test set, compute distance to every test molecule
    # If min distance < distance_cutoff, drop it (exclude from both train and test)
    train_indices = []
    dropped_indices = []

    logger.info("Applying distance-based buffering: removing train molecules within %.2f of test set", distance_cutoff)

    # Pre-extract test fingerprints for efficiency
    test_fps = [fps[i] for i in candidate_test_indices]

    for idx in valid_indices:
        if idx in candidate_test_set:
            # This is in the candidate test set - keep it as test (do NOT add to train)
            pass
        else:
            # This is a candidate train molecule - check distance to ALL test molecules
            mol_fp = fps[idx]
            # Compute similarity to all test fingerprints
            sims = DataStructs.BulkTanimotoSimilarity(mol_fp, test_fps)
            # Find minimum distance (1 - maximum similarity)
            if sims:  # Should always be true if we have test molecules
                max_sim = max(sims)
                min_dist = 1.0 - max_sim
                if min_dist >= distance_cutoff:
                    # Far enough from all test molecules - keep in train
                    train_indices.append(idx)
                else:
                    # Too close to at least one test molecule - drop entirely
                    dropped_indices.append(idx)
                    logger.debug("Dropping molecule %d (min distance=%.3f to test set)", idx, min_dist)
            else:
                # No test molecules - keep in train (shouldn't happen with valid target_test_frac)
                train_indices.append(idx)

    # The test set is exactly our candidate test set (we don't drop test molecules)
    test_indices = candidate_test_indices

    # Log results
    logger.info("Buffered split results:")
    logger.info("  Candidate test set size: %d", len(test_indices))
    logger.info("  Clean train set size: %d", len(train_indices))
    logger.info("  Dropped count: %d (%.1f%% of candidate pool)",
                len(dropped_indices),
                100.0 * len(dropped_indices) / (len(train_indices) + len(dropped_indices)) if (len(train_indices) + len(dropped_indices)) > 0 else 0.0)
    print(f"[Buffered Split] Test size: {len(test_indices)}, Clean train size: {len(train_indices)}, Dropped: {len(dropped_indices)} ({100.0 * len(dropped_indices) / (len(train_indices) + len(dropped_indices)):.1f}% of candidate pool)")

    # EXHAUSTIVE VERIFICATION: verify no leakage
    logger.info("Running exhaustive leakage verification...")
    violations = []

    # Check every test molecule against every train molecule
    train_fps = [fps[i] for i in train_indices]
    for test_idx in test_indices:
        test_fp = fps[test_idx]
        if train_fps:  # Only check if we have training molecules
            sims = DataStructs.BulkTanimotoSimilarity(test_fp, train_fps)
            if sims:  # Should always be true if we have train molecules
                max_sim = max(sims)
                min_dist = 1.0 - max_sim
                if min_dist < distance_cutoff:
                    violations.append((test_idx, min_dist))
                    logger.warning("LEAKAGE VIOLATION: test molecule %d is %.3f from nearest train molecule", test_idx, min_dist)

    if violations:
        error_msg = f"Buffered split verification failed: {len(violations)} leakage violations found. First few: {violations[:5]}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    else:
        logger.info("Exhaustive verification passed: 0 leakage violations found across %d test molecules", len(test_indices))
        print(f"[Buffered Split Verification] 0 leakage violations found across {len(test_indices)} test molecules")

    # Final verification: ensure no overlap between train and test
    train_set = set(train_indices)
    test_set = set(test_indices)
    overlap = train_set.intersection(test_set)
    if overlap:
        raise RuntimeError(f"Train/test overlap detected: {overlap}")

    logger.info("Final split: %d train / %d test", len(train_indices), len(test_indices))
    return train_indices, test_indices


def buffered_leave_one_group_out(
    smiles: Sequence[str],
    fp_radius: int = 2,
    fp_bits: int = 2048,
    distance_cutoff: float = 0.1,
    min_cluster_size: int = 10,
    random_state: int = 42,
) -> list[tuple[list[int], list[int]]]:
    """Leave-one-group-out evaluation with explicit buffering.

    Similar to buffered_split but iterates over each cluster above min_cluster_size,
    holding out each cluster in turn as test set, then applies exhaustive
    distance-based buffering to the training candidates.

    Returns:
        List of (train_indices, test_indices) tuples, one for each group held out.
    """
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.ML.Cluster import Butina
    from rdkit import DataStructs

    # Generate Morgan fingerprints as RDKit ExplicitBitVect objects (lower-triangular order)
    logger.info("Computing Morgan fingerprints for buffered leave-one-group-out (radius=%d, bits=%d)",
                fp_radius, fp_bits)
    fps = []
    valid_indices = []

    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=fp_radius, fpSize=fp_bits)
        fp = gen.GetFingerprint(mol)  # Returns ExplicitBitVect
        fps.append(fp)
        valid_indices.append(i)

    if not fps:
        raise ValueError("No valid molecules for buffered leave-one-group-out")

    # Compute pairwise Tanimoto distances in lower-triangular order
    logger.info("Computing pairwise Tanimoto distances for %d molecules", len(fps))
    distance_matrix = []
    nfps = len(fps)
    for i in range(1, nfps):
        # Compute similarity of fingerprint i with all previous fingerprints 0..i-1
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        # Convert similarity to distance (1 - similarity) and extend list
        distance_matrix.extend([1.0 - s for s in sims])

    # Perform clustering
    logger.info("Clustering with Butina algorithm (distance cutoff=%.2f)", distance_cutoff)
    clusters = Butina.ClusterData(distance_matrix, nfps, distance_cutoff, isDistData=True)

    # Filter clusters by minimum size
    clusters_sorted = sorted([c for c in clusters if len(c) >= min_cluster_size], key=len, reverse=True)
    logger.info("Found %d clusters with size >= %d", len(clusters_sorted), min_cluster_size)

    if not clusters_sorted:
        logger.warning("No clusters meet minimum size requirement of %d", min_cluster_size)
        return []

    # Generate leave-one-group-out splits with buffering
    splits = []
    for i, test_cluster in enumerate(clusters_sorted):
        # Test set is this cluster
        test_indices = [valid_indices[idx] for idx in test_cluster]

        # Candidate training pool is everything not in this test cluster
        candidate_train_indices = [idx for idx in valid_indices if idx not in test_indices]
        logger.info("Processing group %d (size=%d) as test set", i, len(test_indices))

        # Apply exhaustive distance-based buffering to candidate training set
        train_indices = []
        dropped_indices = []

        # Pre-extract test fingerprints for efficiency
        test_fps = [fps[idx] for idx in test_indices]

        for idx in candidate_train_indices:
            # This is a candidate train molecule - check distance to ALL test molecules
            mol_fp = fps[idx]
            # Compute similarity to all test fingerprints
            sims = DataStructs.BulkTanimotoSimilarity(mol_fp, test_fps)
            # Find minimum distance (1 - maximum similarity)
            if sims:  # Should always be true if we have test molecules
                max_sim = max(sims)
                min_dist = 1.0 - max_sim
                if min_dist >= distance_cutoff:
                    # Far enough from all test molecules - keep in train
                    train_indices.append(idx)
                else:
                    # Too close to at least one test molecule - drop entirely
                    dropped_indices.append(idx)
                    logger.debug("Dropping molecule %d (min distance=%.3f to test set %d)", idx, min_dist, i)
            else:
                # No test molecules - keep in train (shouldn't happen)
                train_indices.append(idx)

        # Log results for this fold
        logger.info("Group %d results:", i)
        logger.info("  Test set size: %d", len(test_indices))
        logger.info("  Clean train set size: %d", len(train_indices))
        logger.info("  Dropped count: %d (%.1f%%)",
                    len(dropped_indices),
                    100.0 * len(dropped_indices) / len(candidate_train_indices) if len(candidate_train_indices) > 0 else 0.0)
        print(f"[LOCO Group {i}/{len(clusters_sorted)}] Test size: {len(test_indices)}, Clean train size: {len(train_indices)}, Dropped: {len(dropped_indices)} ({100.0 * len(dropped_indices) / len(candidate_train_indices) if len(candidate_train_indices) > 0 else 0.0:.1f}%)")

        # EXHAUSTIVE VERIFICATION for this fold: verify no leakage
        logger.info("Running exhaustive leakage verification for group %d...", i)
        violations = []

        # Check every test molecule against every train molecule
        train_fps = [fps[idx] for idx in train_indices]
        for test_idx in test_indices:
            test_fp = fps[test_idx]
            if train_fps:  # Only check if we have training molecules
                sims = DataStructs.BulkTanimotoSimilarity(test_fp, train_fps)
                if sims:  # Should always be true if we have train molecules
                    max_sim = max(sims)
                    min_dist = 1.0 - max_sim
                    if min_dist < distance_cutoff:
                        violations.append((test_idx, min_dist, i))  # Include group index
                        logger.warning("LEAKAGE VIOLATION in group %d: test molecule %d is %.3f from nearest train molecule", i, test_idx, min_dist)

        if violations:
            error_msg = f"Buffered leave-one-group-out verification failed for group {i}: {len(violations)} leakage violations found. First few: {violations[:5]}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        else:
            logger.info("Exhaustive verification passed for group %d: 0 leakage violations found across %d test molecules", i, len(test_indices))
            print(f"[LOCO Group {i} Verification] 0 leakage violations found across {len(test_indices)} test molecules")

        # Final verification: ensure no overlap between train and test for this fold
        train_set = set(train_indices)
        test_set = set(test_indices)
        overlap = train_set.intersection(test_set)
        if overlap:
            raise RuntimeError(f"Train/test overlap detected in group {i}: {overlap}")

        splits.append((train_indices, test_indices))
        logger.info("Group %d held out: %d train / %d test", i, len(train_indices), len(test_indices))

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
def train_leave_one_group_out(X, y, meta, models_dir, model_name, target,
                               fp_radius=2, fp_bits=2048, distance_cutoff=0.1,
                               min_cluster_size=10):
    """Leave-one-group-out evaluation: hold out each group in turn, report distribution."""
    smiles = meta[SMILES_COL].tolist() if SMILES_COL in meta.columns else []
    if len(smiles) != len(X):
        logger.warning(
            "Metadata length mismatch (%d vs %d); cannot perform leave-one-group-out.",
            len(smiles), len(X),
        )
        return None, None, None

    splits = buffered_leave_one_group_out(
        smiles, fp_radius=fp_radius, fp_bits=fp_bits,
        distance_cutoff=distance_cutoff, min_cluster_size=min_cluster_size
    )

    if not splits:
        logger.warning("No valid groups for leave-one-group-out evaluation")
        return None, None, None

    fold_metrics: list[dict[str, float]] = []
    # For pooled metrics: collect all out-of-fold predictions and true values
    all_oof_preds = []
    all_oof_true = []

    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        model = fit_model(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[test_idx])
        m = regression_metrics(y.iloc[test_idx], preds)
        print(f"[LOCO Group {fold}/{len(splits)}] " + _fmt(m))
        fold_metrics.append(m)

        # Collect out-of-fold predictions for pooled metrics
        all_oof_preds.extend(preds)
        all_oof_true.extend(y.iloc[test_idx])

    if not fold_metrics:
        return None, None, None

    # Compute pooled metrics (PRIMARY)
    pooled_metrics = regression_metrics(all_oof_true, all_oof_preds)

    # Compute per-fold mean +/- std (SECONDARY)
    agg = {
        key: (
            float(np.mean([m[key] for m in fold_metrics])),
            float(np.std([m[key] for m in fold_metrics])),
        )
        for key in fold_metrics[0]
    }

    # Print both pooled (primary) and per-fold mean +/- std (secondary)
    print(f"[Leave-One-Group-Out] {target} POOLED METRICS (PRIMARY): " + _fmt(pooled_metrics))
    print(f"[Leave-One-Group-Out] {target} mean +/- std across {len(fold_metrics)} groups (SECONDARY, unstable for small groups):")
    for key, (mean, std) in agg.items():
        print(f"    {key:<5}: {mean:.4f} +/- {std:.4f}")

    # Also return per-group results for detailed inspection
    per_group_results = []
    for i, (train_idx, test_idx) in enumerate(splits):
        model = fit_model(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[test_idx])
        m = regression_metrics(y.iloc[test_idx], preds)
        per_group_results.append({
            'group': i+1,
            'group_size': len([idx for idx in meta[SMILES_COL].tolist() if idx in (set(range(len(meta[SMILES_COL].tolist()))) - set(train_idx))]),  # Approximate
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

    # Return both pooled metrics (as primary agg) and per-group results
    return final, pooled_metrics, per_group_results


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
                        fp_radius=2, fp_bits=2048, distance_cutoff=0.1):
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
        elif split_method == "buffered":
            train_idx, test_idx = buffered_split(smiles, fp_radius=fp_radius, fp_bits=fp_bits,
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
               fp_radius=2, fp_bits=2048, distance_cutoff=0.1):
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
        elif split_method == "buffered":
            train_idx, test_idx = buffered_split(meta[SMILES_COL].tolist(),
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
        description="Train an XGBoost transfection-efficiency regressor with scaffold or buffered splitting."
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
    parser.add_argument("--split-method", choices=["scaffold", "buffered"], default="scaffold",
                        help="Method to use for train/test split (default: scaffold)")
    parser.add_argument("--cv-clusters", action="store_true",
                        help="Perform leave-one-group-out evaluation instead of regular CV")
    parser.add_argument("--fp-radius", type=int, default=2,
                        help="Radius for Morgan fingerprint in clustering (default: 2)")
    parser.add_argument("--fp-bits", type=int, default=2048,
                        help="Number of bits for Morgan fingerprint in clustering (default: 2048)")
    parser.add_argument("--distance-cutoff", type=float, default=0.1,
                        help="Distance cutoff for buffered splitting (default: 0.1)")
    parser.add_argument("--min-cluster-size", type=int, default=10,
                        help="Minimum group size for leave-one-group-out (default: 10)")
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
        # Leave-one-group-out evaluation
        result = train_leave_one_group_out(
            X, y, meta, args.models_dir, args.model_name, args.target,
            fp_radius=args.fp_radius, fp_bits=args.fp_bits,
            distance_cutoff=args.distance_cutoff, min_cluster_size=args.min_cluster_size
        )
        if result[0] is None:  # Training failed
            logger.error("Leave-one-group-out evaluation failed")
            return
        # Unpack the result
        final_model, pooled_metrics, per_group_results = result
        # Print the pooled metrics as primary
        print(f"[Leave-One-Group-Out] {args.target} POOLED METRICS (PRIMARY): " + _fmt(pooled_metrics))
        # Print the per-group results as secondary (mean +/- std across groups)
        if per_group_results:
            # We have a list of dicts for each group
            # We'll compute the mean and std for each metric across groups
            r2_vals = [m['R2'] for m in per_group_results]
            rmse_vals = [m['RMSE'] for m in per_group_results]
            mae_vals = [m['MAE'] for m in per_group_results]
            mean_r2 = np.mean(r2_vals)
            std_r2 = np.std(r2_vals)
            mean_rmse = np.mean(rmse_vals)
            std_rmse = np.std(rmse_vals)
            mean_mae = np.mean(mae_vals)
            std_mae = np.std(mae_vals)
            print(f"[Leave-One-Group-Out] {args.target} mean +/- std across {len(per_group_results)} groups (SECONDARY, unstable for small groups):")
            print(f"    R2   : {mean_r2:.4f} +/- {std_r2:.4f}")
            print(f"    RMSE : {mean_rmse:.4f} +/- {std_rmse:.4f}")
            print(f"    MAE  : {mean_mae:.4f} +/- {std_mae:.4f}")
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
