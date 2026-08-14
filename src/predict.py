"""Inference for LNP transfection-efficiency prediction, with applicability-domain checking.

Feature extraction is imported from data_processing.py rather than reimplemented here
(previously this file duplicated the Morgan fingerprint + descriptor logic, which risked
silent drift between training-time and inference-time features). Predictions on a
candidate SMILES outside the validated applicability domain are flagged explicitly,
not silently returned as if equally reliable.

Run as::

    python src/predict.py --smiles "CCCC..." --reference data/raw/dataset.csv
    python src/predict.py --input candidates.csv --reference data/raw/dataset.csv --output ranked.csv
"""

from __future__ import annotations

import argparse
import warnings

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs

from data_processing import smiles_to_mol, featurize_mol, MORGAN_GENERATOR

warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
warnings.filterwarnings("ignore", category=UserWarning, module="pickle")

# Validated in this project's Validation Design + Applicability analysis:
# distance_cutoff=0.1 is the only cutoff that produces a genuinely leak-free
# buffered split with a usable training set size on this library. Predictions
# for candidates farther than this from every training compound are outside
# the domain this model has actually been shown to generalize to.
DEFAULT_APPLICABILITY_CUTOFF = 0.1


def build_reference_library(reference_csv: str, smiles_col: str = "SMILES"):
    """Load the training library and precompute its fingerprints once,
    for reuse across many applicability-domain checks."""
    df = pd.read_csv(reference_csv)
    mols = [smiles_to_mol(s) for s in df[smiles_col]]
    valid = [(s, m) for s, m in zip(df[smiles_col], mols) if m is not None]
    ref_smiles = [s for s, _ in valid]
    ref_fps = [MORGAN_GENERATOR.GetFingerprint(m) for _, m in valid]
    return ref_smiles, ref_fps


def check_applicability_domain(
    query_smiles: str,
    ref_smiles: list[str],
    ref_fps: list,
    cutoff: float = DEFAULT_APPLICABILITY_CUTOFF,
) -> dict:
    """Nearest-neighbor Tanimoto distance to the training library, and an
    explicit in/out-of-domain flag. Does not attempt to explain WHY a
    prediction might be wrong, only whether the model has been validated
    to make predictions this far from anything it was trained on."""
    mol = smiles_to_mol(query_smiles)
    if mol is None:
        return {"in_domain": False, "distance": None, "nearest_neighbor": None,
                "reason": "SMILES did not parse"}
    query_fp = MORGAN_GENERATOR.GetFingerprint(mol)
    sims = DataStructs.BulkTanimotoSimilarity(query_fp, ref_fps)
    max_sim = max(sims)
    nn_idx = int(np.argmax(sims))
    distance = 1 - max_sim
    return {
        "in_domain": distance < cutoff,
        "distance_to_nearest_training_compound": distance,
        "nearest_neighbor_smiles": ref_smiles[nn_idx],
        "confidence_note": (
            "within validated applicability domain" if distance < cutoff
            else f"OUTSIDE validated domain (distance {distance:.3f} >= cutoff {cutoff}); "
                 "this model's generalization to compounds this novel has not been "
                 "established, and out-of-distribution evaluation on this library "
                 "showed a substantially wider error spread than in-domain predictions"
        ),
    }


def featurize_for_model(smiles: str) -> dict | None:
    """Build the feature dict for one SMILES using data_processing.py's
    functions directly, never a locally reimplemented version."""
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None
    return featurize_mol(mol)


def predict_one(model, smiles: str, ref_smiles, ref_fps, cutoff) -> dict:
    feats = featurize_for_model(smiles)
    domain = check_applicability_domain(smiles, ref_smiles, ref_fps, cutoff)
    if feats is None:
        return {"smiles": smiles, "predicted_transfection": None, **domain}
    # Named DataFrame, single row -- XGBoost aligns by column name, not position,
    # so this is safe regardless of what order the model was actually trained with.
    X = pd.DataFrame([feats])
    pred = float(model.predict(X)[0])
    return {"smiles": smiles, "predicted_transfection": pred, **domain}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Predict transfection efficiency with applicability-domain flagging."
    )
    parser.add_argument("--smiles", type=str, help="Single SMILES string.")
    parser.add_argument("--input", type=str, help="CSV with a SMILES column, for batch/ranking mode.")
    parser.add_argument("--output", type=str, default="data/predictions.csv")
    parser.add_argument("--model", type=str, default="models/xgb_transfection.pkl")
    parser.add_argument("--reference", type=str, required=True,
                         help="Training library CSV (e.g. data/raw/dataset.csv), used for the "
                              "applicability-domain nearest-neighbor check.")
    parser.add_argument("--cutoff", type=float, default=DEFAULT_APPLICABILITY_CUTOFF)
    parser.add_argument("--smiles-col", default="SMILES")
    args = parser.parse_args(argv)

    model = joblib.load(args.model)
    ref_smiles, ref_fps = build_reference_library(args.reference, args.smiles_col)
    print(f"Loaded reference library: {len(ref_smiles)} compounds, applicability cutoff={args.cutoff}")

    if args.smiles:
        result = predict_one(model, args.smiles, ref_smiles, ref_fps, args.cutoff)
        print(f"\nSMILES: {args.smiles}")
        if result["predicted_transfection"] is None:
            print(f"Error: {result.get('reason', 'could not featurize')}")
        else:
            print(f"Predicted Transfection Efficiency: {result['predicted_transfection']:.4f}")
            print(f"Applicability: {result['confidence_note']}")

    elif args.input:
        df = pd.read_csv(args.input)
        if args.smiles_col not in df.columns:
            raise ValueError(f"Input CSV must contain a '{args.smiles_col}' column.")

        rows = [predict_one(model, s, ref_smiles, ref_fps, args.cutoff) for s in df[args.smiles_col]]
        out_df = pd.DataFrame(rows)

        n_failed = out_df["predicted_transfection"].isna().sum()
        n_out_of_domain = (~out_df["in_domain"]).sum()
        if n_failed:
            print(f"Warning: {n_failed} SMILES failed to parse and have no prediction.")
        print(f"{n_out_of_domain}/{len(out_df)} candidates are OUTSIDE the validated applicability domain.")

        # Rank in-domain candidates first (most trustworthy), by predicted value descending;
        # out-of-domain candidates are kept but sorted separately and clearly labeled, not
        # silently interleaved with validated predictions.
        in_domain = out_df[out_df["in_domain"] & out_df["predicted_transfection"].notna()] \
            .sort_values("predicted_transfection", ascending=False)
        out_domain = out_df[~out_df["in_domain"] | out_df["predicted_transfection"].isna()]
        ranked = pd.concat([in_domain, out_domain], ignore_index=True)
        ranked.to_csv(args.output, index=False)

        print(f"\nTop 5 in-domain candidates by predicted transfection:")
        print(in_domain[["smiles", "predicted_transfection", "distance_to_nearest_training_compound"]]
              .head(5).to_string(index=False))
        print(f"\nSaved full ranked results -> {args.output}")

    else:
        print("Please provide either --smiles '<string>' or --input '<path/to/candidates.csv>'.")


if __name__ == "__main__":
    main()
