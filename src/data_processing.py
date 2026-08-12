"""Data processing for LNP transfection-efficiency prediction.

Reads a raw CSV of ionizable-lipid SMILES strings together with measured
transfection efficiency (AGILE dataset format: ``SMILES`` and ``Transfection``
columns), converts each SMILES to an RDKit ``Mol`` object (dropping any that
fail to parse), and computes a combined molecular representation:

* **ECFP4 (Morgan) fingerprints** -- radius = 2, 2048 bits.
* **Physicochemical descriptors** -- molecular weight, number of rotatable
  bonds, topological polar surface area (TPSA), and calculated LogP.

The resulting feature matrix, the regression targets, and a small metadata
table (ID / SMILES, used downstream for scaffold splitting) are written to
``data/processed/``.

This module is importable *and* runnable as a CLI, e.g.::

    python src/data_processing.py --input data/raw/dataset.csv
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Configuration constants
# --------------------------------------------------------------------------- #
MORGAN_RADIUS: int = 2
MORGAN_BITS: int = 2048

# Modern (non-deprecated) Morgan/ECFP4 generator -- created once and reused.
MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=MORGAN_RADIUS, fpSize=MORGAN_BITS
)

# (attribute name on rdkit.Chem.Descriptors, output column name)
DESCRIPTORS: tuple[tuple[str, str], ...] = (
    ("MolWt", "MolWt"),
    ("NumRotatableBonds", "NumRotatableBonds"),
    ("TPSA", "TPSA"),
    ("MolLogP", "MolLogP"),
)

DEFAULT_TARGET_COLS: tuple[str, ...] = ("Transfection",)
FEATURES_FILENAME = "features.csv"
TARGETS_FILENAME = "targets.csv"
METADATA_FILENAME = "metadata.csv"


# --------------------------------------------------------------------------- #
# SMILES parsing
# --------------------------------------------------------------------------- #
def smiles_to_mol(smiles: str) -> Optional[Chem.rdchem.Mol]:
    """Parse a SMILES string into a sanitised RDKit ``Mol``.

    Wrapped in ``try/except`` because SMILES harvested from public databases
    are frequently malformed; any parse failure returns ``None`` instead of
    raising.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception as exc:  # noqa: BLE001 -- RDKit raises varied errors
        logger.warning("RDKit raised while parsing %r: %s", smiles, exc)
        return None
    if mol is None or mol.GetNumAtoms() == 0:
        logger.warning("Invalid or empty molecule for SMILES: %r", smiles)
        return None
    return mol


# --------------------------------------------------------------------------- #
# Feature computation
# --------------------------------------------------------------------------- #
def compute_morgan_fingerprint(
    mol: Chem.rdchem.Mol,
    radius: int = MORGAN_RADIUS,
    n_bits: int = MORGAN_BITS,
) -> np.ndarray:
    """Return the ECFP4 Morgan fingerprint as a 1-D uint8 numpy array.

    Uses the modern ``rdFingerprintGenerator`` API (the legacy
    ``AllChem.GetMorganFingerprintAsBitVect`` is deprecated in recent RDKit).
    For the default ``n_bits``/``radius`` a shared generator is reused.
    """
    if radius != MORGAN_RADIUS or n_bits != MORGAN_BITS:
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    else:
        gen = MORGAN_GENERATOR
    arr = gen.GetFingerprintAsNumPy(mol).astype(np.uint8)
    return arr


def compute_descriptors(mol: Chem.rdchem.Mol) -> dict[str, float]:
    """Compute the configured physicochemical descriptors for one molecule."""
    values: dict[str, float] = {}
    for attr, col in DESCRIPTORS:
        func = getattr(Descriptors, attr)
        values[col] = float(func(mol))
    return values


def featurize_mol(mol: Chem.rdchem.Mol) -> dict[str, float]:
    """Build the full feature dict for a single valid molecule.

    Combines the 2048 Morgan fingerprint bits (``FP_0`` .. ``FP_2047``) with
    the four physicochemical descriptors.
    """
    record = compute_descriptors(mol)
    fp_bits = compute_morgan_fingerprint(mol)
    record.update({f"FP_{i}": int(b) for i, b in enumerate(fp_bits)})
    return record


# --------------------------------------------------------------------------- #
# CSV loading
# --------------------------------------------------------------------------- #
def load_raw_csv(
    path: str,
    smiles_col: str = "SMILES",
) -> pd.DataFrame:
    """Load the raw dataset and validate that it has the required columns.

    Raises
    ------
    ValueError
        If the ``SMILES`` column is missing.
    FileNotFoundError
        If ``path`` does not exist.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Raw CSV not found: {path}")
    df = pd.read_csv(path)
    if smiles_col not in df.columns:
        raise ValueError(
            f"Input CSV must contain a '{smiles_col}' column. "
            f"Found columns: {list(df.columns)}"
        )
    return df


def _resolve_target_columns(
    df: pd.DataFrame, target_cols: Sequence[str]
) -> list[str]:
    """Return only the requested target columns actually present in ``df``."""
    present = [c for c in target_cols if c in df.columns]
    if not present:
        raise ValueError(
            "None of the target columns "
            f"{list(target_cols)} were found in the CSV. "
            f"Available columns: {list(df.columns)}"
        )
    return present


# --------------------------------------------------------------------------- #
# Main processing routine
# --------------------------------------------------------------------------- #
def process_and_save(
    csv_path: str,
    out_dir: str = "data/processed",
    smiles_col: str = "SMILES",
    id_col: str = "ID",
    target_cols: Sequence[str] = DEFAULT_TARGET_COLS,
) -> dict[str, str]:
    """Run featurisation end-to-end and persist the outputs.

    Parameters
    ----------
    csv_path : str
        Path to the raw CSV.
    out_dir : str
        Directory to write ``features.csv``, ``targets.csv`` and
        ``metadata.csv`` (created if missing).
    smiles_col, id_col : str
        Column names for SMILES and the optional identifier.
    target_cols : sequence of str
        Candidate target columns; only those present are retained. At least
        one target must exist.

    Returns
    -------
    dict
        Mapping of artefact name -> absolute path written.
    """
    df = load_raw_csv(csv_path, smiles_col=smiles_col)
    targets_present = _resolve_target_columns(df, target_cols)
    logger.info(
        "Loaded %d rows from %s; targets in use: %s",
        len(df), csv_path, targets_present,
    )

    feature_rows, target_rows, meta_rows = [], [], []
    dropped = 0
    for _, row in df.iterrows():
        mol = smiles_to_mol(row[smiles_col])
        if mol is None:
            dropped += 1
            continue
        feature_rows.append(featurize_mol(mol))
        target_rows.append({t: row[t] for t in targets_present})
        meta_rows.append({
            id_col: row[id_col] if id_col in df.columns else None,
            smiles_col: row[smiles_col],
        })

    if not feature_rows:
        raise RuntimeError(
            "No valid molecules were parsed from the CSV -- nothing to write."
        )

    if dropped:
        logger.warning(
            "Dropped %d / %d rows with invalid SMILES.", dropped, len(df)
        )

    features_df = pd.DataFrame(feature_rows)
    targets_df = pd.DataFrame(target_rows)
    metadata_df = pd.DataFrame(meta_rows)

    # Reset the integer index so the three frames stay row-aligned.
    features_df.reset_index(drop=True, inplace=True)
    targets_df.reset_index(drop=True, inplace=True)
    metadata_df.reset_index(drop=True, inplace=True)

    os.makedirs(out_dir, exist_ok=True)
    paths = {
        "features": os.path.abspath(
            os.path.join(out_dir, FEATURES_FILENAME)
        ),
        "targets": os.path.abspath(
            os.path.join(out_dir, TARGETS_FILENAME)
        ),
        "metadata": os.path.abspath(
            os.path.join(out_dir, METADATA_FILENAME)
        ),
    }
    features_df.to_csv(paths["features"], index=False)
    targets_df.to_csv(paths["targets"], index=False)
    metadata_df.to_csv(paths["metadata"], index=False)

    logger.info("Wrote %d feature vectors -> %s", len(features_df), paths)
    logger.info("Feature columns: %d fingerprints + %d descriptors",
                MORGAN_BITS, len(DESCRIPTORS))
    return paths


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Featurise a SMILES CSV into Morgan fingerprints + descriptors."
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to the raw CSV (must contain a SMILES column).",
    )
    parser.add_argument(
        "--out-dir", "-o", default="data/processed",
        help="Directory for the processed feature/target files.",
    )
    parser.add_argument(
        "--smiles-col", default="SMILES",
        help="Name of the SMILES column in the input CSV.",
    )
    parser.add_argument(
        "--id-col", default="ID",
        help="Name of the optional identifier column.",
    )
    parser.add_argument(
        "--targets", nargs="+", default=list(DEFAULT_TARGET_COLS),
        help="Candidate target column names (those present are retained).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = _build_parser().parse_args(argv)
    process_and_save(
        csv_path=args.input,
        out_dir=args.out_dir,
        smiles_col=args.smiles_col,
        id_col=args.id_col,
        target_cols=tuple(args.targets),
    )


if __name__ == "__main__":
    main()
