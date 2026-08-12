#!/usr/bin/env python3
"""Test script to verify that the refactored predict.py produces identical features."""

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator

# Modern (non-deprecated) Morgan/ECFP4 generator -- created once and reused.
MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=2, fpSize=2048
)

def extract_features_from_smiles_original(smiles_str):
    """Original implementation from predict.py."""
    mol = Chem.MolFromSmiles(smiles_str)
    if mol is None:
        return None

    # 2048 Morgan Fingerprint bits (using modern rdFingerprintGenerator)
    fp_array = MORGAN_GENERATOR.GetFingerprintAsNumPy(mol).astype(np.uint8)

    # Physical Descriptors
    mol_wt = Descriptors.MolWt(mol)
    num_rot = Descriptors.NumRotatableBonds(mol)
    tpsa = Descriptors.TPSA(mol)
    logp = Descriptors.MolLogP(mol)

    descriptors = np.array([mol_wt, num_rot, tpsa, logp])
    return np.concatenate([fp_array, descriptors])  # [fingerprints, descriptors]

def extract_features_from_smiles_new(smiles_str):
    """New implementation using data_processing functions."""
    from src.data_processing import smiles_to_mol, featurize_mol

    mol = smiles_to_mol(smiles_str)
    if mol is None:
        return None

    features_dict = featurize_mol(mol)
    # Convert dictionary to numpy array in the correct order matching the training data
    ordered_features = []
    # Add descriptors in order
    for desc in ['MolWt', 'NumRotatableBonds', 'TPSA', 'MolLogP']:
        ordered_features.append(features_dict[desc])
    # Add fingerprints in order
    for i in range(2048):
        ordered_features.append(features_dict[f'FP_{i}'])
    return np.array(ordered_features, dtype=np.float32)  # [descriptors, fingerprints]

# Test with a few SMILES from the dataset
test_smiles = [
    "CC(C)CC1=C(C=C(C=C1)O)O",
    "CCCCCCCCCCCCNC(=O)C(CCCCCOC(=O)CCCCCCCC)NCCN(C)C",
    "CCO",
    "CC(C)O",
]

print("Testing feature equivalence...")
print("=" * 60)

for smi in test_smiles:
    print(f"SMILES: {smi}")

    # Get features from both implementations
    feats_orig = extract_features_from_smiles_original(smi)
    feats_new = extract_features_from_smiles_new(smi)

    if feats_orig is None or feats_new is None:
        print("  Error: One or both implementations failed to generate features")
        continue

    print(f"  Original shape: {feats_orig.shape}, New shape: {feats_new.shape}")
    print(f"  Original dtype: {feats_orig.dtype}, New dtype: {feats_new.dtype}")

    # Check if they're equal
    if np.array_equal(feats_orig, feats_new):
        print("  PASS: Features are identical")
    else:
        print("  FAIL: Features differ!")
        # Show first few differences
        diff_idx = np.where(feats_orig != feats_new)[0]
        if len(diff_idx) > 0:
            print(f"    First difference at index {diff_idx[0]}: {feats_orig[diff_idx[0]]} vs {feats_new[diff_idx[0]]}")
        print(f"    Max difference: {np.max(np.abs(feats_orig - feats_new))}")

    print()

print("=" * 60)
print("Note: The original implementation returns [fingerprints, descriptors]")
print("      The new implementation returns [descriptors, fingerprints]")
print("      To be equivalent, we need to adjust the order in one of them.")