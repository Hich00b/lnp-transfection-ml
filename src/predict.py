import argparse
import joblib
import pandas as pd
import numpy as np
import warnings
from rdkit import Chem

# Suppress XGBoost serialization warnings for a cleaner terminal output
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
warnings.filterwarnings("ignore", category=UserWarning, module="pickle")

# Import feature extraction functions from data_processing module
from data_processing import smiles_to_mol, featurize_mol

def main():
    parser = argparse.ArgumentParser(description="Run transfection predictions using pre-trained model.")
    parser.add_argument("--smiles", type=str, help="Single SMILES string for instant prediction.")
    parser.add_argument("--input", type=str, help="Path to input CSV containing a SMILES column.")
    parser.add_argument("--output", type=str, default="data/predictions.csv", help="Path to save output CSV.")
    parser.add_argument("--model", type=str, default="models/xgb_transfection.pkl", help="Path to trained model file.")
    args = parser.parse_args()

    # Load trained model
    model = joblib.load(args.model)

    if args.smiles:
        mol = smiles_to_mol(args.smiles)
        if mol is None:
            print("Error: Invalid SMILES string provided.")
        else:
            features_dict = featurize_mol(mol)
            # Convert dictionary to numpy array in the correct order matching the training data
            feature_values = [features_dict[key] for key in sorted(features_dict.keys())
                            if key.startswith('FP_') or key in ['MolWt', 'NumRotatableBonds', 'TPSA', 'MolLogP']]
            # Ensure we have the right order: descriptors first, then fingerprints sorted by index
            ordered_features = []
            # Add descriptors in order
            for desc in ['MolWt', 'NumRotatableBonds', 'TPSA', 'MolLogP']:
                ordered_features.append(features_dict[desc])
            # Add fingerprints in order
            for i in range(2048):
                ordered_features.append(features_dict[f'FP_{i}'])
            feats = np.array(ordered_features, dtype=np.float32)
            prediction = model.predict([feats])[0]
            print(f"\nSMILES: {args.smiles}")
            print(f"Predicted Transfection Efficiency: {prediction:.4f}\n")

    elif args.input:
        df = pd.read_csv(args.input)
        if "SMILES" not in df.columns:
            raise ValueError("Input CSV must contain a 'SMILES' column.")

        features_list = []
        valid_indices = []

        for idx, sm in enumerate(df["SMILES"]):
            mol = smiles_to_mol(sm)
            if mol is not None:
                features_dict = featurize_mol(mol)
                # Convert dictionary to numpy array in the correct order matching the training data
                ordered_features = []
                # Add descriptors in order
                for desc in ['MolWt', 'NumRotatableBonds', 'TPSA', 'MolLogP']:
                    ordered_features.append(features_dict[desc])
                # Add fingerprints in order
                for i in range(2048):
                    ordered_features.append(features_dict[f'FP_{i}'])
                feats = np.array(ordered_features, dtype=np.float32)
                features_list.append(feats)
                valid_indices.append(idx)

        features_matrix = np.array(features_list)
        preds = model.predict(features_matrix)

        out_df = df.iloc[valid_indices].copy()
        out_df["Predicted_Transfection"] = preds
        out_df.to_csv(args.output, index=False)
        print(f"\nSuccessfully generated predictions for {len(preds)} valid SMILES.")
        print(f"Saved results to -> {args.output}\n")

    else:
        print("Please provide either --smiles '<string>' or --input '<path/to/file.csv>'.")

if __name__ == "__main__":
    main()