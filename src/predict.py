import argparse
import joblib
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem

def extract_features_from_smiles(smiles_str):
    """Converts a SMILES string into 2048 Morgan bits + 4 physical descriptors."""
    mol = Chem.MolFromSmiles(smiles_str)
    if mol is None:
        return None
    
    # 2048 Morgan Fingerprint bits
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    fp_array = np.array(fp)
    
    # Physical Descriptors
    mol_wt = Descriptors.MolWt(mol)
    num_rot = Descriptors.NumRotatableBonds(mol)
    tpsa = Descriptors.TPSA(mol)
    logp = Descriptors.MolLogP(mol)
    
    descriptors = np.array([mol_wt, num_rot, tpsa, logp])
    return np.concatenate([fp_array, descriptors])

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
        feats = extract_features_from_smiles(args.smiles)
        if feats is None:
            print("Error: Invalid SMILES string provided.")
        else:
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
            feats = extract_features_from_smiles(sm)
            if feats is not None:
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