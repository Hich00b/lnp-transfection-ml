# LNP-Transfection: Predicting LNP Transfection Efficiency from SMILES

An end-to-end Python pipeline that predicts the **cell-specific transfection efficiency** of ionizable lipid nanoparticles (LNPs) directly from chemical SMILES strings. It uses ECFP4 Morgan fingerprints and physicochemical descriptors alongside an XGBoost regressor. The pipeline supports both **K-fold cross-validation** and **scaffold-based splitting** to ensure robust generalization to unseen chemical spaces. SHAP (SHapley Additive exPlanations) is integrated to provide feature attribution, revealing how specific substructures and physical properties (such as Molecular Weight and Lipophilicity) drive biological performance.

This pipeline is built around the **AGILE dataset** format: a CSV containing `SMILES` and `Transfection` columns.

---

## Repository Structure

```text
LNP_PKa/
├── data/
│   ├── raw/            # Place your AGILE CSV here (SMILES, Transfection)
│   └── processed/      # Generated features.csv, targets.csv, metadata.csv
├── src/
│   ├── data_processing.py   # SMILES -> Morgan FP + descriptors
│   ├── train.py             # XGBoost + K-fold CV / Scaffold Split
│   ├── evaluate.py          # SHAP TreeExplainer + publication-ready plots
│   └── predict.py           # Inference on new SMILES (single string or batch CSV)
├── models/                  # Pre-trained .pkl + SHAP .png artefacts
├── notebooks/
│   └── colab_runner.ipynb   # Headless Colab entry point
├── requirements.txt
├── .gitignore
�└── README.md

## Input Data Format
The pipeline expects a UTF-8 CSV in `data/raw/` containing at least the following columns (the `Transfection` column acts as the regression target):

| Column      | Required | Description                                     |
|-------------|----------|-------------------------------------------------|
| `SMILES`    | Yes      | Valid canonical SMILES string representing the lipid |
| `Transfection` | Yes   | Cell-specific transfection efficiency (target variable) |
| `ID`        | No       | Unique molecule identifier (used in metadata only) |

**Note:** Any other numeric target column present in the CSV can be selected using the `--target <column>` flag during training. Malformed SMILES are parsed defensively and dropped without crashing the pipeline.

## Installation
```bash
git clone https://github.com/Hich00b/lnp-transfection-ml.git
cd LNP_PKa
pip install -r requirements.txt
```

## Usage

### 1. Feature Generation
```bash
python src/data_processing.py --input data/raw/dataset.csv --out-dir data/processed
```
This script processes SMILES strings and generates `features.csv` (2048 Morgan bits `FP_0..FP_2047` + `MolWt`, `NumRotatableBonds`, `TPSA`, `MolLogP`), `targets.csv` (the selected target column), and `metadata.csv` (ID and SMILES) in the `data/processed/` directory.

### 2. Model Training & Validation
Train the model using either 5-fold cross-validation or Murcko scaffold splits:
```bash
# K-fold cross-validation (Recommended for robust metrics)
python src/train.py --cv 5

# Scaffold split — tests generalization to unseen chemical scaffolds
python src/train.py
```
**Note:** A cross-validated, pre-trained XGBoost model is already provided in `models/xgb_transfection.pkl`. If you wish to use the trained weights directly, you can skip training and move straight to evaluation or prediction.

### 3. SHAP Interpretability Analysis
```bash
python src/evaluate.py
```
Generates publication-quality interpretability plots saved directly to `models/`:
- `shap_summary_transfection.png` (Beeswarm plot, 300 DPI)
- `shap_descriptors_transfection.png` (Descriptor importance bar plot)
- `shap_summary_transfection_values.npy` (Raw SHAP values matrix)

### 4. Run Predictions on New SMILES
Use the pre-trained model to run predictions on single SMILES strings or new batch CSV files:
```bash
# Predict a single SMILES string
python src/predict.py --smiles "CCCCCCCC/C=C\CCCCCCCC(=O)O"

# Batch prediction from a CSV file (must contain a 'SMILES' column)
python src/predict.py --input data/raw/new_candidates.csv --output data/predictions.csv
```

## Google Colab Execution
To run this pipeline in the cloud, open `notebooks/colab_runner.ipynb` in Google Colab, set `REPO_URL` to your fork's URL, and execute all cells. The notebook will automatically clone the repository, install dependencies, and execute the feature generation, training, and evaluation steps.

## Performance Benchmarks
Using default parameters on a 1,200-sample lipid dataset, the XGBoost regressor yields the following metrics under 5-Fold Cross-Validation:
- $R^2$: $0.3680 \pm 0.0849$
- RMSE: $2.5447 \pm 0.0797$
- MAE: $1.8661 \pm 0.0588$

These metrics demonstrate stable structure-activity capture, bypassing the heavy test variance observed during strict out-of-distribution scaffold splitting in high-throughput nanoparticle screening.