# LNP-Transfection: Predicting LNP transfection efficiency from SMILES

An end-to-end Python pipeline that predicts the **cell-specific transfection
efficiency** of ionizable lipid nanoparticles (LNPs) directly from chemical
SMILES strings, using ECFP4 Morgan fingerprints plus physicochemical
descriptors and an XGBoost regressor trained with **scaffold-based
splitting** (to test generalisation to unseen chemical space). SHAP provides
feature attribution explaining which substructures and descriptors drive
transfection efficiency.

The pipeline is built around the **AGILE dataset** format: a CSV with
`SMILES` and `Transfection` columns.

## Repository structure

```
LNP_PKa/
├── data/
│   ├── raw/            # place your AGILE CSV here (SMILES, Transfection)
│   └── processed/      # generated features.csv, targets.csv, metadata.csv
├── src/
│   ├── data_processing.py   # SMILES -> Morgan FP + descriptors
│   ├── train.py             # XGBoost + scaffold split / K-fold CV
│   └── evaluate.py          # SHAP TreeExplainer + publication plots
├── models/                  # saved .pkl + SHAP .png artefacts
├── notebooks/
│   └── colab_runner.ipynb   # headless Colab entry point
├── requirements.txt
├── .gitignore
└── README.md
```

## Input data format (AGILE)

The pipeline expects a UTF-8 CSV in `data/raw/` with **at least** these
columns (the `Transfection` column holds the regression target):

| Column           | Required | Description                                          |
|------------------|----------|------------------------------------------------------|
| `SMILES`         | **yes**  | Valid canonical SMILES string                        |
| `Transfection`   | **yes**  | Cell-specific transfection efficiency (regression target) |
| `ID`             | no       | Unique molecule identifier (used in metadata only)   |

Any other numeric target column present in the CSV can be selected with
`--target <column>` if you need to train against a different property.
Malformed SMILES are parsed defensively and dropped (count logged); they do
not crash the pipeline.

## Installation

```bash
git clone https://github.com/Hich00b/lnp-transfection-ml.git
cd LNP_PKa
pip install -r requirements.txt
```

## Usage

### 1. Featurise the raw dataset

```bash
python src/data_processing.py --input data/raw/dataset.csv --out-dir data/processed
```
Writes `features.csv` (2048 Morgan bits `FP_0..FP_2047` + `MolWt`,
`NumRotatableBonds`, `TPSA`, `MolLogP`), `targets.csv` (the `Transfection`
column), and `metadata.csv` (ID + SMILES, used for scaffold splitting) to
`data/processed/`.

### 2. Train the transfection model

```bash
# Scaffold split (default) — tests generalisation to new chemical scaffolds
python src/train.py

# K-fold cross-validation, then refit a final model on all data
python src/train.py --cv 5
```
`Transfection` is the default target, so no `--target` flag is needed. Prints
R^2, RMSE and MAE and saves `models/xgb_transfection.pkl` (plus
`split_indices.csv`).

### 3. SHAP interpretability

```bash
python src/evaluate.py
```
Produces `models/shap_summary_transfection.png` (beeswarm, 300 DPI) and
`models/shap_descriptors_transfection.png` (mean |SHAP| bar plot for the
four interpretable descriptors), plus the raw
`shap_summary_transfection_values.npy`.

### Run on Google Colab

Open `notebooks/colab_runner.ipynb` in Colab, set `REPO_URL` to your fork,
then run all cells. It clones the repo, installs dependencies, and executes
the three pipeline steps headlessly. The notebook's `RAW_CSV` defaults to
`data/raw/dataset.csv` and relies on the transfection defaults (no extra
flags).

## Pipeline

```
raw CSV ──► data_processing.py ──► features.csv + targets.csv + metadata.csv
  (SMILES, Transfection)                  │
                                          ▼
                                   train.py (scaffold split / K-fold)
                                          │
                            ┌─────────────┴──────────────┐
                            ▼                            ▼
                  xgb_transfection.pkl          metrics (R², RMSE, MAE)
                            │
                            ▼
                     evaluate.py (SHAP)
                            │
                   ┌────────┴──────────────┐
                   ▼                       ▼
  shap_summary_transfection.png   shap_descriptors_transfection.png
```

## Notes

- **Scaffold split** uses Murcko scaffolds (`rdkit.Chem.Scaffolds.MurckoScaffold`);
  the largest scaffold groups fill the training set (~80%) and the held-out
  scaffolds form the test set (~20%), so test molecules share no scaffold
  with training. This is a stricter, more chemically honest generalisation
  test than random splitting.
- All plots use publication-quality styling: large fonts, no top/right
  spines, 300 DPI export.
- The code contains **no dummy data** — supply your own AGILE CSV in
  `data/raw/`.
