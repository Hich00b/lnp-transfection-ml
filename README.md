# LNP-Transfection: Predicting LNP Transfection Efficiency from SMILES

An end-to-end Python pipeline that predicts the **cell-specific transfection efficiency** of ionizable lipid nanoparticles (LNPs) directly from chemical SMILES strings. It uses ECFP4 Morgan fingerprints and physicochemical descriptors alongside an XGBoost regressor. The pipeline supports **K-fold cross-validation**, **Murcko scaffold splitting**, and **buffered similarity splitting** to ensure robust generalization to unseen chemical spaces. The buffered similarity splitting approach enforces a minimum distance buffer between training and test molecules, providing mathematically guaranteed separation. SHAP (SHapley Additive exPlanations) is integrated to provide feature attribution, revealing how specific substructures and physical properties (such as Molecular Weight and Lipophilicity) drive biological performance.

This pipeline is built around the **AGILE dataset** format: a CSV containing `SMILES` and `Transfection` columns.

## Quick Start: Predict a Molecule

New users who only want predictions can skip model training entirely:
```bash
git clone https://github.com/Hich00b/lnp-transfection-ml.git
cd lnp-transfection-ml/
pip install -r requirements.txt
python src/predict.py --smiles "CCCCCCCC/C=C\CCCCCCCC(=O)O" --reference data/raw/dataset.csv --model models/xgb_transfection.pkl
```
The output includes an applicability domain flag; predictions flagged "OUTSIDE validated domain" should be treated as unreliable extrapolation, not a confident number. For a no-install alternative, see `notebooks/quick_predict.ipynb`. The rest of this README, from "Reproducing the Paper's Results" onward, covers the full pipeline used to produce the shipped model and is not required reading for users who only want predictions.

---

## Repository Structure

```text
lnp-transfection-ml/
├── data/
│   ├── raw/            # Place your AGILE CSV here (SMILES, Transfection)
│   └── processed/      # Generated features.csv, targets.csv, metadata.csv
├── src/
│   ├── data_processing.py   # SMILES -> Morgan FP + descriptors
│   ├── train.py             # XGBoost + K-fold CV / Scaffold Split / Buffered LOCO
│   ├── evaluate.py          # SHAP TreeExplainer + publication-ready plots
│   └── predict.py           # Inference on new SMILES (single string or batch CSV)
├── models/                  # Pre-trained .pkl + SHAP .png artefacts
├── notebooks/
│   ├── colab_runner.ipynb   # Full pipeline: feature gen, training, evaluation, prediction
│   └── quick_predict.ipynb  # Prediction only, no training -- fastest way to try the model
├── requirements.txt
├── .gitignore
└── README.md
```

## Input Data Format

The pipeline expects a UTF-8 CSV in `data/raw/` containing at least the following columns (the `Transfection` column acts as the regression target):

| Column         | Required | Description                                             |
|----------------|----------|---------------------------------------------------------|
| `SMILES`       | Yes      | Valid canonical SMILES string representing the lipid    |
| `Transfection` | Yes      | Cell-specific transfection efficiency (target variable) |
| `ID`           | No       | Unique molecule identifier (used in metadata only)      |

**Note:** Any other numeric target column present in the CSV can be selected using the `--target <column>` flag during training. Malformed SMILES are parsed defensively and dropped without crashing the pipeline.

## Installation

```bash
git clone https://github.com/Hich00b/lnp-transfection-ml.git
cd lnp-transfection-ml/
pip install -r requirements.txt
```

## Usage

### 1. Feature Generation

```bash
python src/data_processing.py --input data/raw/dataset.csv --out-dir data/processed
```

This script processes SMILES strings and generates `features.csv` (2048 Morgan bits `FP_0..FP_2047` + `MolWt`, `NumRotatableBonds`, `TPSA`, `MolLogP`), `targets.csv` (the selected target column), and `metadata.csv` (ID and SMILES) in the `data/processed/` directory.

### 2. Model Training & Validation

Train the model using 5-fold cross-validation, Murcko scaffold splits, a single buffered similarity split, or buffered leave-one-cluster-out (LOCO):

```bash
# Random 5-fold cross-validation -- measures IN-DISTRIBUTION performance
# (structurally familiar compounds)
python src/train.py --cv 5

# Murcko scaffold split — tests generalization to unseen chemical scaffolds
python src/train.py

# Single buffered similarity split — fast sanity check only; reports a
# single-split metric that does NOT match the paper's pooled numbers
# (see Performance Benchmarks below)
python src/train.py --split-method buffered --distance-cutoff 0.1

# Buffered leave-one-cluster-out (LOCO), 50 folds -- measures
# OUT-OF-DISTRIBUTION performance (structurally novel chemotypes).
# This is the PRIMARY protocol reported in Table 1 of the manuscript.
python src/train.py --cv-clusters --distance-cutoff 0.1 --min-cluster-size 10
```

Random 5-fold CV and buffered LOCO answer different questions rather than competing for "most recommended": CV tells you how well the model fits compounds similar to what it has already seen, while buffered LOCO tells you how well it generalizes to structurally novel chemotypes, the more realistic setting for prioritizing new candidate lipids. See Performance Benchmarks below for why buffered LOCO is the protocol used for the paper's headline out-of-distribution result.

**Parameters for buffered splitting:**
- `--distance-cutoff`: Maximum Tanimoto distance allowed between train and test molecules (default: 0.1)
- `--min-cluster-size`: Minimum group size for leave-one-group-out evaluation (default: 10)

**Note:** A cross-validated, pre-trained XGBoost model is already provided in `models/xgb_transfection.pkl`. If you wish to use the trained weights directly, you can skip training and move straight to evaluation or prediction. Regardless of which option above is run, the saved model is always refit on the full 1200-compound dataset before deployment (see "Which training option should I use?" below).

### 3. SHAP Interpretability Analysis

```bash
python src/evaluate.py
```

Generates publication-quality interpretability plots saved directly to `models/`:
- `shap_summary_transfection.png` (Beeswarm plot, 300 DPI)
- `shap_descriptors_transfection.png` (Descriptor importance bar plot)
- `shap_summary_transfection_values.npy` (Raw SHAP values matrix)

**Important:** this script evaluates SHAP on the held-out test set recorded in `models/split_indices.csv`, written by the most recent `train.py` run. If you last ran `train.py` with `--cv-clusters` (LOCO), that split file has no single held-out set (LOCO uses 50 separate folds instead), and `evaluate.py` will print a warning and fall back to evaluating SHAP on the entire dataset rather than a genuine held-out set. To get genuine held-out SHAP values, run `train.py --split-method buffered` (or the default scaffold split) immediately before `evaluate.py`, without running any other training command in between. Pass `--strict` to make this a hard error instead of a warning, useful when you want a guarantee that only genuine held-out SHAP is produced:

```bash
python src/evaluate.py --strict
```

### 4. Run Predictions on New SMILES

Use the pre-trained model to run predictions on single SMILES strings or new batch CSV files:

```bash
# Predict a single SMILES string
python src/predict.py --smiles "CCCCCCCC/C=C\CCCCCCCC(=O)O" --reference data/raw/dataset.csv --model models/xgb_transfection.pkl

# Batch prediction from a CSV file (must contain a 'SMILES' column)
python src/predict.py --input data/raw/new_candidates.csv --reference data/raw/dataset.csv --model models/xgb_transfection.pkl --output data/predictions.csv
```

## Google Colab Execution

Two notebooks are provided, depending on what you need:

- **`notebooks/quick_predict.ipynb`** — prediction only, using the pre-trained model already committed in the repo. No feature generation, no training. Fastest way to try the model on your own candidate SMILES.
- **`notebooks/colab_runner.ipynb`** — the full pipeline: clones the repo, installs dependencies, generates features, trains the model (buffered LOCO by default), runs SHAP interpretability, and finishes with single-SMILES and batch prediction cells so you can try the model on new candidates at the end of the same run.

Open either notebook in Google Colab, set `REPO_URL` to your fork's URL if needed, and execute all cells.

## Performance Benchmarks

Using default parameters on the 1,200-sample AGILE lipid library, pooled out-of-fold metrics (not per-fold averages, see the paper's Methods 5.5 for why pooled metrics are the primary statistic):

**Random 5-Fold Cross-Validation (in-distribution performance):**
- $R^2$: $0.4901 \pm 0.0490$
- RMSE: $2.2925 \pm 0.1205$
- MAE: $1.7234 \pm 0.0819$

**Buffered Leave-One-Cluster-Out, 50 folds (out-of-distribution performance, PRIMARY metric):**
- $R^2$: $0.3820$
- RMSE: $2.5789$
- MAE: $1.9743$
- This is the protocol reported in Table 1 of the manuscript. The exact value differs slightly from the manuscript's reported R²=0.354 due to environment-dependent floating-point and library-version differences between this run and the original Google Colab environment; both values are self-consistent within their respective environments.

**A note on the single buffered split option** (`--split-method buffered`, no `--cv-clusters` flag): this reports a single-split validation metric ($R^2=0.2100$, RMSE=$2.7504$, MAE=$2.0618$, test=237/train=814/dropped=149, 15.5%). Given the fix described below, this does not correspond to a different deployed model, all three training options now produce the same full-dataset-refit deployed model, only the reported validation metric differs. This single-split R² (~0.21) is noisier than either pooled metric above and should not be treated as a benchmark result.

## Which training option should I use?

For most users, running the buffered LOCO option (`--cv-clusters`) once to produce `models/xgb_transfection.pkl` is sufficient, and it matches the protocol used for the paper's reported results. Regardless of which training option is run, the deployed `model.pkl` is always refit on the complete 1200-compound dataset before being saved, so users predicting new molecules do not need to worry about which training option was used to produce the currently-saved model.

What matters most for prediction reliability on a new candidate is the applicability domain flag in `predict.py`'s output, not which training option produced the model. Always check the `in_domain` flag before trusting a prediction.

**Note:** A cross-validated, pre-trained XGBoost model is already provided in `models/xgb_transfection.pkl`. If you wish to use the trained weights directly, you can skip training and move straight to evaluation or prediction.