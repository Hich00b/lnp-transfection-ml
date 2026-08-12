"""LNP-Transfection: Prediction of transfection efficiency for ionizable
lipid nanoparticles (LNPs) from SMILES strings (AGILE dataset format).

Submodules
----------
data_processing
    SMILES parsing, ECFP4 Morgan fingerprints + physicochemical descriptors.
train
    XGBoost regression with scaffold-based train/test splitting.
evaluate
    SHAP TreeExplainer analysis and publication-quality plots.
"""

__all__ = ["data_processing", "train", "evaluate"]
__version__ = "0.1.0"
