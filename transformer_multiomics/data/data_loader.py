import pandas as pd

from transformer_multiomics.config import DATA_PATH, DATA_PREFIX


def ensure_directories_exist(*paths):
    """
    Ensure that the given directories exist. Create them if they don't.
    Print status for each directory.
    """
    for path in paths:
        if path.exists():
            print(f"Directory exists: {path}")
        else:
            path.mkdir(parents=True)
            print(f"Directory created: {path}")


def load_datasets():
    """
    Load datasets from the data directory using the prefix defined in config.py.

    Returns:
        dict: Dictionary containing loaded datasets.
    """
    datasets = {
        "methylation": pd.read_csv(DATA_PATH / f"{DATA_PREFIX}imputed_methylation.csv"),
        "metabolomics": pd.read_csv(DATA_PATH / f"{DATA_PREFIX}imputed_metabolomics.csv"),
        "proteomics": pd.read_csv(DATA_PATH / f"{DATA_PREFIX}imputed_proteomics.csv"),
        "transcriptomics": pd.read_csv(DATA_PATH / f"{DATA_PREFIX}imputed_transcriptomics.csv"),
        "cnv": pd.read_csv(DATA_PATH / f"{DATA_PREFIX}imputed_copynumber.csv"),
    }
    return datasets
