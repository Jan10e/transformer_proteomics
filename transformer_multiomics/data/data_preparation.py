import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


class MultiOmicsDataset(Dataset):
    """
    PyTorch Dataset for multi-omics data.
    Each sample is a dictionary of omics features and optionally a target value.
    """

    def __init__(self, X_dict: dict[str, np.ndarray], y: np.ndarray = None):
        """
        Args:
            X_dict: Dictionary mapping omics names to numpy arrays of features.
            y: Array of target values (optional).
        """
        self.X_dict: dict[str, torch.FloatTensor] = {k: torch.FloatTensor(v) for k, v in X_dict.items()}
        self.y: torch.FloatTensor = torch.FloatTensor(y) if y is not None else None

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(next(iter(self.X_dict.values())))

    def __getitem__(
        self, idx: int
    ) -> tuple[dict[str, torch.FloatTensor], torch.FloatTensor] | dict[str, torch.FloatTensor]:
        """
        Get a single sample by index.
        Returns:
            Tuple of (omics features dict, target tensor) if `y` is not None,
            or just the omics features dict if `y` is None.
        """
        if self.y is not None:
            return {k: v[idx] for k, v in self.X_dict.items()}, self.y[idx]
        else:
            return {k: v[idx] for k, v in self.X_dict.items()}


def prepare_data_loaders(
    omics_set: list[str],
    datasets: dict[str, pd.DataFrame] | None = None,
    target_dataset: str = "proteomics",
    test_size: float = 0.2,
    batch_size: int = 32,
    random_state: int = 42,
    val_split: float = 0.1,
) -> tuple[DataLoader, DataLoader | None, DataLoader, dict[str, int], int]:
    """
    Prepare PyTorch DataLoaders for multi-omics integration.

    Args:
        omics_set: List of omics dataset names to use as input.
        datasets: Dictionary mapping omics names to pandas DataFrames.
        target_dataset: Name of the target omics dataset.
        test_size: Fraction of data to use for testing.
        batch_size: Batch size for DataLoaders.
        random_state: Random seed for reproducibility.
        val_split: Fraction of training data to use for validation.

    Returns:
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data (or None).
        test_loader: DataLoader for test data.
        input_dims: Dictionary of input dimensions per omics.
        output_dim: Output dimension of the target.
    """
    if datasets is None:
        try:
            datasets = globals()["datasets"]
        except KeyError:
            raise ValueError("No datasets provided and no global `datasets` dictionary found")

    # Extract input features for each omics
    X_dict: dict[str, np.ndarray] = {}
    for dataset_name in omics_set:
        if dataset_name in datasets:
            # Exclude the first column (assumed to be ID)
            X_dict[dataset_name] = datasets[dataset_name].iloc[:, 1:].values
        else:
            raise ValueError(f"Dataset `{dataset_name}` not found in provided datasets")

    # Extract target values (excluding ID column)
    y: np.ndarray = datasets[target_dataset].iloc[:, 1:].values
    input_dims: dict[str, int] = {dataset_name: X_dict[dataset_name].shape[1] for dataset_name in omics_set}
    output_dim: int = y.shape[1]

    n_samples: int = len(y)
    indices: np.ndarray = np.arange(n_samples)
    # Split into train+val and test
    train_val_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=random_state)

    if val_split > 0:
        # Further split train+val into train and val
        train_indices, val_indices = train_test_split(
            train_val_indices, test_size=val_split / (1 - test_size), random_state=random_state
        )
        X_val_dict: dict[str, np.ndarray] = {k: v[val_indices] for k, v in X_dict.items()}
        y_val: np.ndarray = y[val_indices]
        val_dataset = MultiOmicsDataset(X_val_dict, y_val)
        val_loader: DataLoader | None = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    else:
        train_indices = train_val_indices
        val_loader = None

    # Prepare train and test datasets/loaders
    X_train_dict: dict[str, np.ndarray] = {k: v[train_indices] for k, v in X_dict.items()}
    X_test_dict: dict[str, np.ndarray] = {k: v[test_indices] for k, v in X_dict.items()}
    y_train: np.ndarray = y[train_indices]
    y_test: np.ndarray = y[test_indices]

    train_dataset = MultiOmicsDataset(X_train_dict, y_train)
    test_dataset = MultiOmicsDataset(X_test_dict, y_test)
    train_loader: DataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader: DataLoader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, input_dims, output_dim


def prepare_test_data(data, omics_set, batch_size=32):
    """
    Prepare test data for prediction.

    Parameters:
    -----------
    data : dict
        Dictionary of DataFrames containing omics data.
    omics_set : list
        List of omics types to use for prediction.
    batch_size : int
        Batch size for the DataLoader.

    Returns:
    --------
    data_loader : DataLoader
        DataLoader containing the test data.
    input_dims : dict
        Dictionary mapping each omics type to its dimensionality.
    """
    # Prepare input data dictionary
    X_dict = {}
    for omics_type in omics_set:
        if omics_type in data:
            # Exclude patient ID column (assuming it's the first column)
            X_dict[omics_type] = data[omics_type].iloc[:, 1:].values
        else:
            raise ValueError(f"Omics type '{omics_type}' not found in the provided data")

    # Create input dimensions dictionary
    input_dims = {omics_type: X.shape[1] for omics_type, X in X_dict.items()}

    # Create dataset and dataloader
    dataset = MultiOmicsDataset(X_dict, y=None)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return data_loader, input_dims
