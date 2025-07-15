from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformer_multiomics.data.data_preparation import prepare_data_loaders
from transformer_multiomics.models.transformer import MultiOmicsTransformerFusion
from transformer_multiomics.training.evaluater import compute_metrics
from transformer_multiomics.utils.plots import plot_loss_curves


def evaluate_model_attn_weight(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    return_attention_weights: bool = False,
) -> tuple[float, list[torch.Tensor] | None]:
    """
    Evaluate the model on a given DataLoader.

    Args:
        model: PyTorch model.
        data_loader: DataLoader for evaluation.
        criterion: Loss function.
        device: Device to run on.
        return_attention_weights: Whether to return attention weights.

    Returns:
        Tuple of (average loss, list of attention weights or None).
    """
    model.eval()
    total_loss = 0.0
    all_attention_weights = [] if return_attention_weights else None

    with torch.no_grad():
        for x_dict, targets in data_loader:
            x_dict = {k: v.to(device) for k, v in x_dict.items()}
            targets = targets.to(device)
            result = model(x_dict)
            if return_attention_weights:
                outputs, attn_weights = result if isinstance(result, tuple) else (result, None)
                all_attention_weights.append(attn_weights.cpu())
            else:
                outputs = result if not isinstance(result, tuple) else result[0]
            loss = criterion(outputs, targets)
            total_loss += loss.item() * targets.size(0)

    avg_loss = total_loss / len(data_loader.dataset)
    return avg_loss, all_attention_weights


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epochs: int = 100,
    patience: int = 15,
    model_name: str = "model",
    save_path: str | Path | None = None,
    log_interval: int = 10,
    return_attention_weights: bool = False,
) -> (
    tuple[nn.Module, list[float], list[float], float]
    | tuple[nn.Module, list[float], list[float], float, list[torch.Tensor]]
):
    """
    General training loop for PyTorch models with early stopping.

    Args:
        model: PyTorch model to train.
        train_loader: DataLoader for training data.
        test_loader: DataLoader for test data.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to run on (cuda/cpu).
        epochs: Maximum number of epochs.
        patience: Early stopping patience.
        model_name: Name for saving the model.
        save_path: Path to save the best model (optional).
        log_interval: How often to print progress.
        return_attention_weights: Whether model returns attention weights.

    Returns:
        Trained model, train losses, test losses, best loss, and optionally attention weights.
    """
    best_loss = float("inf")
    counter = 0
    best_model_state = None
    train_losses: list[float] = []
    test_losses: list[float] = []

    print(f"Starting training for {model_name}...")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for x_dict, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            x_dict = {k: v.to(device) for k, v in x_dict.items()}
            targets = targets.to(device)
            optimizer.zero_grad()
            if return_attention_weights:
                outputs, _ = model(x_dict)
            else:
                outputs = model(x_dict)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * targets.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # Evaluate on test set
        epoch_test_loss, _ = evaluate_model_attn_weight(
            model, test_loader, criterion, device, return_attention_weights=False
        )
        test_losses.append(epoch_test_loss)

        # Early stopping logic
        if epoch_test_loss < best_loss:
            best_loss = epoch_test_loss
            counter = 0
            best_model_state = model.state_dict().copy()
        else:
            counter += 1

        if epoch % log_interval == 0 or epoch == epochs - 1:
            print(
                f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_train_loss:.4f} | "
                f"Test Loss: {epoch_test_loss:.4f} | Early Stopping: {counter}/{patience}"
            )

        if counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            model.load_state_dict(best_model_state)
            break

    model.load_state_dict(best_model_state)

    # Optionally collect attention weights from best model
    final_attention_weights = None
    if return_attention_weights:
        print("Collecting attention weights from best model...")
        _, final_attention_weights = evaluate_model_attn_weight(
            model, test_loader, criterion, device, return_attention_weights=True
        )

    # Save model if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        model_file = save_path / f"best_{model_name}_model.pt"
        torch.save(model.state_dict(), model_file)
        print(f"Best model saved to {model_file}")

    if return_attention_weights and final_attention_weights:
        return model, train_losses, test_losses, best_loss, final_attention_weights
    else:
        return model, train_losses, test_losses, best_loss


# TODO: remove this when refactoring is complete
# def train_and_evaluate(
#     omics_set: List[str],
#     best_params: Dict[str, Any],
#     datasets: Dict[str, Any],
#     device: torch.device
# ) -> Dict[str, Any]:
#     """
#     Train and evaluate the model with the best hyperparameters for a specific omics combination.
#
#     Args:
#         omics_set: List of omics types to include.
#         best_params: Dictionary of best hyperparameters.
#         datasets: Dictionary containing pandas DataFrames for each omics dataset.
#         device: Device to run on.
#
#     Returns:
#         Dictionary containing performance metrics and training history.
#     """
#     print(f"\n{'-'*80}")
#     print(f"Training final model for omics set: {omics_set}")
#     print(f"{'-'*80}")
#
#     epochs = 200
#     train_loader, val_loader, test_loader, input_dims, output_dim = prepare_data_loaders(
#         omics_set, datasets=datasets
#     )
#
#     # Instantiate model with best hyperparameters
#     model = MultiOmicsTransformerFusion(
#         input_dims=input_dims,
#         output_dim=output_dim,
#         num_heads=best_params["num_heads"],
#         num_layers=best_params["num_layers"],
#         hidden_dim=best_params["hidden_dim"],
#         dropout_rate=best_params["dropout_rate"],
#         fusion_method=best_params["fusion_method"],
#         activation_function=best_params["activation_function"]
#     ).to(device)
#
#     criterion = nn.MSELoss()
#     optimiser = optim.AdamW(
#         model.parameters(),
#         lr=best_params["learning_rate"],
#         weight_decay=best_params["weight_decay"]
#     )
#     lr_scheduler = CosineAnnealingLR(
#         optimiser,
#         T_max=epochs,
#         eta_min=1e-6
#     )
#
#     best_val_loss = float("inf")
#     patience = 20
#     counter = 0
#     train_losses: List[float] = []
#     val_losses: List[float] = []
#     best_model_state = None
#     early_stop_epoch = 0
#
#     for epoch in range(epochs):
#         model.train()
#         running_loss = 0.0
#
#         for inputs_dict, targets in train_loader:
#             inputs_dict = {k: v.to(device) for k, v in inputs_dict.items()}
#             targets = targets.to(device)
#             optimiser.zero_grad()
#             outputs = model(inputs_dict)
#             loss = criterion(outputs, targets)
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             optimiser.step()
#             running_loss += loss.item() * targets.size(0)
#
#         lr_scheduler.step()
#         epoch_train_loss = running_loss / len(train_loader.dataset)
#         train_losses.append(epoch_train_loss)
#
#         # Validation phase
#         epoch_val_loss, _ = evaluate_model(
#             model, val_loader, criterion, device, return_attention_weights=False
#         )
#         val_losses.append(epoch_val_loss)
#
#         if epoch_val_loss < best_val_loss:
#             best_val_loss = epoch_val_loss
#             counter = 0
#             best_model_state = model.state_dict().copy()
#             early_stop_epoch = epoch
#         else:
#             counter += 1
#
#         if counter >= patience:
#             print(f"Early stopping at epoch {epoch+1}")
#             break
#
#         if epoch % 10 == 0 or epoch == epochs - 1:
#             print(f"Epoch {epoch}/{epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")
#             print(f"Learning rate: {optimiser.param_groups[0]['lr']:.6f}")
#
#     model.load_state_dict(best_model_state)
#     model_name = f"best_model_{'_'.join(omics_set)}.pth"
#     torch.save(model.state_dict(), MODEL_PATH / model_name)
#
#     # Evaluate on test set
#     model.eval()
#     all_predictions = []
#     all_targets = []
#     with torch.no_grad():
#         for inputs_dict, targets in test_loader:
#             inputs_dict = {k: v.to(device) for k, v in inputs_dict.items()}
#             targets = targets.to(device)
#             outputs = model(inputs_dict)
#             all_predictions.append(outputs.cpu().numpy())
#             all_targets.append(targets.cpu().numpy())
#
#     all_predictions = np.concatenate(all_predictions, axis=0)
#     all_targets = np.concatenate(all_targets, axis=0)
#
#     # Calculate metrics
#     r2 = r2_score(all_targets.flatten(), all_predictions.flatten())
#     rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
#     mae = mean_absolute_error(all_targets, all_predictions)
#     evs = explained_variance_score(all_targets.flatten(), all_predictions.flatten())
#
#     results = {
#         "R2": r2,
#         "RMSE": rmse,
#         "MAE": mae,
#         "EVS": evs,
#         "Best_Val_Loss": best_val_loss,
#         "Hyperparameters": best_params,
#         "Train_Losses": train_losses,
#         "Val_Losses": val_losses,
#         "Early_Stop_Epoch": early_stop_epoch
#     }
#
#     print(f"\nPerformance for omics set {omics_set}:")
#     print(f"R-squared (R²): {r2:.4f}")
#     print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
#     print(f"Mean Absolute Error (MAE): {mae:.4f}")
#     print(f"Explained Variance Score: {evs:.4f}")
#
#     # Plot learning curves
#     plt.figure(figsize=(10, 6))
#     plt.plot(train_losses, label="Training Loss", color="blue")
#     plt.plot(val_losses, label="Validation Loss", color="red")
#     plt.axvline(x=early_stop_epoch, color="green", linestyle="--", label="Early Stopping Point")
#     plt.xlabel("Epochs")
#     plt.ylabel("Loss (MSE)")
#     plt.title(f"Learning Curves for Omics Combination: {'+'.join(omics_set)}")
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.show()
#
#     return results


def build_model(omics_set, best_params, input_dims, output_dim, device):
    model_param_keys = [
        "num_heads",
        "num_layers",
        "hidden_dim",
        "dropout_rate",
        "fusion_method",
        "activation_function",
    ]
    model_args = {k: best_params[k] for k in model_param_keys if k in best_params}
    model = MultiOmicsTransformerFusion(input_dims=input_dims, output_dim=output_dim, **model_args).to(device)
    return model


def build_optimizer(model, best_params):
    lr = best_params.get("learning_rate", 1e-3)
    weight_decay = best_params.get("weight_decay", 0)
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def build_scheduler(optimizer, epochs):
    return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)


def train_and_evaluate(
    omics_set: list[str],
    best_params: dict[str, Any],
    datasets: dict[str, Any],
    device: torch.device,
    epochs: int = 50,
    patience: int = 10,
    save_path: str | Path | None = None,
) -> dict[str, Any]:
    print(f"\n{'-'*80}")
    print(f"Training final model for omics set: {omics_set}")

    train_loader, val_loader, test_loader, input_dims, output_dim = prepare_data_loaders(omics_set, datasets=datasets)
    model = build_model(omics_set, best_params, input_dims, output_dim, device)
    criterion = nn.MSELoss()
    optimizer = build_optimizer(model, best_params)
    scheduler = build_scheduler(optimizer, epochs)

    trained_model, train_losses, val_losses, best_val_loss = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
        patience=patience,
        model_name=f"best_model_{'_'.join(omics_set)}",
        save_path=save_path,
    )

    from transformer_multiomics.training.evaluater import evaluate_model_optim

    test_loss, _ = evaluate_model_optim(model=trained_model, loader=test_loader, criterion=nn.MSELoss(), device=device)

    all_predictions = []
    all_targets = []
    trained_model.eval()
    with torch.no_grad():
        for inputs_dict, targets in test_loader:
            inputs_dict = {k: v.to(device) for k, v in inputs_dict.items()}
            targets = targets.to(device)
            outputs = trained_model(inputs_dict)
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # get metrics
    metrics = compute_metrics(all_targets, all_predictions)
    metrics.update(
        {
            "Best_Val_Loss": best_val_loss,
            "Train_Losses": train_losses,
            "Val_Losses": val_losses,
        }
    )

    # plot loss curves
    plot_loss_curves(train_losses, val_losses, patience)

    return metrics
