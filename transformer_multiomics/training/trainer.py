import torch
from tqdm import tqdm
from pathlib import Path

import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score


from transformer_multiomics.config import MODEL_PATH, RESULT_PATH, DATA_PATH
from transformer_multiomics.models.transformer import ModularMultiOmicsTransformer
from transformer_multiomics.data.data_preparation import prepare_data_loaders


def train_model(model, train_loader, test_loader, criterion, optimizer, device, 
                epochs=100, patience=15, model_name="model", save_path=None, 
                log_interval=10, return_attention_weights=False):
    """
    General training loop for PyTorch models with early stopping.
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run on (cuda/cpu)
        epochs: Maximum number of epochs
        patience: Early stopping patience
        model_name: Name for saving the model
        save_path: Path to save the best model (optional)
        log_interval: How often to print progress
        return_attention_weights: Whether model returns attention weights (for transformer)
        
    Returns:
        tuple: (trained_model, train_losses, test_losses, best_loss) or 
               (trained_model, train_losses, test_losses, best_loss, attention_weights) 
               if return_attention_weights=True
    """
    
    # Early stopping setup
    best_loss = float("inf")
    counter = 0
    best_model_state = None
    
    # Track losses
    train_losses = []
    test_losses = []
    
    print(f"Starting training for {model_name}...")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for x_dict, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            # Move data to device
            x_dict = {k: v.to(device) for k, v in x_dict.items()}
            targets = targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            
            # Handle models that return attention weights
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
        
        # Evaluation phase
        model.eval()
        test_loss = 0.0
        all_attention_weights = [] if return_attention_weights else None
        
        with torch.no_grad():
            for x_dict, targets in test_loader:
                x_dict = {k: v.to(device) for k, v in x_dict.items()}
                targets = targets.to(device)
                
                # Forward pass
                if return_attention_weights:
                    outputs, attn_weights = model(x_dict)
                    all_attention_weights.append(attn_weights.cpu())
                else:
                    outputs = model(x_dict)
                
                loss = criterion(outputs, targets)
                test_loss += loss.item() * targets.size(0)
        
        epoch_test_loss = test_loss / len(test_loader.dataset)
        test_losses.append(epoch_test_loss)
        
        # Early stopping check
        if epoch_test_loss < best_loss:
            best_loss = epoch_test_loss
            counter = 0
            best_model_state = model.state_dict().copy()
        else:
            counter += 1
        
        # Progress logging
        if epoch % log_interval == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_train_loss:.4f} | "
                  f"Test Loss: {epoch_test_loss:.4f} | Early Stopping: {counter}/{patience}")
        
        # Early stopping
        if counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            model.load_state_dict(best_model_state)
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # If we need attention weights, collect them from the best model
    final_attention_weights = None
    if return_attention_weights:
        print("Collecting attention weights from best model...")
        model.eval()
        final_attention_weights = []
        with torch.no_grad():
            for x_dict, targets in test_loader:
                x_dict = {k: v.to(device) for k, v in x_dict.items()}
                targets = targets.to(device)
                
                outputs, attn_weights = model(x_dict)
                final_attention_weights.append(attn_weights.cpu())
    
    # Save model if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        model_file = save_path / f"best_{model_name}_model.pt"
        torch.save(model.state_dict(), model_file)
        print(f"Best model saved to {model_file}")
    
    # Return attention weights if collected
    if return_attention_weights and final_attention_weights:
        return model, train_losses, test_losses, best_loss, final_attention_weights
    else:
        return model, train_losses, test_losses, best_loss
    

def train_and_evaluate(omics_set, best_params, datasets, device):
    """
    Train and evaluate the model with the best hyperparameters for specific omics combination
    
    Parameters:
    -----------
    omics_set : list
        List of omics types to include
    best_params : dict
        Dictionary of best hyperparameters
    datasets : dict
        Dictionary containing pandas DataFrames for each omics dataset
        
    Returns:
    --------
    results : dict
        Dictionary containing performance metrics and training history
    """
    print(f"\n{'-'*80}")
    print(f"Training final model for omics set: {omics_set}")
    print(f"{'-'*80}")
    
    epochs = 200
    
    train_loader, val_loader, test_loader, input_dims, output_dim = prepare_data_loaders(omics_set, datasets=datasets)
    
    # Use the best hyperparameters to instantiate the model
    model = ModularMultiOmicsTransformer(
        input_dims=input_dims,
        output_dim=output_dim,
        num_heads=best_params["num_heads"],
        num_layers=best_params["num_layers"],
        hidden_dim=best_params["hidden_dim"],
        dropout_rate=best_params["dropout_rate"],
        fusion_method=best_params["fusion_method"],
        activation_function=best_params["activation_function"]
    ).to(device)
    
    # Loss function and optimiser
    criterion = nn.MSELoss()
    optimiser = optim.AdamW(
        model.parameters(),
        lr=best_params["learning_rate"],
        weight_decay=best_params["weight_decay"]
    )
    
    # Learning rate scheduler
    lr_scheduler = CosineAnnealingLR(
        optimiser,
        T_max=epochs,
        eta_min=1e-6
    )
    
    # Training loop
    best_val_loss = float("inf")
    patience = 20  # Slightly more patience for final model
    counter = 0
    train_losses = []
    val_losses = []
    best_model_state = None
    early_stop_epoch = 0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for inputs_dict, targets in train_loader:
            inputs_dict = {k: v.to(device) for k, v in inputs_dict.items()}
            targets = targets.to(device)
            
            optimiser.zero_grad()
            outputs = model(inputs_dict)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimiser.step()
            
            running_loss += loss.item() * targets.size(0)
        
        lr_scheduler.step()
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs_dict, targets in val_loader:
                inputs_dict = {k: v.to(device) for k, v in inputs_dict.items()}
                targets = targets.to(device)
                
                outputs = model(inputs_dict)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * targets.size(0)
        
        epoch_val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        # Early stopping check
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            counter = 0
            best_model_state = model.state_dict().copy()
            early_stop_epoch = epoch
        else:
            counter += 1
            
        if counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        # Print progress periodically
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}/{epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")
            print(f"Learning rate: {optimiser.param_groups[0]['lr']:.6f}")
    
    # Load the best model state
    model.load_state_dict(best_model_state)
    
    # Save the trained model
    model_name = f"best_model_{'_'.join(omics_set)}.pth"
    torch.save(model.state_dict(), MODEL_PATH / model_name)
    
    # Evaluate model on test set
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs_dict, targets in test_loader:
            inputs_dict = {k: v.to(device) for k, v in inputs_dict.items()}
            targets = targets.to(device)
            
            outputs = model(inputs_dict)
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    # Concatenate all batches
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Calculate overall metrics
    r2 = r2_score(all_targets.flatten(), all_predictions.flatten())
    rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
    mae = mean_absolute_error(all_targets, all_predictions)
    evs = explained_variance_score(all_targets.flatten(), all_predictions.flatten())
    
    # Store results
    results = {
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae, 
        "EVS": evs,
        "Best_Val_Loss": best_val_loss,
        "Hyperparameters": best_params,
        "Train_Losses": train_losses,
        "Val_Losses": val_losses,
        "Early_Stop_Epoch": early_stop_epoch
    }
    
    print(f"\nPerformance for omics set {omics_set}:")
    print(f"R-squared (R²): {r2:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Explained Variance Score: {evs:.4f}")
    
    # Plot learning curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss", color="blue")
    plt.plot(val_losses, label="Validation Loss", color="red")
    plt.axvline(x=early_stop_epoch, color="green", linestyle="--", label="Early Stopping Point")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (MSE)")
    plt.title(f"Learning Curves for Omics Combination: {'+'.join(omics_set)}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return results
