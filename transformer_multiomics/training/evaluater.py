import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import display
from scipy.stats import pearsonr
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm

from transformer_multiomics.config import MODEL_PATH
from transformer_multiomics.data.data_preparation import prepare_data_loaders
from transformer_multiomics.models.transformer import MultiOmicsTransformerFusion


def evaluate_model(model, test_loader, device, is_transformer=False, model_name="Model"):
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for x_dict, targets in tqdm(test_loader, desc=f"Evaluating {model_name}"):
            targets = targets.to(device)
            x_dict = {k: v.to(device) for k, v in x_dict.items()}

            outputs = model(x_dict)
            if is_transformer:
                outputs = outputs[0]  # unpack predictions from (preds, attn_weights)

            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Combine all batches
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Compute evaluation metrics
    r2 = r2_score(all_targets.flatten(), all_predictions.flatten())
    rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
    mae = mean_absolute_error(all_targets.flatten(), all_predictions.flatten())
    evs = explained_variance_score(all_targets.flatten(), all_predictions.flatten())

    print(f"{model_name} Performance:")
    print(f"R-squared (R²): {r2:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Explained Variance Score: {evs:.4f}")

    return {
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "evs": evs,
        "predictions": all_predictions,
        "targets": all_targets,
    }


def evaluate_model_optim(model, loader, criterion, device, return_attention_weights=False):
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for inputs_dict, targets in loader:
            inputs_dict = {k: v.to(device) for k, v in inputs_dict.items()}
            targets = targets.to(device)
            outputs = model(inputs_dict)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * targets.size(0)
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    return avg_loss, (all_predictions, all_targets)


def summarise_results(all_results, best_set):
    # Create summary table
    summary = []
    for omics_set, results in all_results.items():
        hyperparams = results.get("Hyperparameters", {})
        summary.append(
            {
                "Omics Set": "+".join(omics_set),
                "R2": results["R2"],
                "RMSE": results["RMSE"],
                "MAE": results["MAE"],
                "EVS": results["EVS"],
                "Fusion Method": hyperparams.get("fusion_method", "N/A"),
                "Activation": hyperparams.get("activation_function", "N/A"),
            }
        )

    # Sort by R2 score
    summary_sorted = sorted(summary, key=lambda x: x["R2"], reverse=True)

    print("\n\n===== SUMMARY OF ALL COMBINATIONS =====")
    print(f"{'Omics Set':<30} {'R²':<10} {'RMSE':<10} {'MAE':<10} {'EVS':<10} {'Fusion':<15} {'Activation':<10}")
    print("-" * 90)
    for entry in summary_sorted:
        print(
            f"{entry['Omics Set']:<30} {entry['R2']:<10.4f} {entry['RMSE']:<10.4f} {entry['MAE']:<10.4f} "
            f"{entry['EVS']:<10.4f} {entry['Fusion Method']:<15} {entry['Activation']:<10}"
        )

    # Plot performance metrics for different combinations
    plt.figure(figsize=(12, 8))

    # Extract data for plotting
    combinations = ["+".join(comb) for comb in all_results]
    r2_values = [results["R2"] for results in all_results.values()]
    rmse_values = [results["RMSE"] for results in all_results.values()]

    # Sort by performance
    sorted_indices = np.argsort(r2_values)[::-1]
    combinations = [combinations[i] for i in sorted_indices]
    r2_values = [r2_values[i] for i in sorted_indices]
    rmse_values = [rmse_values[i] for i in sorted_indices]

    x = np.arange(len(combinations))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(14, 8))
    ax2 = ax1.twinx()

    # Plot R^2 (higher is better)
    bars1 = ax1.bar(x - width / 2, r2_values, width, label="R²", color="blue", alpha=0.7)
    ax1.set_ylabel("R² Score (higher is better)", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.set_ylim(0, max(r2_values) * 1.1)

    # Plot RMSE (lower is better)
    bars2 = ax2.bar(x + width / 2, rmse_values, width, label="RMSE", color="red", alpha=0.7)
    ax2.set_ylabel("RMSE (lower is better)", color="red")
    ax2.tick_params(axis="y", labelcolor="red")
    ax2.set_ylim(0, max(rmse_values) * 1.1)

    # Highlight the best combination
    best_idx = combinations.index("+".join(best_set))
    bars1[best_idx].set_color("darkblue")
    bars1[best_idx].set_alpha(1.0)
    bars2[best_idx].set_color("darkred")
    bars2[best_idx].set_alpha(1.0)

    ax1.set_xticks(x)
    ax1.set_xticklabels(combinations, rotation=45, ha="right")
    ax1.set_title("Performance Comparison of Different Omics Combinations")

    # Add legends
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.show()

    # Plot learning curves for the best combination
    best_results = all_results[tuple(best_set)]
    plt.figure(figsize=(10, 6))
    plt.plot(best_results["Train_Losses"], label="Training Loss", color="blue")
    plt.plot(best_results["Val_Losses"], label="Validation Loss", color="red")
    plt.axvline(x=best_results["Early_Stop_Epoch"], color="green", linestyle="--", label="Early Stopping Point")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (MSE)")
    plt.title(f"Learning Curves for Best Omics Combination: {'+'.join(best_set)}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Analyse fusion methods and activations
    fusion_methods = [results.get("Hyperparameters", {}).get("fusion_method") for results in all_results.values()]
    fusion_methods = [m for m in fusion_methods if m is not None]

    if len(fusion_methods) > 0:
        plt.figure(figsize=(10, 6))
        fusion_counts = pd.Series(fusion_methods).value_counts()
        plt.bar(fusion_counts.index, fusion_counts.values)
        plt.title("Frequency of Fusion Methods in Top Performing Models")
        plt.xlabel("Fusion Method")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()

    return summary_sorted


def analyse_feature_level_performance(best_omics_set, best_params, datasets, device=None):
    """
    Analyse which individual genes/features were predicted best by the model

    Parameters:
    -----------
    best_omics_set : list
        The best combination of omics types
    best_params : dict
        Best hyperparameters for the model
    datasets : dict
        Dictionary containing the datasets

    Returns:
    --------
    feature_metrics : pd.DataFrame
        DataFrame with performance metrics for each feature
    """
    # Ensure device is set
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'-'*80}")
    print(f"Analysing feature-level performance for best model: {best_omics_set}")
    print(f"{'-'*80}")

    # Prepare data loaders - FIXED: Pass the datasets parameter
    train_loader, val_loader, test_loader, input_dims, output_dim = prepare_data_loaders(
        best_omics_set, datasets=datasets, test_size=0.2, batch_size=64
    )

    # Get the feature names from the proteomics data
    target_feature_names = datasets["proteomics"].columns.tolist()[1:]  # Skip the ID column

    # Separate out parameters from model vs. training process
    model_params = {
        k: v for k, v in best_params.items() if k not in ["learning_rate", "weight_decay", "batch_size", "epochs"]
    }
    model = MultiOmicsTransformerFusion(input_dims=input_dims, output_dim=output_dim, **model_params).to(device)

    # Load the best model weights
    model_name = f"best_model_{'_'.join(best_omics_set)}.pth"
    model.load_state_dict(torch.load(MODEL_PATH / model_name), strict=False)
    model.eval()

    # Collect predictions and targets
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for inputs_dict, targets in test_loader:
            inputs_dict = {k: v.to(device) for k, v in inputs_dict.items()}
            targets = targets.to(device)

            outputs = model(inputs_dict)
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate metrics for each feature
    feature_metrics = []
    for i in range(output_dim):
        y_true = all_targets[:, i]
        y_pred = all_predictions[:, i]

        # Calculate metrics
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        corr, p_value = pearsonr(y_true, y_pred)

        # Store in dictionary
        feature_metrics.append(
            {
                "Feature": target_feature_names[i] if i < len(target_feature_names) else f"Feature_{i}",
                "R2": r2,
                "RMSE": rmse,
                "MAE": mae,
                "Correlation": corr,
                "P_value": p_value,
            }
        )

    # Sort by R2 (descending)
    feature_metrics = pd.DataFrame(feature_metrics)
    feature_metrics = feature_metrics.sort_values("R2", ascending=False)
    feature_metrics.to_csv("feature_level_performance.csv", index=False)

    # Display top and bottom performing features
    print("\nTop 10 best predicted features:")
    display(feature_metrics.head(10))

    print("\nBottom 10 worst predicted features:")
    display(feature_metrics.tail(10))

    # Plot distribution of R2 scores
    plt.figure(figsize=(12, 6))
    plt.hist(feature_metrics["R2"], bins=50, alpha=0.7)
    plt.axvline(
        feature_metrics["R2"].mean(), color="red", linestyle="--", label=f"Mean R² = {feature_metrics['R2'].mean():.4f}"
    )
    plt.axvline(
        feature_metrics["R2"].median(),
        color="green",
        linestyle="--",
        label=f"Median R² = {feature_metrics['R2'].median():.4f}",
    )
    plt.title("Distribution of R² Scores Across Features")
    plt.xlabel("R² Score")
    plt.ylabel("Number of Features")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Plot a few example scatter plots for best and worst features
    num_plots = min(3, len(feature_metrics))
    plt.figure(figsize=(15, 5))

    # Best features
    for i in range(num_plots):
        plt.subplot(1, num_plots, i + 1)
        feature = feature_metrics.iloc[i]["Feature"]
        idx = target_feature_names.index(feature) if feature in target_feature_names else i

        plt.scatter(all_targets[:, idx], all_predictions[:, idx], alpha=0.5)
        plt.plot(
            [min(all_targets[:, idx]), max(all_targets[:, idx])],
            [min(all_targets[:, idx]), max(all_targets[:, idx])],
            "r--",
        )
        plt.title(f"{feature}\nR² = {feature_metrics.iloc[i]['R2']:.4f}")
        plt.xlabel("True Value")
        plt.ylabel("Predicted Value")

    plt.tight_layout()
    plt.show()

    return feature_metrics


def compute_metrics(targets, predictions):
    return {
        "R2": r2_score(targets.flatten(), predictions.flatten()),
        "RMSE": np.sqrt(mean_squared_error(targets, predictions)),
        "MAE": mean_absolute_error(targets, predictions),
        "EVS": explained_variance_score(targets.flatten(), predictions.flatten()),
    }
