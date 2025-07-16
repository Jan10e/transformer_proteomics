import matplotlib.pyplot as plt
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from IPython.display import HTML, display
from torch.optim.lr_scheduler import CosineAnnealingLR

from transformer_multiomics.data.data_preparation import prepare_data_loaders
from transformer_multiomics.models.transformer import MultiOmicsTransformerFusion
from transformer_multiomics.training.trainer import train_and_evaluate


def optimise_hyperparams(omics_set, datasets, n_trials=30, device=None):
    """
    Optimise hyperparameters for specific combination of omics data

    Parameters:
    -----------
    omics_set : list
        List of omics types to include
    datasets : dict
        Dictionary containing pandas DataFrames for each omics dataset
    n_trials : int
        Number of optimisation trials

    Returns:
    --------
    best_params : dict
        Dictionary of best hyperparameters
    """
    # Ensure device is set
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*80}")
    print(f"Optimising hyperparameters for omics set: {omics_set}")
    print(f"{'='*80}")

    # Prepare data loaders with the selected omics types
    train_loader, val_loader, test_loader, input_dims, output_dim = prepare_data_loaders(omics_set, datasets=datasets)

    # Number of epochs for each trial
    epochs = 50

    # Define the Optuna objective function
    def objective(trial):
        # Suggest hyperparameters
        num_heads = trial.suggest_categorical("num_heads", [4, 8, 16])
        num_layers = trial.suggest_int("num_layers", 2, 6)
        hidden_dim = trial.suggest_int("hidden_dim", 128, 512, step=64)
        dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5, step=0.1)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

        # Additional hyperparameters for ModularMultiOmicsTransformer
        fusion_method = trial.suggest_categorical(
            "fusion_method", ["hierarchical", "late", "gated", "weighted", "cross_attention"]
        )
        activation_function = trial.suggest_categorical("activation_function", ["gelu", "relu"])

        # Instantiate the model with the suggested hyperparameters
        model = MultiOmicsTransformerFusion(
            input_dims=input_dims,
            output_dim=output_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            fusion_method=fusion_method,
            activation_function=activation_function,
        ).to(device)

        # Define loss function and optimiser
        criterion = nn.MSELoss()
        optimiser = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Define the learning rate scheduler
        lr_scheduler = CosineAnnealingLR(optimiser, T_max=epochs, eta_min=1e-6)

        # Training loop
        best_val_loss = float("inf")
        patience = 10
        counter = 0
        early_stop = False
        for epoch in range(epochs):
            if early_stop:
                break

            # Training phase
            model.train()
            running_loss = 0.0
            for inputs_dict, targets in train_loader:
                inputs_dict = {k: v.to(device) for k, v in inputs_dict.items()}
                targets = targets.to(device)

                optimiser.zero_grad()
                outputs = model(inputs_dict)
                loss = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimiser.step()

                running_loss += loss.item() * targets.size(0)

            lr_scheduler.step()

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

            # Early stopping
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                counter = 0
            else:
                counter += 1

            if counter >= patience:
                early_stop = True

        return best_val_loss

    # Run Optuna optimisation
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    # Get best hyperparameters
    best_params = study.best_params
    print("\nBest hyperparameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")

    # Check optimisation history
    print(f"\nBest validation loss: {study.best_value:.6f}")

    return best_params


def progressive_omics_selection(omics_types, datasets, n_trials=30, device=None):
    """
    Progressively select the best combination of input omics types

    Parameters:
    -----------
    omics_types : list
        List of all available omics types
    datasets : dict
        Dictionary containing pandas DataFrames for each omics dataset
    n_trials : int
        Number of optimisation trials per combination

    Returns:
    --------
    all_results : dict
        Dictionary containing results for all tested combinations
    current_set : list
        Best combination of omics types
    """
    # Ensure device is set
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Start with the best single modality (transcriptomics)
    current_set = ["transcriptomics"]
    all_results = {}

    # Get the baseline performance with just transcriptomics
    best_params = optimise_hyperparams(current_set, datasets, n_trials=n_trials)
    results = train_and_evaluate(current_set, best_params, datasets, device)
    all_results[tuple(current_set)] = results
    best_r2 = results["R2"]

    print(f"\nBaseline with {current_set}: R² = {best_r2:.4f}")

    # Track progress in a dataframe for better viewing
    progress_df = pd.DataFrame(
        [{"Step": 0, "Current Set": "+".join(current_set), "R²": best_r2, "RMSE": results["RMSE"]}]
    )

    display(HTML("<h3>Progressive Selection Progress</h3>"))
    display(progress_df)

    # Progressively add one modality at a time
    step = 1
    for _ in range(len(omics_types) - 1):
        candidate_results = {}
        step_df = []

        # Try adding each remaining modality
        for omics in omics_types:
            if omics not in current_set:
                test_set = current_set + [omics]
                print(f"\nTesting omics set: {test_set}")

                # Optimise hyperparameters for this combination
                best_params = optimise_hyperparams(test_set, datasets, n_trials=n_trials)

                # Train and evaluate with best hyperparameters
                results = train_and_evaluate(test_set, best_params, datasets, device)
                candidate_results[omics] = results
                all_results[tuple(test_set)] = results

                # Check Hyperparameters
                print("All results keys:", all_results.keys())
                for key, value in all_results.items():
                    print(f"Key: {key}, Contains 'Hyperparameters': {'Hyperparameters' in value}")

                step_df.append(
                    {
                        "Added Omics": omics,
                        "Test Set": "+".join(test_set),
                        "R²": results["R2"],
                        "RMSE": results["RMSE"],
                        "Improvement": results["R2"] - best_r2,
                    }
                )

                print(f"Adding {omics} → R² = {results['R2']:.4f} (Δ = {results['R2'] - best_r2:+.4f})")

        # Display this step's results
        step_df = pd.DataFrame(step_df)
        display(HTML(f"<h4>Step {step} - Candidate Additions</h4>"))
        display(step_df.sort_values("R²", ascending=False))

        # Find best modality to add
        best_omics = max(candidate_results, key=lambda x: candidate_results[x]["R2"])
        if candidate_results[best_omics]["R2"] > best_r2:
            current_set.append(best_omics)
            best_r2 = candidate_results[best_omics]["R2"]
            print(f"\nAdded {best_omics}. New best set: {current_set}, R²: {best_r2:.4f}")

            # Update progress
            progress_df = pd.concat(
                [
                    progress_df,
                    pd.DataFrame(
                        [
                            {
                                "Step": step,
                                "Current Set": "+".join(current_set),
                                "R²": best_r2,
                                "RMSE": candidate_results[best_omics]["RMSE"],
                                "Added": best_omics,
                            }
                        ]
                    ),
                ]
            )

            display(HTML("<h3>Updated Progressive Selection Progress</h3>"))
            display(progress_df)

            step += 1
        else:
            print(f"\nNo further improvement. Best set: {current_set}, R²: {best_r2:.4f}")
            break

    # Plot progression of performance
    plt.figure(figsize=(10, 6))
    plt.plot(progress_df["Step"], progress_df["R²"], "o-", color="blue", linewidth=2, markersize=8)
    plt.xlabel("Selection Step")
    plt.ylabel("R² Score")
    plt.title("Progression of Model Performance")
    plt.grid(True, alpha=0.3)
    for i, row in progress_df.iterrows():
        if i > 0:  # Skip first point (no addition)
            plt.annotate(
                f"+{row['Added']}",
                (row["Step"], row["R²"]),
                xytext=(10, 0),
                textcoords="offset points",
                fontsize=10,
                fontweight="bold",
            )
    plt.tight_layout()
    plt.show()

    return all_results, current_set, progress_df


# def progressive_omics_selection(
#     omics_types,
#     datasets,
#     n_trials=30,
#     device=None,
#     model_class=None,
#     model_kwargs=None,
#     criterion=None,
#     optimizer_class=None,
#     optimizer_kwargs=None,
#     scheduler_class=None,
#     scheduler_kwargs=None,
#     train_and_evaluate_fn=train_and_evaluate
# ):
#     """
#     Progressively select the best combination of input omics types.
#     Additional arguments are passed to train_and_evaluate for flexibility.
#     """
#     if device is None:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     current_set = ["transcriptomics"]
#     all_results = {}
#
#     best_params = optimise_hyperparams(current_set, datasets, n_trials=n_trials)
#     results = train_and_evaluate_fn(
#         current_set, best_params, datasets, device,
#         model_class=model_class,
#         model_kwargs=model_kwargs,
#         criterion=criterion,
#         optimizer_class=optimizer_class,
#         optimizer_kwargs=optimizer_kwargs,
#         scheduler_class=scheduler_class,
#         scheduler_kwargs=scheduler_kwargs
#     )
#     all_results[tuple(current_set)] = results
#     best_r2 = results["R2"]
#
#     print(f"\nBaseline with {current_set}: R² = {best_r2:.4f}")
#
#     # Track progress in a dataframe for better viewing
#     progress_df = pd.DataFrame([{
#         "Step": 0,
#         "Current Set": "+".join(current_set),
#         "R²": best_r2,
#         "RMSE": results["RMSE"]
#     }])
#
#     display(HTML("<h3>Progressive Selection Progress</h3>"))
#     display(progress_df)
#
#     # Progressively add one modality at a time
#     step = 1
#     for _ in range(len(omics_types) - 1):
#         candidate_results = {}
#         step_df = []
#
#         # Try adding each remaining modality
#         for omics in omics_types:
#             if omics not in current_set:
#                 test_set = current_set + [omics]
#                 print(f"\nTesting omics set: {test_set}")
#
#                 # Optimise hyperparameters for this combination
#                 best_params = optimise_hyperparams(test_set, datasets, n_trials=n_trials)
#
#                 # Train and evaluate with best hyperparameters
#                 results = train_and_evaluate(test_set, best_params, datasets, device)
#                 candidate_results[omics] = results
#                 all_results[tuple(test_set)] = results
#
#                 step_df.append({
#                     "Added Omics": omics,
#                     "Test Set": "+".join(test_set),
#                     "R²": results["R2"],
#                     "RMSE": results["RMSE"],
#                     "Improvement": results["R2"] - best_r2
#                 })
#
#                 print(f"Adding {omics} → R² = {results['R2']:.4f} (Δ = {results['R2'] - best_r2:+.4f})")
#
#         # Display this step's results
#         step_df = pd.DataFrame(step_df)
#         display(HTML(f"<h4>Step {step} - Candidate Additions</h4>"))
#         display(step_df.sort_values("R²", ascending=False))
#
#         # Find best modality to add
#         best_omics = max(candidate_results, key=lambda x: candidate_results[x]["R2"])
#         if candidate_results[best_omics]["R2"] > best_r2:
#             current_set.append(best_omics)
#             best_r2 = candidate_results[best_omics]["R2"]
#             print(f"\nAdded {best_omics}. New best set: {current_set}, R²: {best_r2:.4f}")
#
#             # Update progress
#             progress_df = pd.concat([
#                 progress_df,
#                 pd.DataFrame([{
#                     "Step": step,
#                     "Current Set": "+".join(current_set),
#                     "R²": best_r2,
#                     "RMSE": candidate_results[best_omics]["RMSE"],
#                     "Added": best_omics
#                 }])
#             ])
#
#             display(HTML("<h3>Updated Progressive Selection Progress</h3>"))
#             display(progress_df)
#
#             step += 1
#         else:
#             print(f"\nNo further improvement. Best set: {current_set}, R²: {best_r2:.4f}")
#             break
#
#     # Plot progression of performance
#     plt.figure(figsize=(10, 6))
#     plt.plot(progress_df["Step"], progress_df["R²"], "o-", color="blue", linewidth=2, markersize=8)
#     plt.xlabel("Selection Step")
#     plt.ylabel("R² Score")
#     plt.title("Progression of Model Performance")
#     plt.grid(True, alpha=0.3)
#     for i, row in progress_df.iterrows():
#         if i > 0:  # Skip first point (no addition)
#             plt.annotate(f"+{row['Added']}",
#                          (row["Step"], row["R²"]),
#                          xytext=(10, 0),
#                          textcoords="offset points",
#                          fontsize=10,
#                          fontweight="bold")
#     plt.tight_layout()
#     plt.show()
#
#     return all_results, current_set, progress_df
