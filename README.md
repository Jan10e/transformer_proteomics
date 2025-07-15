<!--
-*- coding: utf-8 -*-

 Author: Jantine Broek <jantine.broek@simmons-simmons.com>
 License: MIT
-->

# Transformer Proteomics


## Table of Contents
1. [General Info](#general-info)
2. [Model and Data](#model-and-data)
3. [Build and Run](#build-and-run)
4. [License](#license)



<a name="general-info"></a>
## General Info

This repository contains a project for predictive modelling of proteomics data using a Transformer-based architecture. The goal is to predict proteomics data using multi-omics data (transcriptomics, methylation, metabolomics, cnv) as input. The project compares a baseline MLP model with the multi-omic Transformer models to evaluate their performance and suitability for this task.


### Repository Structure

```
.
├── data/                               # Directory for input data files
├── models/                             # Directory for model files (.pth)
├── results/                            # Directory for prediction results
├── EMBL_transformer_explore.ipynb      # Data, model building, tuning, etc
├── EMBL_transformer_best_model.ipynb   # Best model, feature analysis
└── README.md                           
```


<a name="model-and-data"></a>
## Model and Data

### Task Type and Model Decision
This task is a regression task, where one type of omics data is predicted using others. Both VAEs and Transformers are suitable choices: VAEs excel with noisy data and handle missing values well, while Transformers are powerful for capturing complex interactions between features through their self-attention mechanism.

I selected a Transformer model because the multi-head attention can learn different types of relationships between omics data features and capturing various correlations that might exist. The self-attention mechanism allows the model to identify important feature interactions, which I think are important in multi-omics data integration. While VAEs would also be suitable, particularly for handling noise and missing values, the potential complex interactions between omics data features made Transformers my preferred choice.

### Data
The aim is to use multiple input omics datasets to predict proteomics data. Input omics data was selected by iteratively testing combinations to determine which best predicted proteomics. This indicated that transcriptomics data only as the optimal input combination for proteomics prediction.



<a name="build-and-run"></a>
## Build and Run

### How to Build

This repository uses Git Large File Storage (Git LFS) to handle the large `.pth` model files. To clone and use this repository:

1. Install Git LFS if you haven't already:
   ```bash
   # For Ubuntu/Debian
   apt-get install git-lfs

   # For macOS with Homebrew
   brew install git-lfs

   # For Windows with Chocolatey
   choco install git-lfs
   ```

2. Enable Git LFS:
   ```bash
   git lfs install
   ```

3. Ensure you have Python 3.8 or higher installed.
4. Clone the repository:
   ```bash
    git clone https://github.com/Jan10e/transformer_multiomics.git
    cd transformer_multiomics
   ```

5. Pull the LFS files:
   ```bash
   git lfs pull
   ```

6. Create a (conda) environment. For example:

    ```bash
    conda create -n transformer_multiomics python=3.11
    conda activate transformer_multiomics
    ```

7. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

### How to Run

#### Required Data Files

The model expects the following data files in the `data/` directory:

- `20231023_092657_imputed_methylation.csv`
- `20231023_092657_imputed_metabolomics.csv`
- `20231023_092657_imputed_proteomics.csv`
- `20231023_092657_imputed_transcriptomics.csv`
- `20231023_092657_imputed_copynumber.csv`

#### Build Transformer Model

-  Run the **EMBL_transformer_explore.ipynb** to see the development of the Transformer model 
    ```bash
    EMBL_transformer_explore.ipynb
    ```

This script will:
1. Load the omics data
2. Inspect the data
3. Create base MLP model
4. Develop Transformer model
5. Hyperparameter tuning
6. Progressive input omics selection


#### Running the Best-Performing Model

-  Run the **EMBL_transformer_best_model.ipynb** to see the development of the Transformer model 
    ```bash
    EMBL_transformer_best_model.ipynb
    ```
The script will:
1. Load the omics data
2. Load the pre-trained model
3. Make predictions for proteomics values
4. Save the predictions to `results/predicted_proteomics.csv`
5. Perform feature analysis



## Model Details

The model is a transformer-based architecture that:
- Uses a "gated" fusion method to combine different omics modalities
- Was trained primarily on transcriptomics data to predict proteomics features
- Implements self-attention mechanisms to capture complex relationships between features