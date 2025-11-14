# COVID-19 Severity Prediction Project 🏥

This repository contains the complete workflow for predicting the **severity** of COVID-19 outcomes (hospitalization, ICU admission, or mortality) using patient demographic, symptom, and comorbidity data.

## Project Structure

The project employs a modular, single-level directory structure, with configuration, documentation, and all Python modules located in the root directory.

| File Name | Role |
| :--- | :--- |
| **`main.py`** | Main entry point and pipeline orchestrator. |
| **`config.json`** | Stores model hyperparameters and file paths. |
| **`data_preprocessing.py`** | Handles data loading, target creation (`severity_flag`), and train/test split. |
| **`feature_engineering.py`** | Manages feature scaling, one-hot encoding, and **Principal Component Analysis (PCA)** for dimensionality reduction. |
| **`models.py`** | Defines and trains **Logistic Regression, Random Forest, and SVM** models with class weighting. |
| **`evaluation.py`** | Performs robust **Bootstrap Validation** to calculate performance metrics (Accuracy, F1-Score, ROC-AUC, AUPRC) and Confidence Intervals. |
| **`utils.py`** | Helper functions for file I/O and summary generation. |
| **`project-implementation-summary.md`** | Generated output file detailing final results. |

## Models Used

The three selected models utilize **class weighting** (`class_weight='balanced'`) to address the significant class imbalance inherent in predicting severe outcomes.

## Prerequisites

1.  Python 3.8+
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Workflow Summary

The pipeline follows five phases:

1.  **Data Preparation**: Load data and create the binary `severity_flag`.
2.  **Feature Engineering**: Apply Standard Scaling, One-Hot Encoding, and **PCA** on symptom/comorbidity features.
3.  **Modeling**: Train and tune three models (LR, RF, SVM), optimizing for **F1-Score**.
4.  **Evaluation**: Assess the best model's performance stability using **Bootstrap Validation** and report **AUPRC** and **ROC-AUC**.
5.  **Finalization**: Save the best model and preprocessor, and generate the final summary document.

## Running the Pipeline

Ensure you are in the project's root directory:

```bash
python main.py
