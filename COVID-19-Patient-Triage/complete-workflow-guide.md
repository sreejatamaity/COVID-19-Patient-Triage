# Detailed Workflow Guide: COVID-19 Severity Prediction

This document outlines the step-by-step process of the predictive modeling pipeline executed by `main.py`.

## Phase 1: Setup and Data Preparation (main.py, config.json, data_preprocessing.py)

1.  **Configuration Loading**: `main.py` loads necessary constants, paths, and model configurations from `config.json`.
2.  **Data Loading**: `data_preprocessing.py` reads the `covid_symptoms_severity_prediction.csv` file.
3.  **Target Creation**: A new binary target feature, `severity_flag`, is created: `1` if the patient was hospitalized, admitted to ICU, or died, and `0` otherwise.
4.  **Train/Test Split**: The data is split into training and testing sets using stratified sampling to maintain the original class ratio in the subsets.

---

## Phase 2: Feature Engineering and Preprocessing (feature_engineering.py)

1.  **Feature Grouping**: Features are categorized as numerical (`age`), categorical (`gender`, `vaccination_status`), and binary (symptoms/comorbidities).
2.  **Transformer Pipeline**: A `ColumnTransformer` is created:
    * Numerical features (`age`) are **Standard Scaled**.
    * Categorical features are **One-Hot Encoded**.
    * Binary features (symptoms/comorbidities) are **Standard Scaled**, and then **Principal Component Analysis (PCA)** is applied to reduce their dimensionality while retaining 90% of the variance.
3.  **Application**: The pipeline is fitted on the training data and then applied to both the training and testing sets. The fitted preprocessor is saved to the disk.

---

## Phase 3: Model Training and Hyperparameter Tuning (models.py)

1.  **Model Instantiation**: Logistic Regression, Random Forest Classifier, and SVC models are initialized. **Class weights** are applied to mitigate class imbalance.
2.  **Hyperparameter Tuning**: A cross-validated search (using `GridSearchCV` or similar) is performed on each model to find the best hyperparameters defined in `config.json`. The **F1-score** is used as the primary metric to optimize, as it balances Precision and Recall, crucial for medical diagnostics where false negatives and false positives are equally important concerns.
3.  **Best Model Selection**: The model with the highest average F1-score across its cross-validation folds is selected as the best performer.

---

## Phase 4: Robust Evaluation (evaluation.py)

1.  **Bootstrap Validation**: The best model is evaluated using a **Bootstrap Validation** technique. This involves repeatedly sampling the test set (with replacement) to create many smaller "replicate" test sets.
2.  **Metric Calculation**: Performance metrics (**Accuracy**, **Precision**, **Recall**, **F1-Score**, **ROC-AUC**, and **AUPRC**) are calculated on each bootstrap sample.
3.  **Confidence Intervals (CIs)**: The 95% Confidence Intervals for each metric are calculated from the distribution of scores across the bootstrap samples, providing a robust estimate of the model's expected performance range on unseen data.

---

## Phase 5: Finalization and Deployment (main.py)

1.  **Saving Assets**: The fitted preprocessor and the final best model object are saved to the `results/` directory using `joblib`.
2.  **Summary Generation**: A comprehensive summary of the pipeline execution, best model performance, and key takeaways is compiled in `project-implementation-summary.md`.