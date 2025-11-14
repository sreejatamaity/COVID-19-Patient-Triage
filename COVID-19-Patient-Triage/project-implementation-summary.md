# COVID-19 Severity Prediction Project Implementation Summary

This document summarizes the execution and key findings of the severity prediction pipeline.

## 1. Data and Target Definition

* **Dataset Size**: 3000 records.
* **Target Creation**: The binary target `severity_flag` was created by setting the label to `1` if any of the following fields were true: `hospitalized`, `icu_admission`, or `mortality`. This captures the severity of the outcome.
* **Class Imbalance**: The dataset exhibited 70.83% non-severe, 29.17% severe. All models utilized `class_weight='balanced'` to compensate.

## 2. Best Model Performance (Bootstrap 95% CI)

The best model selected based on the cross-validated F1-score was the **Random Forest**.

| Metric | Mean Score | 95% Confidence Interval |
| :--- | :---: | :--- |
| **Accuracy** | 0.9156 | (0.8950, 0.9384) |
| **Precision** | 0.9256 | (0.8950, 0.9384) |
| **Recall** | 0.7741 | (0.8950, 0.9384) |
| **F1-Score** | 0.8427 | (0.8950, 0.9384) |
| **ROC-AUC** | 0.9733 | (0.8950, 0.9384) |
| **AUPRC** | 0.9402 | (0.8950, 0.9384) |

## 3. Key Feature Importance

The following features were identified as the most impactful for predicting severe outcomes (based on the best model):

1.  age (Score: 0.1863)
2.  vaccination_status_Unvaccinated (Score: 0.1056)
3.  cough (Score: 0.0671)
4.  shortness_of_breath (Score: 0.0661)
5.  loss_of_smell (Score: 0.0619)

## 4. Conclusion

The pipeline successfully trained and evaluated multiple models for predicting severe COVID-19 outcomes. The **Random Forest** provided the most robust performance, demonstrating robust performance metrics.. Future work could involve deep feature analysis and integration of more complex deep learning models.