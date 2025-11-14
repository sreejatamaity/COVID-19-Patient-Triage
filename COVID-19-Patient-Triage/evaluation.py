import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from tqdm import tqdm

class Evaluator:
    def __init__(self, model):
        self.model = model

    def bootstrap_validation(self, X_test, y_test, n_bootstrap=100):
        """
        Performs bootstrap resampling to estimate model performance and confidence intervals.
        Includes AUROC (ROC-AUC) and AUPRC (Average Precision).
        """
        print(f"\n--- Running Bootstrap Validation (n={n_bootstrap}) ---")
        
        metrics = {
            'Accuracy': [],
            'Precision': [],
            'Recall': [],
            'F1-Score': [],
            'ROC-AUC': [],
            'AUPRC': []
        }
        
        n_samples = len(X_test)
        
        for _ in tqdm(range(n_bootstrap), desc="Bootstrapping"):
            # Sample with replacement
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_sample = X_test.iloc[indices]
            y_sample = y_test.iloc[indices]
            
            # Make predictions
            y_pred = self.model.predict(X_sample)
            
            # Calculate probabilities if supported
            y_proba = None
            if hasattr(self.model, 'predict_proba'):
                y_proba = self.model.predict_proba(X_sample)[:, 1]

            # Calculate metrics
            metrics['Accuracy'].append(accuracy_score(y_sample, y_pred))
            metrics['Precision'].append(precision_score(y_sample, y_pred, zero_division=0))
            metrics['Recall'].append(recall_score(y_sample, y_pred, zero_division=0))
            metrics['F1-Score'].append(f1_score(y_sample, y_pred, zero_division=0))
            
            if y_proba is not None:
                metrics['ROC-AUC'].append(roc_auc_score(y_sample, y_proba))
                metrics['AUPRC'].append(average_precision_score(y_sample, y_proba))
            else:
                metrics['ROC-AUC'].append(np.nan)
                metrics['AUPRC'].append(np.nan)


        # Convert to numpy arrays for easier calculation
        for key in metrics:
            metrics[key] = np.array(metrics[key])

        print("\nBootstrap Results:")
        for metric, values in metrics.items():
            mean = np.nanmean(values)
            ci_lower = np.nanpercentile(values, 2.5)
            ci_upper = np.nanpercentile(values, 97.5)
            print(f"  {metric}: Mean={mean:.4f}, 95% CI=({ci_lower:.4f}, {ci_upper:.4f})")
            
        return metrics