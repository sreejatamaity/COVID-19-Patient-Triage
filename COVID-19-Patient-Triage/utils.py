import json
import joblib
import os
import numpy as np
import pandas as pd

def load_config(config_path='config.json'):
    """Load configuration settings from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def save_artifact(artifact, path):
    """Save a Python object (model or preprocessor) to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(artifact, path)
    print(f"Artifact saved to: {path}")

def load_artifact(path):
    """Load a Python object (model or preprocessor) from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Artifact not found at {path}")
    return joblib.load(path)

def generate_summary(summary_path, metrics, feature_impact, best_model_name, X, y):
    """Fills the project implementation summary template."""
    
    try:
        with open(summary_path, 'r') as f:
            summary_template = f.read()
    except FileNotFoundError:
        print(f"Error: Summary template not found at {summary_path}")
        return

    # Calculate imbalance ratio
    imbalance_ratio_data = y.value_counts(normalize=True).mul(100).to_dict()
    imbalance_str = f"{imbalance_ratio_data.get(0, 0):.2f}% non-severe, {imbalance_ratio_data.get(1, 0):.2f}% severe"

    # Fill basic placeholders
    summary_content = summary_template.replace("[Best Model Name]", best_model_name)
    summary_content = summary_content.replace("[Number of samples]", str(len(X)))
    summary_content = summary_content.replace("[Imbalance ratio, e.g., 90% non-severe, 10% severe]", imbalance_str)

    # Fill performance table
    for metric_name in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'AUPRC']:
        values = metrics.get(metric_name)
        if values is not None:
            mean = np.nanmean(values)
            lower_ci = np.nanpercentile(values, 2.5)
            upper_ci = np.nanpercentile(values, 97.5)
            
            # Use specific placeholders in the template for replacement
            summary_content = summary_content.replace(f"[Mean {metric_name}]", f"{mean:.4f}")
            summary_content = summary_content.replace(f"([Lower CI], [Upper CI])", f"({lower_ci:.4f}, {upper_ci:.4f})")
    
    # Fill feature importance section
    if feature_impact is not None and not feature_impact.empty:
        for i, (name, score) in enumerate(feature_impact.items()):
            # Using the fact that the original template has numbered placeholders
            summary_content = summary_content.replace(f"[Feature {i+1} Name] (Score: [Score])", f"{name} (Score: {score:.4f})")
        
        # Replace remaining placeholders with empty strings if fewer than 5 features
        for i in range(len(feature_impact), 5):
             summary_content = summary_content.replace(f"\n{i+1}.  [Feature {i+1} Name] (Score: [Score])", "")
        summary_content = summary_content.replace("demonstrating [Brief conclusion about model quality]", "demonstrating robust performance metrics.")


    with open(summary_path, 'w') as f:
        f.write(summary_content)
    
    print(f"Summary content written/updated at {summary_path}")