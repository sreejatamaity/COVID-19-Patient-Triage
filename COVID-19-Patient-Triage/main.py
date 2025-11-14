import os
import pandas as pd
import numpy as np
import joblib
from utils import load_config, save_artifact, generate_summary
from data_preprocessing import DataProcessor
from feature_engineering import FeatureEngineer
from models import ModelBuilder
from evaluation import Evaluator

def main():
    # --- Configuration and Setup ---
    config = load_config()
    OUTPUT_DIR = config['output_dir']
    PREPROCESSOR_PATH = os.path.join(OUTPUT_DIR, config['preprocessor_path'])
    BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, config['best_model_path'])
    SUMMARY_PATH = 'project-implementation-summary.md'
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # --- Phase 1: Data Processing ---
    print("🚀 Starting COVID-19 Severity Prediction Pipeline...")
    print("\n--- Phase 1: Data Loading & Target Creation ---")
    data_processor = DataProcessor()
    data_processor.load_data()
    imbalance_ratio = data_processor.create_target()
    X, y = data_processor.separate_features_and_target()
    X_train, X_test, y_train, y_test = data_processor.split_data()

    # --- Phase 2: Feature Engineering and Preprocessing ---
    print("\n--- Phase 2: Feature Engineering and Preprocessing ---")
    feature_engineer = FeatureEngineer(X_train)
    preprocessor = feature_engineer.build_preprocessor()
    
    # Fit and transform training data
    X_train_processed = feature_engineer.fit_transform(X_train)
    # Transform test data (keep original DataFrame indexing for bootstrap)
    X_test_processed_df = feature_engineer.transform(X_test) 
    
    save_artifact(preprocessor, PREPROCESSOR_PATH)
    feature_names = feature_engineer.get_feature_names_out()

    # --- Phase 3: Model Training and Tuning ---
    print("\n--- Phase 3: Model Training and Tuning ---")
    model_builder = ModelBuilder()
    best_model, best_model_name = model_builder.tune_and_train_models(X_train_processed, y_train)
    
    save_artifact(best_model, BEST_MODEL_PATH)
    
    # --- Phase 4: Robust Evaluation (Bootstrap) ---
    print("\n--- Phase 4: Robust Evaluation (Bootstrap Validation) ---")
    evaluator = Evaluator(best_model)
    
    # Re-index X_test_processed_df to match y_test index for correct sampling
    X_test_processed_df.index = y_test.index 
    
    # Run bootstrap validation, which now includes AUROC and AUPRC
    bootstrap_metrics = evaluator.bootstrap_validation(X_test_processed_df, y_test, n_bootstrap=200)

    # --- Phase 5: Feature Importance and Summary Generation ---
    print("\n--- Phase 5: Feature Importance and Summary Generation ---")
    
    # Feature Importance (using the best model)
    feature_impact = None
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        feature_impact = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(5)
    elif hasattr(best_model, 'coef_'):
        coefficients = best_model.coef_[0] if best_model.coef_.ndim > 1 else best_model.coef_
        feature_impact = pd.Series(np.abs(coefficients), index=feature_names).sort_values(ascending=False).head(5)
    
    if feature_impact is not None:
        print("\nTop 5 Features impacting Severity Prediction:")
        print(feature_impact)

    # Generate the final summary markdown file, which will now include AUPRC data
    generate_summary(SUMMARY_PATH, bootstrap_metrics, feature_impact, best_model_name, X, y)
    
    print("\n✅ Pipeline execution complete!")
    print(f"Best model: **{best_model_name}** saved to `{BEST_MODEL_PATH}`")


if __name__ == '__main__':
    main()