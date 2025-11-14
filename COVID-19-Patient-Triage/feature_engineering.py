import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from utils import load_config
import numpy as np

class FeatureEngineer:
    def __init__(self, X):
        self.config = load_config()
        self.X = X
        self.preprocessor = None
        self.numerical_features = ['age']
        self.categorical_features = ['gender', 'vaccination_status']
        self.all_symptom_comorbidity_features = [
            'fever', 'cough', 'fatigue', 'shortness_of_breath', 'loss_of_smell', 
            'headache', 'diabetes', 'hypertension', 'heart_disease', 'asthma', 'cancer'
        ]
        # Features to apply PCA to: symptoms and comorbidities (all binary)
        self.pca_features = [col for col in self.all_symptom_comorbidity_features if col in self.X.columns]


    def build_preprocessor(self):
        """
        Builds and fits the ColumnTransformer pipeline, including PCA on binary features.
        """
        
        # 1. Numerical Pipeline: Standard Scaling for 'age'
        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        # 2. Categorical Pipeline: One-Hot Encoding
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # 3. PCA Pipeline for Binary Features (Symptoms/Comorbidities)
        pca_transformer = Pipeline(steps=[
            # Scale binary features before PCA (important for PCA)
            ('scaler', StandardScaler()),
            # PCA aims to explain 90% of the variance
            ('pca', PCA(n_components=0.90, random_state=self.config['random_state'])) 
        ])

        # 4. Combine Preprocessing Steps
        # Note: verbose_feature_names_out=False with set_output(transform="pandas") simplifies names
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_features),
                ('pca', pca_transformer, self.pca_features)
            ],
            remainder='drop',
            verbose_feature_names_out=False
        ).set_output(transform="pandas")

        return self.preprocessor

    def fit_transform(self, X_train):
        """Fits the preprocessor on training data and transforms it."""
        print("\nApplying Preprocessing and Feature Engineering...")
        X_train_processed = self.preprocessor.fit_transform(X_train)
        
        # Report PCA components used
        n_components = self.preprocessor.named_transformers_['pca'].named_steps['pca'].n_components_
        print(f"PCA reduced {len(self.pca_features)} binary features to **{n_components}** components (retaining 90% variance).")

        return X_train_processed

    def transform(self, X_data):
        """Transforms new or test data."""
        X_data_processed = self.preprocessor.transform(X_data)
        return X_data_processed

    def get_feature_names_out(self):
        """
        Retrieves the final feature names.
        Since set_output(transform="pandas") is used, we just return the DataFrame columns.
        """
        # Call transform on a small dummy data set to get the column names correctly.
        # This is the standard robust way when using set_output(transform='pandas').
        dummy_df = self.preprocessor.transform(self.X.head(1))
        
        # The feature names from get_feature_names_out() are already simplified when using pandas output
        return np.array(dummy_df.columns)