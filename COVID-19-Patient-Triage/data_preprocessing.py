import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from utils import load_config

class DataProcessor:
    def __init__(self):
        self.config = load_config()
        self.data_path = self.config['data_path']
        self.random_state = self.config['random_state']
        self.test_size = self.config['test_size']
        self.df = None
        self.X = None
        self.y = None
        self.feature_columns = None

    def load_data(self):
        """Loads the CSV file."""
        self.df = pd.read_csv(self.data_path)
        print(f"Data loaded successfully. Shape: {self.df.shape}")

    def create_target(self):
        """Creates the binary target variable 'severity_flag'."""
        # Severity: 1 if hospitalized OR icu_admission OR mortality is 1, else 0.
        self.df[self.config['target_column']] = np.where(
            (self.df['hospitalized'] == 1) | 
            (self.df['icu_admission'] == 1) | 
            (self.df['mortality'] == 1), 
            1, 
            0
        )
        
        # Calculate imbalance ratio for console output
        imbalance_ratio = self.df[self.config['target_column']].value_counts(normalize=True).mul(100).to_dict()
        imbalance_str = f"{imbalance_ratio.get(0, 0):.2f}% non-severe, {imbalance_ratio.get(1, 0):.2f}% severe"
        
        print(f"Target variable '{self.config['target_column']}' created. Imbalance: {imbalance_str}")
        return imbalance_str

    def separate_features_and_target(self):
        """Separates features and target, discarding intermediate columns."""
        features_to_exclude = ['hospitalized', 'icu_admission', 'mortality', self.config['target_column']]
        self.X = self.df.drop(columns=features_to_exclude)
        self.y = self.df[self.config['target_column']]
        self.feature_columns = self.X.columns.tolist()
        return self.X, self.y

    def split_data(self):
        """Splits data into training and testing sets using stratification."""
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state, stratify=self.y
        )
        print(f"Data split: Train ({X_train.shape[0]} samples), Test ({X_test.shape[0]} samples).")
        return X_train, X_test, y_train, y_test