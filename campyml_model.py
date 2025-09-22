#!/usr/bin/env python3
"""
CampyML Model - XGBoost for Campylobacter classification
Training and Prediction
"""

import pandas as pd
import numpy as np
import pickle
import argparse
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')


class CampyMLModel:
    def __init__(self, model_path=None):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.model_path = model_path
        self.feature_columns = None

    def load_data(self, file_path):
        """Load data from CSV file"""
        print(f"Loading data from {file_path}...")
        df = pd.read_csv(file_path, low_memory=False)
        print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df

    def preprocess_data(self, df, target_column='source'):
        """Data preprocessing"""
        print("Preprocessing data...")

        # Remove rows with null target
        df = df.dropna(subset=[target_column])

        # Separate features and target
        X = df.drop([target_column], axis=1, errors='ignore')
        y = df[target_column]

        # Convert categorical columns to numeric
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = pd.factorize(X[col])[0]

        # Remove columns with all NaN
        X = X.dropna(axis=1, how='all')

        # Fill missing values with -1
        X = X.fillna(-1)

        # Save column names
        self.feature_columns = X.columns.tolist()

        # Encode il target
        y_encoded = self.label_encoder.fit_transform(y)

        print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
        print(f"Classes found: {self.label_encoder.classes_}")

        return X, y_encoded

    def train(self, X, y, test_size=0.2, random_state=42, use_smote=True):
        """Training del modello XGBoost"""

        # If dataset is too small, use all data for training
        n_classes = len(np.unique(y))
        min_samples_needed = n_classes * 2  # At least 2 samples per class

        if len(X) < min_samples_needed or len(X) < 20:
            print(f"Small dataset ({len(X)} samples), using all data for training...")
            X_train, y_train = X, y
            X_test, y_test = X, y  # Use same data for validation
        else:
            print("Splitting train/test...")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )

        if use_smote and len(X_train) >= 20:  # Only use SMOTE with sufficient data
            print("Applying SMOTE for class balancing...")
            smote = SMOTE(random_state=random_state)
            X_train, y_train = smote.fit_resample(X_train, y_train)
        elif use_smote:
            print("Skipping SMOTE (insufficient data for balancing)")

        print("Training XGBoost model...")
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            objective='multi:softprob',
            random_state=random_state,
            n_jobs=-1
        )

        self.model.fit(X_train, y_train)

        # Evaluation
        print("\nEvaluation on test set:")
        y_pred = self.model.predict(X_test)

        print("\nClassification Report:")
        print(classification_report(
            y_test, y_pred,
            target_names=self.label_encoder.classes_
        ))

        return self.model

    def save_model(self, path=None):
        """Save model and label encoder"""
        if path is None:
            path = self.model_path or 'campyml_model.pkl'

        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'feature_columns': self.feature_columns
        }

        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {path}")

    def load_model(self, path=None):
        """Load saved model"""
        if path is None:
            path = self.model_path or 'campyml_model.pkl'

        with open(path, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.label_encoder = model_data['label_encoder']
        self.feature_columns = model_data.get('feature_columns', None)
        print(f"Model loaded from {path}")

    def predict(self, X):
        """Prediction on new data"""
        if self.model is None:
            raise ValueError("Model not loaded. Use load_model() first.")

        # Ensure columns are in the same order
        if self.feature_columns:
            X = X[self.feature_columns]

        # Preprocessing
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = pd.factorize(X[col])[0]
        X = X.fillna(-1)

        # Prediction
        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)

        # Decode predictions
        predictions = self.label_encoder.inverse_transform(y_pred)

        return predictions, y_pred_proba


def main():
    parser = argparse.ArgumentParser(description='CampyML Model - Training e Prediction')
    parser.add_argument('--mode', choices=['train', 'predict'], required=True,
                        help='Mode: train or predict')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to CSV data file')
    parser.add_argument('--model', type=str, default='campyml_model.pkl',
                        help='Model path (for saving or loading)')
    parser.add_argument('--target', type=str, default='source',
                        help='Target column name (default: source)')
    parser.add_argument('--output', type=str, default='predictions.csv',
                        help='Output file for predictions')

    args = parser.parse_args()

    # Initialize model
    model = CampyMLModel(model_path=args.model)

    if args.mode == 'train':
        # Training
        print("=== TRAINING MODE ===")
        df = model.load_data(args.data)
        X, y = model.preprocess_data(df, target_column=args.target)
        model.train(X, y)
        model.save_model()
        print("\nTraining completed!")

    else:  # predict
        # Prediction
        print("=== PREDICTION MODE ===")
        model.load_model()
        df = model.load_data(args.data)

        # Remove target column if present
        if args.target in df.columns:
            df = df.drop([args.target], axis=1)

        predictions, probabilities = model.predict(df)

        # Save predictions
        results = pd.DataFrame({
            'prediction': predictions,
            'confidence': probabilities.max(axis=1)
        })

        # Add probabilities for each class
        for i, class_name in enumerate(model.label_encoder.classes_):
            results[f'prob_{class_name}'] = probabilities[:, i]

        results.to_csv(args.output, index=False)
        print(f"\nPredictions saved to {args.output}")
        print(f"Total predictions: {len(predictions)}")
        print(f"\nPrediction distribution:")
        print(results['prediction'].value_counts())


if __name__ == '__main__':
    main()