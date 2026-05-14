"""
Tennis Match Prediction Pipeline.

Complete end-to-end pipeline for:
1. Loading and processing data
2. Feature engineering (NO DATA LEAKAGE)
3. Model training and calibration
4. Evaluation and saving

Usage:
    from src.pipeline import TennisPredictionPipeline
    
    pipeline = TennisPredictionPipeline()
    pipeline.run(start_year=2012, end_year=2025, test_year=2025)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple
import logging
from datetime import datetime

from .data_processor import DataProcessor
from .feature_engineer import FeatureEngineer
from .model_trainer import ModelTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TennisPredictionPipeline:
    """
    End-to-end pipeline for tennis match prediction.
    
    Design Principles:
    1. NO DATA LEAKAGE - strict temporal causality
    2. Reproducibility - fixed random seeds
    3. Modularity - each component can be used independently
    4. Production-ready - saved models ready for inference
    """
    
    def __init__(
        self,
        data_path: str = 'tml-data',
        output_path: str = 'data/processed',
        model_path: str = 'models'
    ):
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.model_path = Path(model_path)
        
        # Initialize components
        self.data_processor = DataProcessor(data_path)
        self.feature_engineer = FeatureEngineer()
        self.model_trainer = ModelTrainer()
        
        # Data storage
        self.df_raw = None
        self.df_processed = None
        self.df_features = None
        self.df_train = None
        self.df_test = None
        
    def run(
        self,
        start_year: int = 2012,
        end_year: int = 2026,
        test_year: int = 2026,
        run_cv: bool = True,
        save_data: bool = True,
        save_model: bool = True
    ) -> Dict:
        """
        Run complete pipeline.
        
        Args:
            start_year: First year of data
            end_year: Last year of data
            test_year: Year to use as test set
            run_cv: Whether to run cross-validation
            save_data: Whether to save processed data
            save_model: Whether to save trained model
            
        Returns:
            Dictionary with metrics and model info
        """
        logger.info("="*80)
        logger.info("TENNIS MATCH PREDICTION PIPELINE")
        logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*80)
        
        results = {}
        
        # Step 1: Load and clean data
        logger.info("\nSTEP 1: Data Loading & Cleaning")
        
        self.df_raw = self.data_processor.load_data(start_year, end_year)
        self.df_processed = self.data_processor.clean_data()
        
        # Step 2: Create features
        logger.info("\nSTEP 2: Feature Engineering")
        
        self.df_features = self.feature_engineer.create_all_features(self.df_processed)
        
        # Validate no data leakage
        leakage_free = self.feature_engineer.validate_no_leakage(self.df_features)
        if not leakage_free:
            logger.error("DATA LEAKAGE DETECTED! Aborting pipeline.")
            return {'error': 'data_leakage'}
        
        # Step 3: Temporal split
        logger.info("\nSTEP 3: Temporal Split")
        
        # Split based on original year (before duplication)
        self.df_train = self.df_features[
            self.df_features['data_year'] < test_year
        ].copy() if 'data_year' in self.df_features.columns else None
        
        self.df_test = self.df_features[
            self.df_features['data_year'] >= test_year
        ].copy() if 'data_year' in self.df_features.columns else None
        
        # If data_year was dropped, use temporal split from processed data
        if self.df_train is None or len(self.df_train) == 0:
            train_processed, test_processed = self.data_processor.temporal_split(
                self.df_processed, test_year
            )
            
            # Re-run feature engineering on split data
            self.df_train = self.feature_engineer.create_all_features(train_processed)
            self.df_test = self.feature_engineer.create_all_features(test_processed)
        
        # Drop data_year if present (not a feature)
        if 'data_year' in self.df_train.columns:
            self.df_train = self.df_train.drop(columns=['data_year'])
        if 'data_year' in self.df_test.columns:
            self.df_test = self.df_test.drop(columns=['data_year'])
        
        logger.info(f"Train size: {len(self.df_train):,}")
        logger.info(f"Test size:  {len(self.df_test):,}")
        
        # Save processed data if requested
        if save_data:
            self._save_processed_data()
        
        # Step 4: Prepare data for modeling
        logger.info("\nSTEP 4: Data Preparation")
        
        X_train, y_train, X_test, y_test = self.model_trainer.prepare_data(
            self.df_train, self.df_test, target_col='p1_won'
        )
        
        # Step 5: Cross-validation (optional)
        if run_cv:
            logger.info("\nSTEP 5: Cross-Validation")
            
            cv_metrics = self.model_trainer.temporal_cross_validation(X_train, y_train)
            results['cv_metrics'] = cv_metrics
        
        # Step 6: Train final model
        logger.info("\nSTEP 6: Model Training")
        
        self.model_trainer.train(X_train, y_train, X_test, y_test)
        
        # Step 7: Calibrate probabilities
        logger.info("\nSTEP 7: Probability Calibration")
        
        self.model_trainer.calibrate(X_train, y_train)
        
        # Step 8: Evaluate
        logger.info("\nSTEP 8: Evaluation")
        
        test_metrics = self.model_trainer.evaluate(X_test, y_test)
        results['test_metrics'] = test_metrics
        
        # Feature importance
        feature_importance = self.model_trainer.get_feature_importance(top_n=20)
        results['feature_importance'] = feature_importance
        
        # Step 9: Save model
        if save_model:
            logger.info("\nSTEP 9: Save Model")
            
            self.model_trainer.save_model(str(self.model_path))
            self._save_metrics(test_metrics, start_year, end_year, test_year)
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info("PIPELINE COMPLETE")
        logger.info("="*80)
        logger.info(f"\nResults:")
        logger.info(f"  ROC-AUC:   {test_metrics['roc_auc']:.4f}")
        logger.info(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
        logger.info(f"  Log Loss:  {test_metrics['log_loss']:.4f}")
        logger.info(f"  F1-Score:  {test_metrics['f1']:.4f}")
        
        return results
    
    def _save_processed_data(self):
        """Save processed train and test data."""
        self.output_path.mkdir(exist_ok=True, parents=True)
        
        train_path = self.output_path / 'train_features.csv'
        test_path = self.output_path / 'test_features.csv'
        
        self.df_train.to_csv(train_path, index=False)
        self.df_test.to_csv(test_path, index=False)
        
        logger.info(f"Saved: {train_path}")
        logger.info(f"Saved: {test_path}")
    
    def _save_metrics(self, metrics: Dict, start_year: int, end_year: int, test_year: int):
        """Save training metrics to file."""
        self.model_path.mkdir(exist_ok=True, parents=True)
        
        # Save to text file (for backward compatibility)
        metrics_txt_path = self.model_path / 'model_metrics.txt'
        with open(metrics_txt_path, 'w') as f:
            f.write(f"Model Metrics (Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\\n")
            f.write(f"Training Period: {start_year}-{end_year}\\n")
            f.write(f"Test Year: {test_year}\\n")
            f.write(f"\\nPerformance:\\n")
            f.write(f"  ROC-AUC:   {metrics['roc_auc']:.4f}\\n")
            f.write(f"  Accuracy:  {metrics['accuracy']:.4f}\\n")
            f.write(f"  Log Loss:  {metrics['log_loss']:.4f}\\n")
            f.write(f"  F1-Score:  {metrics['f1']:.4f}\\n")
            f.write(f"  Precision: {metrics['precision']:.4f}\\n")
            f.write(f"  Recall:    {metrics['recall']:.4f}\\n")
        
        # Save to CSV for history tracking
        metrics_csv_path = self.model_path / 'metrics_history.csv'
        
        # Create new row
        new_row = pd.DataFrame([{
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'train_period': f"{start_year}-{end_year}",
            'test_year': test_year,
            'roc_auc': metrics['roc_auc'],
            'accuracy': metrics['accuracy'],
            'log_loss': metrics['log_loss'],
            'f1_score': metrics['f1'],
            'precision': metrics['precision'],
            'recall': metrics['recall']
        }])
        
        # Append to existing or create new
        if metrics_csv_path.exists():
            existing = pd.read_csv(metrics_csv_path)
            combined = pd.concat([existing, new_row], ignore_index=True)
            combined.to_csv(metrics_csv_path, index=False)
        else:
            new_row.to_csv(metrics_csv_path, index=False)
        
        logger.info(f"Saved metrics: {metrics_txt_path}")
        logger.info(f"Updated history: {metrics_csv_path}")

    def retrain_on_full_data(self) -> Dict:
        """
        Retrain model on complete dataset (train + test).
        
        Use this after validating model performance to get
        the most up-to-date model for production.
        """
        logger.info("\n" + "=" * 80)
        logger.info("RETRAINING ON FULL DATA")
        
        # Combine train and test
        df_full = pd.concat([self.df_train, self.df_test], ignore_index=True)
        
        logger.info(f"Full dataset: {len(df_full):,} rows")
        
        # Prepare data
        feature_cols = [col for col in df_full.columns if col != 'p1_won']
        X_full = df_full[feature_cols].copy()
        y_full = df_full['p1_won'].copy()
        
        # Encode categoricals
        for col in self.model_trainer.categorical_cols:
            if col in X_full.columns:
                X_full[col] = X_full[col].fillna('MISSING').astype(str)
                X_full[col] = self.model_trainer.label_encoders[col].transform(X_full[col])
        
        X_full = X_full.fillna(0)
        
        # Split for early stopping (90/10)
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X_full, y_full, test_size=0.1, random_state=42, stratify=y_full
        )
        
        # Retrain with validation
        self.model_trainer.train(X_train, y_train, X_val, y_val)
        
        # Calibrate on full data
        self.model_trainer.calibrate(X_full, y_full)
        
        # Save updated model
        self.model_trainer.save_model(str(self.model_path))
        
        logger.info("Model retrained on full data and saved!")
        
        return {'status': 'success', 'rows_trained': len(df_full)}
    
    def run_production(
        self,
        start_year: int = 2012,
        end_year: int = 2025
    ) -> Dict:
        """
        Run production pipeline: train on ALL data without test split.
        
        Use this for final deployment when model is validated.
        
        Args:
            start_year: First year of training data
            end_year: Last year of training data
            
        Returns:
            Dictionary with training results
        """
        logger.info("\n" + "=" * 80)
        logger.info("PRODUCTION PIPELINE (TRAIN ON ALL DATA)")
        logger.info(f"Period: {start_year}-{end_year}")
        
        results = {}
        
        # Step 1: Load ALL data
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: DATA LOADING (ALL DATA)")
        
        self.df_raw = self.data_processor.load_data(start_year, end_year)
        self.df_processed = self.data_processor.clean_data(self.df_raw)
        
        # Step 2: Feature engineering on ALL data
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: FEATURE ENGINEERING")
        
        df_features = self.feature_engineer.create_all_features(self.df_processed)
        
        # Validate no leakage
        self.feature_engineer.validate_no_leakage(df_features)
        
        # Drop data_year if present
        if 'data_year' in df_features.columns:
            df_features = df_features.drop(columns=['data_year'])
        
        logger.info(f"Total training data: {len(df_features):,} rows")
        
        # Step 3: Prepare data
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: DATA PREPARATION")
        
        target_col = 'p1_won'
        feature_cols = [col for col in df_features.columns if col != target_col]
        
        X = df_features[feature_cols].copy()
        y = df_features[target_col].copy()
        
        # Encode categoricals
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        
        from sklearn.preprocessing import LabelEncoder
        
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = X[col].fillna('MISSING').astype(str)
            X[col] = le.fit_transform(X[col])
            self.model_trainer.label_encoders[col] = le
        
        self.model_trainer.feature_cols = feature_cols
        self.model_trainer.categorical_cols = categorical_cols
        
        X = X.fillna(0)
        
        logger.info(f"Features: {len(feature_cols)}")
        logger.info(f"Samples: {len(X):,}")
        
        # Step 4: Train model
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: MODEL TRAINING")
        
        # Use 10% of data as validation for early stopping
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.1, random_state=42, stratify=y
        )
        
        self.model_trainer.train(X_train, y_train, X_val, y_val)
        
        # Step 5: Calibrate on full data
        logger.info("\n" + "=" * 80)
        logger.info("STEP 5: PROBABILITY CALIBRATION")
        
        self.model_trainer.calibrate(X, y)
        
        # Step 6: Save model
        logger.info("\n" + "=" * 80)
        logger.info("STEP 6: SAVE MODEL")
        
        self.model_trainer.save_model(str(self.model_path))
        
        results['status'] = 'success'
        results['total_samples'] = len(X)
        results['features'] = len(feature_cols)
        
        logger.info("\n" + "=" * 80)
        logger.info("PRODUCTION PIPELINE COMPLETE")
        logger.info(f"Model trained on {len(X):,} samples")
        logger.info(f"Model saved to: {self.model_path}")
        
        return results


if __name__ == '__main__':
    # Run pipeline
    pipeline = TennisPredictionPipeline()
    results = pipeline.run(
        start_year=2012,
        end_year=2025,
        test_year=2025,
        run_cv=True,
        save_data=True,
        save_model=True
    )
