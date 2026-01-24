"""
Model Trainer for Tennis Match Prediction.

Implements:
1. Temporal cross-validation (NO LEAKAGE!)
2. XGBoost with proper hyperparameter tuning
3. Probability calibration for accurate predictions
4. Comprehensive evaluation metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import pickle
import logging
from datetime import datetime

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, classification_report, confusion_matrix,
    brier_score_loss
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Train and evaluate XGBoost model for tennis match prediction.
    
    Key principles:
    1. Temporal validation - train on past, validate on future
    2. Proper handling of categorical features
    3. Probability calibration for betting applications
    4. No data leakage throughout the pipeline
    """
    
    # Default XGBoost parameters (optimized for calibration and generalization)
    DEFAULT_XGB_PARAMS = {
        'n_estimators': 500,
        'max_depth': 3,           # Very shallow for better generalization
        'learning_rate': 0.02,    # Lower LR
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'min_child_weight': 30,   # Strong regularization
        'gamma': 0.5,             # Strong pruning
        'reg_alpha': 2.0,         # L1 regularization
        'reg_lambda': 5.0,        # L2 regularization
        'scale_pos_weight': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'eval_metric': 'logloss',
        'early_stopping_rounds': 50
    }
    
    def __init__(self):
        self.model = None
        self.calibrated_model = None
        self.label_encoders = {}
        self.feature_cols = []
        self.categorical_cols = []
        self.metrics = {}
        
    def prepare_data(
        self, 
        df_train: pd.DataFrame, 
        df_test: pd.DataFrame,
        target_col: str = 'p1_won'
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Prepare train and test data for modeling.
        
        Handles:
        - Categorical encoding
        - Missing value handling
        - Feature/target separation
        """
        logger.info("")
        logger.info("PREPARING DATA FOR MODELING")
        logger.info("")
        
        # Identify feature columns (exclude target)
        self.feature_cols = [col for col in df_train.columns if col != target_col]
        
        # Identify categorical columns
        self.categorical_cols = df_train[self.feature_cols].select_dtypes(
            include=['object']
        ).columns.tolist()
        
        logger.info(f"Features: {len(self.feature_cols)}")
        logger.info(f"Categorical features: {len(self.categorical_cols)}")
        
        # Prepare copies
        X_train = df_train[self.feature_cols].copy()
        y_train = df_train[target_col].copy()
        X_test = df_test[self.feature_cols].copy()
        y_test = df_test[target_col].copy()
        
        # Encode categorical features
        logger.info("\nEncoding categorical features...")
        
        for col in self.categorical_cols:
            le = LabelEncoder()
            
            # Fit on train + test combined to handle all possible values
            combined_vals = pd.concat([X_train[col], X_test[col]]).fillna('MISSING').astype(str)
            le.fit(combined_vals)
            
            # Transform
            X_train[col] = X_train[col].fillna('MISSING').astype(str)
            X_test[col] = X_test[col].fillna('MISSING').astype(str)
            
            X_train[col] = le.transform(X_train[col])
            X_test[col] = le.transform(X_test[col])
            
            self.label_encoders[col] = le
            logger.info(f"  {col}: {len(le.classes_)} categories")
        
        # Handle remaining NaN values
        X_train = X_train.fillna(0)
        X_test = X_test.fillna(0)
        
        logger.info(f"\nFinal shapes:")
        logger.info(f"  X_train: {X_train.shape}")
        logger.info(f"  X_test:  {X_test.shape}")
        logger.info(f"  Target balance (train): {y_train.mean():.3f}")
        logger.info(f"  Target balance (test):  {y_test.mean():.3f}")
        
        return X_train, y_train, X_test, y_test
    
    def temporal_cross_validation(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5
    ) -> Dict[str, List[float]]:
        """
        Perform temporal cross-validation.
        
        Uses TimeSeriesSplit to ensure no future data leaks into past.
        """
        logger.info("\n" + "")
        logger.info("TEMPORAL CROSS-VALIDATION")
        logger.info("")
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        cv_metrics = {
            'accuracy': [],
            'roc_auc': [],
            'log_loss': [],
            'f1': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
            y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train model for this fold
            model = xgb.XGBClassifier(**self.DEFAULT_XGB_PARAMS)
            model.fit(
                X_fold_train, y_fold_train,
                eval_set=[(X_fold_val, y_fold_val)],
                verbose=False
            )
            
            # Predict
            y_pred = model.predict(X_fold_val)
            y_proba = model.predict_proba(X_fold_val)[:, 1]
            
            # Calculate metrics
            cv_metrics['accuracy'].append(accuracy_score(y_fold_val, y_pred))
            cv_metrics['roc_auc'].append(roc_auc_score(y_fold_val, y_proba))
            cv_metrics['log_loss'].append(log_loss(y_fold_val, y_proba))
            cv_metrics['f1'].append(f1_score(y_fold_val, y_pred))
            
            logger.info(f"  Fold {fold}: AUC={cv_metrics['roc_auc'][-1]:.4f}, "
                       f"Accuracy={cv_metrics['accuracy'][-1]:.4f}, "
                       f"LogLoss={cv_metrics['log_loss'][-1]:.4f}")
        
        # Summary
        logger.info("\nCV Summary:")
        for metric, values in cv_metrics.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            logger.info(f"  {metric}: {mean_val:.4f} ± {std_val:.4f}")
        
        return cv_metrics
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        params: Optional[Dict] = None
    ) -> xgb.XGBClassifier:
        """
        Train XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (for early stopping)
            y_val: Validation target
            params: XGBoost parameters (uses defaults if None)
        """
        logger.info("\n" + "")
        logger.info("TRAINING XGBOOST MODEL")
        logger.info("")
        
        if params is None:
            params = self.DEFAULT_XGB_PARAMS.copy()
        
        # Add early stopping if validation data provided
        if X_val is not None:
            params['early_stopping_rounds'] = 30
        
        self.model = xgb.XGBClassifier(**params)
        
        if X_val is not None:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=10
            )
            logger.info(f"Best iteration: {self.model.best_iteration}")
        else:
            self.model.fit(X_train, y_train)
        
        logger.info("Model trained!")
        
        return self.model
    
    def calibrate(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        method: str = 'isotonic'
    ) -> CalibratedClassifierCV:
        """
        Calibrate model probabilities.
        
        Important for betting applications where accurate
        probability estimates are crucial.
        """
        logger.info("\n" + "")
        logger.info("PROBABILITY CALIBRATION")
        logger.info("")
        
        if self.model is None:
            raise ValueError("Model must be trained before calibration!")
        
        self.calibrated_model = CalibratedClassifierCV(
            self.model,
            method=method,
            cv='prefit'
        )
        
        self.calibrated_model.fit(X_train, y_train)
        
        logger.info(f"Calibration complete (method: {method})")
        
        return self.calibrated_model
    
    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        use_calibrated: bool = True
    ) -> Dict[str, float]:
        """
        Comprehensive model evaluation.
        
        Returns metrics including:
        - Accuracy, Precision, Recall, F1
        - ROC-AUC, Log Loss, Brier Score
        - Confusion matrix
        """
        logger.info("\n" + "")
        logger.info("MODEL EVALUATION")
        logger.info("")
        
        model = self.calibrated_model if (use_calibrated and self.calibrated_model) else self.model
        
        if model is None:
            raise ValueError("No model to evaluate!")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'log_loss': log_loss(y_test, y_proba),
            'brier_score': brier_score_loss(y_test, y_proba)
        }
        
        self.metrics = metrics
        
        # Print results
        logger.info("\nTest Metrics:")
        logger.info("-" * 40)
        logger.info(f"  Accuracy:     {metrics['accuracy']:.4f}")
        logger.info(f"  Precision:    {metrics['precision']:.4f}")
        logger.info(f"  Recall:       {metrics['recall']:.4f}")
        logger.info(f"  F1-Score:     {metrics['f1']:.4f}")
        logger.info(f"  ROC-AUC:      {metrics['roc_auc']:.4f}")
        logger.info(f"  Log Loss:     {metrics['log_loss']:.4f}")
        logger.info(f"  Brier Score:  {metrics['brier_score']:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        logger.info("\nConfusion Matrix:")
        logger.info(f"  [[{cm[0,0]:5d} {cm[0,1]:5d}]")
        logger.info(f"   [{cm[1,0]:5d} {cm[1,1]:5d}]]")
        
        # Classification report
        logger.info("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['P1 Lost', 'P1 Won']))
        
        # Analysis by confidence level
        self._analyze_by_confidence(y_test, y_proba, y_pred)
        
        return metrics
    
    def _analyze_by_confidence(
        self,
        y_true: pd.Series,
        y_proba: np.ndarray,
        y_pred: np.ndarray
    ):
        """Analyze model performance by confidence level."""
        logger.info("\nPerformance by Confidence Level:")
        logger.info("-" * 60)
        
        confidence = np.maximum(y_proba, 1 - y_proba)
        correct = (y_pred == y_true)
        
        bins = [
            ('Very Confident (>70%)', 0.70, 1.00),
            ('Confident (60-70%)', 0.60, 0.70),
            ('Moderate (55-60%)', 0.55, 0.60),
            ('Uncertain (50-55%)', 0.50, 0.55),
        ]
        
        for label, min_conf, max_conf in bins:
            mask = (confidence >= min_conf) & (confidence < max_conf)
            if mask.sum() > 0:
                acc = correct[mask].mean()
                count = mask.sum()
                logger.info(f"  {label:25s} | Matches: {count:5d} | Accuracy: {acc:.2%}")
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get top N most important features."""
        if self.model is None:
            raise ValueError("Model must be trained first!")
        
        importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info(f"\nTop {top_n} Feature Importances:")
        logger.info("-" * 40)
        for _, row in importance.head(top_n).iterrows():
            logger.info(f"  {row['feature']:35s} {row['importance']:.4f}")
        
        return importance.head(top_n)
    
    def save_model(
        self,
        model_dir: str = 'models',
        save_uncalibrated: bool = False
    ):
        """
        Save model and related artifacts.
        
        Saves:
        - Calibrated model (primary)
        - Feature columns list
        - Label encoders
        - Model metrics
        """
        logger.info("\n" + "")
        logger.info("SAVING MODEL")
        logger.info("")
        
        model_path = Path(model_dir)
        model_path.mkdir(exist_ok=True, parents=True)
        
        # Save calibrated model (primary)
        model_to_save = self.calibrated_model if self.calibrated_model else self.model
        
        model_file = model_path / 'xgboost_calibrated_model.pkl'
        with open(model_file, 'wb') as f:
            pickle.dump(model_to_save, f)
        logger.info(f"  Model saved: {model_file}")
        
        # Save feature columns
        features_file = model_path / 'feature_columns.txt'
        with open(features_file, 'w') as f:
            f.write('\n'.join(self.feature_cols))
        logger.info(f"  Features saved: {features_file}")
        
        # Save label encoders
        encoders_file = model_path / 'label_encoders.pkl'
        with open(encoders_file, 'wb') as f:
            pickle.dump(self.label_encoders, f)
        logger.info(f"  Encoders saved: {encoders_file}")
        
        # Save metrics
        metrics_file = model_path / 'model_metrics.txt'
        with open(metrics_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write(f"Tennis Match Prediction Model Metrics\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")
            
            for metric, value in self.metrics.items():
                f.write(f"{metric:15s}: {value:.4f}\n")
        
        logger.info(f"  Metrics saved: {metrics_file}")
        
        # Save uncalibrated model if requested
        if save_uncalibrated and self.model:
            uncal_file = model_path / 'xgboost_model_uncalibrated.pkl'
            with open(uncal_file, 'wb') as f:
                pickle.dump(self.model, f)
            logger.info(f"  Uncalibrated model saved: {uncal_file}")
    
    def load_model(self, model_dir: str = 'models') -> Tuple[Any, List[str], Dict]:
        """
        Load saved model and artifacts.
        
        Returns:
            Tuple of (model, feature_cols, label_encoders)
        """
        model_path = Path(model_dir)
        
        # Load model
        with open(model_path / 'xgboost_calibrated_model.pkl', 'rb') as f:
            self.calibrated_model = pickle.load(f)
        
        # Load features
        with open(model_path / 'feature_columns.txt', 'r') as f:
            self.feature_cols = [line.strip() for line in f.readlines()]
        
        # Load encoders
        with open(model_path / 'label_encoders.pkl', 'rb') as f:
            self.label_encoders = pickle.load(f)
        
        logger.info(f"Model loaded from {model_dir}")
        
        return self.calibrated_model, self.feature_cols, self.label_encoders


if __name__ == '__main__':
    # Example usage
    print("ModelTrainer module - use via TennisPredictionPipeline")
