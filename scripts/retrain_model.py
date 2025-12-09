"""Model retraining with proper categorical feature encoding"""
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
from pathlib import Path

print("="*80)
print("MODEL RETRAINING WITH PROPER ENCODING")
print("="*80)

# 1. Load data
print("\nLoading data...")
train_df = pd.read_csv('../data/processed/train_features.csv')
test_df = pd.read_csv('../data/processed/test_features.csv')

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# 2. Load feature columns
with open('../models/feature_columns.txt', 'r') as f:
    feature_cols = [line.strip() for line in f.readlines()]

print(f"\nFeatures for model: {len(feature_cols)}")

# 3. Find categorical features
categorical_cols = train_df[feature_cols].select_dtypes(include=['object']).columns.tolist()
print(f"\nCategorical features: {len(categorical_cols)}")
for col in categorical_cols:
    print(f"  - {col}: {train_df[col].nunique()} unique values")

# 4. Create LabelEncoders for each categorical feature
print("\nCreating LabelEncoders...")
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    # Fit on all unique values from train + 'MISSING' for NaN
    all_values = train_df[col].dropna().astype(str).tolist() + ['MISSING']
    le.fit(all_values)
    label_encoders[col] = le
    print(f"  {col}: {len(le.classes_)} classes")

# 5. Encode categorical features in train and test
print("\nEncoding categorical features...")
X_train = train_df[feature_cols].copy()
X_test = test_df[feature_cols].copy()
y_train = train_df['p1_won'].values
y_test = test_df['p1_won'].values

for col in categorical_cols:
    # Train
    X_train[col] = X_train[col].fillna('MISSING')
    X_train[col] = label_encoders[col].transform(X_train[col].astype(str))
    
    # Test
    X_test[col] = X_test[col].fillna('MISSING')
    # Handle unseen labels in test
    X_test[col] = X_test[col].apply(lambda x: x if str(x) in label_encoders[col].classes_ else 'MISSING')
    X_test[col] = label_encoders[col].transform(X_test[col].astype(str))

# 6. Fill NaN in numeric features
print("\nFilling NaN values...")
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

print(f"Train shape after encoding: {X_train.shape}")
print(f"Test shape after encoding: {X_test.shape}")
print(f"Data types: {X_train.dtypes.value_counts().to_dict()}")

# 7. Train XGBoost
print("\n" + "="*80)
print("TRAINING XGBOOST")
print("="*80)

xgb_model = xgb.XGBClassifier(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.7,
    colsample_bytree=0.5,
    min_child_weight=5,
    gamma=0.3,
    reg_alpha=0.3,
    reg_lambda=2.0,
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50
)

# 8. Predictions
y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba >= 0.5).astype(int)

# 9. Metrics
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("\n" + "="*80)
print("RESULTS (without calibration - using base XGBoost):")
print("="*80)
print(f"Accuracy: {accuracy:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

# Check probability distribution
print(f"\nPROBABILITY DISTRIBUTION:")
print(f"Min: {y_pred_proba.min():.3f}")
print(f"Max: {y_pred_proba.max():.3f}")
print(f"Mean: {y_pred_proba.mean():.3f}")
print(f"Count >90%: {(y_pred_proba > 0.9).sum()} ({(y_pred_proba > 0.9).sum() / len(y_pred_proba):.1%})")
print(f"Count >95%: {(y_pred_proba > 0.95).sum()} ({(y_pred_proba > 0.95).sum() / len(y_pred_proba):.1%})")

# 10. Without calibration - use base model
print("\n" + "="*80)
print("NO CALIBRATION - Base XGBoost model")
print("="*80)
calibrated_model = xgb_model
y_pred_calibrated = y_pred_proba
y_pred_final = y_pred

accuracy_calibrated = accuracy
roc_auc_calibrated = roc_auc

print(f"Accuracy: {accuracy_calibrated:.4f}")
print(f"ROC-AUC: {roc_auc_calibrated:.4f}")

# 12. Save model and encoders
print("\nSaving model and encoders...")

# Save calibrated model
model_path = Path('../models/xgboost_calibrated_model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(calibrated_model, f)
print(f"  Model: {model_path}")

# Save LabelEncoders
encoders_path = Path('../models/label_encoders.pkl')
with open(encoders_path, 'wb') as f:
    pickle.dump(label_encoders, f)
print(f"  Label Encoders: {encoders_path}")

# Save metrics
metrics_path = Path('../models/model_metrics.txt')
with open(metrics_path, 'w') as f:
    f.write(f"Accuracy: {accuracy_calibrated:.4f}\n")
    f.write(f"ROC-AUC: {roc_auc_calibrated:.4f}\n")
print(f"  Metrics: {metrics_path}")

print("\nDONE! Model retrained and saved.")
print(f"   Accuracy: {accuracy_calibrated:.1%}")
print(f"   ROC-AUC: {roc_auc_calibrated:.4f}")
