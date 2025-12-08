"""
Перетренування моделі з правильним encoding для categorical features
"""
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
print("ПЕРЕТРЕНУВАННЯ МОДЕЛІ З ПРАВИЛЬНИМ ENCODING")
print("="*80)

# 1. Завантаження даних
print("\n📂 Завантаження даних...")
train_df = pd.read_csv('data/processed/train_features.csv')
test_df = pd.read_csv('data/processed/test_features.csv')

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# 2. Завантаження feature columns
with open('saved models/feature_columns.txt', 'r') as f:
    feature_cols = [line.strip() for line in f.readlines()]

print(f"\nFeatures для моделі: {len(feature_cols)}")

# 3. Знаходимо categorical features
categorical_cols = train_df[feature_cols].select_dtypes(include=['object']).columns.tolist()
print(f"\n🔤 Categorical features: {len(categorical_cols)}")
for col in categorical_cols:
    print(f"  - {col}: {train_df[col].nunique()} unique values")

# 4. Створюємо LabelEncoders для КОЖНОГО categorical feature
print("\n🔧 Створення LabelEncoders...")
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    # Fit на всіх унікальних значеннях з train + 'MISSING' для NaN
    all_values = train_df[col].dropna().astype(str).tolist() + ['MISSING']
    le.fit(all_values)
    label_encoders[col] = le
    print(f"  ✅ {col}: {len(le.classes_)} classes")

# 5. Encode categorical features в train та test
print("\n🔢 Encoding categorical features...")
X_train = train_df[feature_cols].copy()
X_test = test_df[feature_cols].copy()
y_train = train_df['p1_won'].values
y_test = test_df['p1_won'].values

for col in categorical_cols:
    # Train
    X_train[col] = X_train[col].fillna('MISSING')  # Заповнюємо NaN
    X_train[col] = label_encoders[col].transform(X_train[col].astype(str))
    
    # Test
    X_test[col] = X_test[col].fillna('MISSING')
    # Handle unseen labels в test
    X_test[col] = X_test[col].apply(lambda x: x if str(x) in label_encoders[col].classes_ else 'MISSING')
    X_test[col] = label_encoders[col].transform(X_test[col].astype(str))

# 6. Заповнюємо NaN у числових features
print("\n🔧 Заповнення NaN values...")
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

print(f"Train shape after encoding: {X_train.shape}")
print(f"Test shape after encoding: {X_test.shape}")
print(f"Типи даних: {X_train.dtypes.value_counts().to_dict()}")

# 7. Тренування XGBoost
print("\n" + "="*80)
print("ТРЕНУВАННЯ XGBOOST")
print("="*80)

xgb_model = xgb.XGBClassifier(
    n_estimators=400,
    max_depth=4,  # Ще менше для зменшення overfitting
    learning_rate=0.03,  # Повільніше навчання
    subsample=0.7,  # Менше рядків на дерево
    colsample_bytree=0.5,  # ЗНАЧНО менше features - зменшує вагу rank
    min_child_weight=5,  # Більше мінімальних даних
    gamma=0.3,  # Більша регуляризація
    reg_alpha=0.3,  # Більша L1
    reg_lambda=2.0,  # Більша L2
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
print("РЕЗУЛЬТАТИ (БЕЗ калібрації - використовуємо базову XGBoost):")
print("="*80)
print(f"Accuracy: {accuracy:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

# Перевіряємо розподіл ймовірностей
print(f"\nРОЗПОДІЛ ЙМОВІРНОСТЕЙ:")
print(f"Min: {y_pred_proba.min():.3f}")
print(f"Max: {y_pred_proba.max():.3f}")
print(f"Mean: {y_pred_proba.mean():.3f}")
print(f"Кількість >90%: {(y_pred_proba > 0.9).sum()} ({(y_pred_proba > 0.9).sum() / len(y_pred_proba):.1%})")
print(f"Кількість >95%: {(y_pred_proba > 0.95).sum()} ({(y_pred_proba > 0.95).sum() / len(y_pred_proba):.1%})")

# 10. БЕЗ калібрації - використовуємо базову модель
print("\n" + "="*80)
print("БЕЗ КАЛІБРАЦІЇ - Базова XGBoost модель")
print("="*80)
calibrated_model = xgb_model  # Просто копіюємо базову модель
y_pred_calibrated = y_pred_proba
y_pred_final = y_pred

accuracy_calibrated = accuracy
roc_auc_calibrated = roc_auc

print(f"Accuracy: {accuracy_calibrated:.4f}")
print(f"ROC-AUC: {roc_auc_calibrated:.4f}")

# 12. Збереження моделі та encoders
print("\n💾 Збереження моделі та encoders...")

# Зберігаємо калібровану модель
model_path = Path('saved models/xgboost_calibrated_model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(calibrated_model, f)
print(f"  ✅ Модель: {model_path}")

# Зберігаємо LabelEncoders
encoders_path = Path('saved models/label_encoders.pkl')
with open(encoders_path, 'wb') as f:
    pickle.dump(label_encoders, f)
print(f"  ✅ Label Encoders: {encoders_path}")

# Зберігаємо metrics
metrics_path = Path('saved models/model_metrics.txt')
with open(metrics_path, 'w') as f:
    f.write(f"Accuracy: {accuracy_calibrated:.4f}\n")
    f.write(f"ROC-AUC: {roc_auc_calibrated:.4f}\n")
print(f"  ✅ Metrics: {metrics_path}")

print("\n✅ ГОТОВО! Модель перетренована та збережена.")
print(f"   Accuracy: {accuracy_calibrated:.1%}")
print(f"   ROC-AUC: {roc_auc_calibrated:.4f}")
