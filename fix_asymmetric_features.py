"""
ВИПРАВЛЕННЯ АСИМЕТРИЧНИХ FEATURES

Проблема: модель дає різні прогнози для "Alcaraz vs Sinner" і "Sinner vs Alcaraz"
Причина: features залежать від порядку гравців (rank_diff, rank_ratio, seed_diff)

Рішення: 
1. Видаляємо асиметричні features
2. Модель завжди передбачає ймовірність перемоги КРАЩОГО гравця
3. В app.py визначаємо хто кращий і показуємо правильний результат
"""
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import xgboost as xgb
from pathlib import Path

print("="*80)
print("ВИПРАВЛЕННЯ АСИМЕТРИЧНИХ FEATURES")
print("="*80)

# 1. Завантаження даних
print("\n📂 Завантаження даних...")
train_df = pd.read_csv('data/processed/train_features.csv')
test_df = pd.read_csv('data/processed/test_features.csv')

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# 2. Завантаження feature columns
with open('saved models/feature_columns.txt', 'r') as f:
    old_feature_cols = [line.strip() for line in f.readlines()]

print(f"\nСтарі features: {len(old_feature_cols)}")

# 3. Видаляємо АСИМЕТРИЧНІ features
asymmetric_features = [
    'rank_diff',           # p1_rank - p2_rank
    'rank_ratio',          # p1_rank / p2_rank
    'rank_points_diff',    # p1_points - p2_points
    'is_p1_favorite',      # p1_rank < p2_rank
    'seed_diff',           # p1_seed - p2_seed
]

print(f"\n❌ Видаляємо асиметричні features:")
for feat in asymmetric_features:
    if feat in old_feature_cols:
        print(f"  - {feat}")

# Нові features БЕЗ асиметричних
new_feature_cols = [f for f in old_feature_cols if f not in asymmetric_features]
print(f"\nНові features: {len(new_feature_cols)}")

# 4. ПЕРЕТВОРЕННЯ ДАНИХ: завжди кращий гравець = P1
print("\n🔄 Перетворення: завжди P1 = кращий за рангом")

def normalize_match_order(df):
    """Міняємо місцями гравців якщо P2 має кращий ранг"""
    df_norm = df.copy()
    
    # Знаходимо рядки де P2 кращий (менший rank)
    swap_mask = df_norm['p2_rank'] < df_norm['p1_rank']
    
    print(f"  Потрібно поміняти місцями: {swap_mask.sum()} з {len(df_norm)} ({swap_mask.sum()/len(df_norm):.1%})")
    
    # Міняємо місцями ВСІ P1/P2 features
    for col in df_norm.columns:
        if col.startswith('p1_') and not col.endswith('_won'):
            p2_col = col.replace('p1_', 'p2_')
            if p2_col in df_norm.columns:
                df_norm.loc[swap_mask, [col, p2_col]] = df_norm.loc[swap_mask, [p2_col, col]].values
    
    # Міняємо H2H features
    if 'h2h_p1_wins' in df_norm.columns and 'h2h_p2_wins' in df_norm.columns:
        df_norm.loc[swap_mask, ['h2h_p1_wins', 'h2h_p2_wins']] = \
            df_norm.loc[swap_mask, ['h2h_p2_wins', 'h2h_p1_wins']].values
    
    # ІНВЕРТУЄМО результат: якщо поміняли місцями, то p1_won = 1-p1_won
    if 'p1_won' in df_norm.columns:
        df_norm.loc[swap_mask, 'p1_won'] = 1 - df_norm.loc[swap_mask, 'p1_won']
    
    return df_norm

train_norm = normalize_match_order(train_df)
test_norm = normalize_match_order(test_df)

# Перевірка
print(f"\nПеревірка: P1 завжди має кращий або рівний rank:")
print(f"  Train: P1_rank <= P2_rank: {(train_norm['p1_rank'] <= train_norm['p2_rank']).sum()} / {len(train_norm)}")
print(f"  Test:  P1_rank <= P2_rank: {(test_norm['p1_rank'] <= test_norm['p2_rank']).sum()} / {len(test_norm)}")

# 5. Створюємо LabelEncoders
categorical_cols = train_norm[new_feature_cols].select_dtypes(include=['object']).columns.tolist()
print(f"\n🔤 Categorical features: {len(categorical_cols)}")

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    all_values = train_norm[col].dropna().astype(str).tolist() + ['MISSING']
    le.fit(all_values)
    label_encoders[col] = le
    print(f"  ✅ {col}: {len(le.classes_)} classes")

# 6. Encoding
print("\n🔢 Encoding categorical features...")
X_train = train_norm[new_feature_cols].copy()
X_test = test_norm[new_feature_cols].copy()
y_train = train_norm['p1_won'].values
y_test = test_norm['p1_won'].values

for col in categorical_cols:
    X_train[col] = X_train[col].fillna('MISSING')
    X_train[col] = label_encoders[col].transform(X_train[col].astype(str))
    
    X_test[col] = X_test[col].fillna('MISSING')
    X_test[col] = X_test[col].apply(lambda x: x if str(x) in label_encoders[col].classes_ else 'MISSING')
    X_test[col] = label_encoders[col].transform(X_test[col].astype(str))

X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

print(f"Train shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")

# 7. Тренування
print("\n" + "="*80)
print("ТРЕНУВАННЯ СИМЕТРИЧНОЇ МОДЕЛІ")
print("="*80)

xgb_model = xgb.XGBClassifier(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.7,
    colsample_bytree=0.6,
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

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("\n" + "="*80)
print("РЕЗУЛЬТАТИ:")
print("="*80)
print(f"Accuracy: {accuracy:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

print(f"\nРОЗПОДІЛ ЙМОВІРНОСТЕЙ:")
print(f"Min: {y_pred_proba.min():.3f}")
print(f"Max: {y_pred_proba.max():.3f}")
print(f"Mean: {y_pred_proba.mean():.3f}")
print(f">90%: {(y_pred_proba > 0.9).sum()} ({(y_pred_proba > 0.9).sum() / len(y_pred_proba):.1%})")
print(f">95%: {(y_pred_proba > 0.95).sum()} ({(y_pred_proba > 0.95).sum() / len(y_pred_proba):.1%})")

# 9. Збереження
print("\n💾 Збереження...")

model_path = Path('saved models/xgboost_symmetric_model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(xgb_model, f)
print(f"  ✅ Модель: {model_path}")

encoders_path = Path('saved models/label_encoders.pkl')
with open(encoders_path, 'wb') as f:
    pickle.dump(label_encoders, f)
print(f"  ✅ Encoders: {encoders_path}")

features_path = Path('saved models/feature_columns.txt')
with open(features_path, 'w') as f:
    for feat in new_feature_cols:
        f.write(f"{feat}\n")
print(f"  ✅ Features ({len(new_feature_cols)}): {features_path}")

metrics_path = Path('saved models/model_metrics.txt')
with open(metrics_path, 'w') as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"ROC-AUC: {roc_auc:.4f}\n")
    f.write(f"Symmetric: True\n")
print(f"  ✅ Metrics: {metrics_path}")

print("\n✅ ГОТОВО! Симетрична модель готова.")
print(f"   Accuracy: {accuracy:.1%}")
print(f"   ROC-AUC: {roc_auc:.4f}")
print(f"\n📝 ВАЖЛИВО: Тепер модель завжди передбачає ймовірність перемоги КРАЩОГО гравця (з меншим rank).")
print("   В app.py потрібно визначити хто кращий і правильно показати результат.")
