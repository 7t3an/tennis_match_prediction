# 📂 ОГЛЯД КОЖНОГО ФАЙЛУ В ПРОЕКТІ

## 🗂️ ГОЛОВНІ ФАЙЛИ

### app.py (844 рядки)
**Призначення:** Головний файл Streamlit додатку - веб-інтерфейс для користувачів

**Секції:**
1. **Imports** (рядки 1-15)
   ```python
   import streamlit as st
   import pandas as pd
   import pickle
   import plotly.graph_objects as go
   ```

2. **load_model()** (рядки 63-82)
   - Завантажує натреновану XGBoost модель
   - Завантажує LabelEncoders для категорій
   - Завантажує список features
   - Кешується для швидкості

3. **load_database()** (рядки 84-106)
   - Завантажує train_features.csv + test_features.csv
   - Об'єднує їх для повної історії
   - Повертає full_db та test_db
   - Кешується

4. **get_unique_players()** (рядки 108-143)
   - Витягує унікальних гравців з test_db (2025)
   - Сортує за останнім матчем
   - Повертає список для selectbox

5. **get_player_stats()** (рядки 145-205)
   - Знаходить ОСТАННІЙ матч гравця
   - Визначає позицію (P1 чи P2)
   - Витягує всі stats (rank, points, rolling stats)
   - Повертає dict з даними

6. **calculate_features()** (рядки 207-370) ⭐ НАЙВАЖЛИВІША
   - Отримує stats обох гравців
   - Обчислює H2H історію
   - Створює seed features
   - Додає user input (surface, level)
   - **НОРМАЛІЗУЄ** (міняє P1/P2 якщо треба)
   - Повертає features + swapped flag

7. **UI - Header** (рядки 372-400)
   - Заголовок додатку
   - Опис функціоналу

8. **TAB 1: Prediction** (рядки 410-700)
   ```python
   # Inputs
   p1_name = st.selectbox(...)
   p2_name = st.selectbox(...)
   surface = st.selectbox(...)
   tourney_level = st.selectbox(...)
   
   # Button
   if st.button("ЗРОБИТИ ПРОГНОЗ"):
       features, swapped = calculate_features(...)
       
       # Encoding
       for col in categorical:
           input_df[col] = encoder.transform(...)
       
       # Prediction
       prob = model.predict_proba(...)
       
       # Invert if swapped
       if swapped:
           prob = 1 - prob
       
       # Display
       st.plotly_chart(gauge_chart)
       st.write(f"{p1}: {prob1:.1%}")
       st.write(f"{p2}: {prob2:.1%}")
   ```

9. **TAB 2: Player History** (рядки 702-800)
   - Показує останні 10 матчів гравця
   - Таблиця результатів
   - Статистика W/L

**Залежності:**
- ✅ saved models/xgboost_calibrated_model.pkl
- ✅ saved models/label_encoders.pkl
- ✅ saved models/feature_columns.txt
- ✅ data/processed/train_features.csv
- ✅ data/processed/test_features.csv

---

### fix_asymmetric_features.py (180 рядків)
**Призначення:** Виправляє проблему асиметричних features

**Що робить:**
1. Завантажує train/test дані
2. Видаляє 5 асиметричних features:
   - rank_diff
   - rank_ratio
   - rank_points_diff
   - is_p1_favorite
   - seed_diff
3. **normalize_match_order()** - нормалізує дані:
   ```python
   if p2_rank < p1_rank:
       # Міняємо P1 ↔ P2
       # Інвертуємо p1_won
   ```
4. Тренує нову модель
5. Зберігає як xgboost_symmetric_model.pkl

**Коли запускати:**
```bash
python fix_asymmetric_features.py
```

**Результат:**
- ✅ saved models/xgboost_symmetric_model.pkl
- ✅ saved models/label_encoders.pkl (оновлені)
- ✅ saved models/feature_columns.txt (48 features)

---

### retrain_model.py (160 рядків)
**Призначення:** Перетренування моделі з новими параметрами

**Секції:**
1. **Load Data** (рядки 18-30)
   ```python
   train = pd.read_csv('train_features.csv')
   test = pd.read_csv('test_features.csv')
   ```

2. **Create LabelEncoders** (рядки 32-50)
   ```python
   for col in categorical_cols:
       encoder = LabelEncoder()
       encoder.fit([...all_values..., 'MISSING'])
       encoders[col] = encoder
   ```

3. **Encode Data** (рядки 52-75)
   ```python
   X_train[col] = encoder.transform(X_train[col].fillna('MISSING'))
   ```

4. **Train XGBoost** (рядки 77-110)
   ```python
   model = XGBClassifier(
       n_estimators=400,
       max_depth=4,
       learning_rate=0.03,
       ...
   )
   model.fit(X_train, y_train)
   ```

5. **Evaluate** (рядки 112-130)
   ```python
   accuracy = accuracy_score(y_test, y_pred)
   roc_auc = roc_auc_score(y_test, y_proba)
   ```

6. **Save** (рядки 132-156)
   ```python
   pickle.dump(model, 'xgboost_calibrated_model.pkl')
   pickle.dump(encoders, 'label_encoders.pkl')
   ```

**Коли запускати:**
```bash
python retrain_model.py
```

---

## 📊 DATA FILES

### data/tml/ (Сирі дані)
**Файли:** 1968.csv, 1969.csv, ..., 2025.csv, ATP_Database.csv

**Формат (winner/loser):**
```csv
tourney_id, tourney_name, surface, draw_size, tourney_level, tourney_date,
match_num, winner_id, winner_seed, winner_entry, winner_name, winner_hand,
winner_ht, winner_ioc, winner_age, loser_id, loser_seed, ...
w_ace, w_df, w_svpt, w_1stIn, w_1stWon, w_2ndWon, w_SvGms, w_bpSaved, w_bpFaced,
l_ace, l_df, l_svpt, ...
minutes, score, ...
```

**Статистика:**
- 1968-2025: ~300,000 матчів
- Колонок: ~50
- Розмір: ~80 MB

---

### data/processed/train_2012_2024.csv
**Призначення:** Очищені дані 2012-2024 для тренування

**Створюється:** notebooks/Prepare_ETL.ipynb

**Зміни від сирих даних:**
- ✅ Видалені дублікати
- ✅ NaN оброблені
- ✅ Типи даних конвертовані
- ✅ Дати парсені
- ✅ Валідація match_id

**Статистика:**
- Рядків: 73,248 матчів
- Колонок: ~62
- Період: 2012-2024

---

### data/processed/train_features.csv
**Призначення:** Готові features для тренування моделі

**Створюється:** notebooks/Feature_Engineering.ipynb

**Трансформації:**
1. **Winner/Loser → P1/P2**
   ```python
   # Кожен матч створює 2 рядки:
   # Original: Alcaraz beat Draper
   # Row 1: P1=Alcaraz, P2=Draper, p1_won=1
   # Row 2: P1=Draper, P2=Alcaraz, p1_won=0
   ```

2. **Rolling Statistics**
   ```python
   # Для кожного гравця обчислює середнє за 10 матчів:
   p1_ace_roll10 = mean(last_10_matches['ace'])
   ```

3. **H2H Features**
   ```python
   h2h_matches = all_previous_matches(P1, P2)
   h2h_p1_wins = count(P1_wins)
   h2h_p1_win_rate = h2h_p1_wins / total
   ```

4. **Seed Features**
   ```python
   is_p1_seeded = not isna(p1_seed)
   p1_seed_tier = categorize(p1_seed)
   ```

**Колонки (63 total):**
```
# Metadata (10)
match_id, tourney_id, tourney_date, p1_name, p2_name, ...

# Target (1)
p1_won (0 або 1)

# Tournament (4)
surface, tourney_level, draw_size, indoor

# P1 Basic (8)
p1_rank, p1_rank_points, p1_age, p1_hand, p1_ht, p1_ioc, p1_seed, p1_entry

# P1 Rolling Stats (9)
p1_ace_roll10, p1_df_roll10, p1_svpt_roll10, p1_1stIn_roll10, 
p1_1stWon_roll10, p1_2ndWon_roll10, p1_SvGms_roll10, 
p1_bpSaved_roll10, p1_bpFaced_roll10

# P2 Basic (8) - те саме

# P2 Rolling Stats (9) - те саме

# H2H (4)
h2h_p1_wins, h2h_p2_wins, h2h_total_matches, h2h_p1_win_rate

# Seed Features (4)
is_p1_seeded, is_p2_seeded, p1_seed_tier, p2_seed_tier

# Encoded (2) - дублікати для backward compatibility
tourney_level_encoded, surface_encoded
```

**Статистика:**
- Рядків: 73,248
- Колонок: 63
- Розмір: ~20 MB

---

### data/processed/test_features.csv
**Те саме що train_features.csv, але для 2025 року**

**Статистика:**
- Рядків: 5,822
- Колонок: 63
- Період: 2025

---

## 🤖 MODEL FILES

### saved models/xgboost_calibrated_model.pkl
**Призначення:** Натренована XGBoost модель

**Тип:** sklearn-compatible pickle object

**Архітектура:**
```
XGBClassifier(
    n_estimators=400,    # 400 decision trees
    max_depth=4,         # Depth of each tree
    learning_rate=0.03,  # Step size
    ...
)
```

**Розмір:** ~2-3 MB

**Як використовувати:**
```python
model = pickle.load(open('xgboost_calibrated_model.pkl', 'rb'))
proba = model.predict_proba(X)  # Shape: (n_samples, 2)
# proba[:, 0] - ймовірність що p1_won = 0
# proba[:, 1] - ймовірність що p1_won = 1
```

---

### saved models/label_encoders.pkl
**Призначення:** LabelEncoder'и для категоріальних features

**Тип:** Dictionary of LabelEncoder objects

**Структура:**
```python
{
    'surface': LabelEncoder(['Carpet', 'Clay', 'Grass', 'Hard', 'MISSING']),
    'tourney_level': LabelEncoder(['A', 'C', 'D', 'F', 'G', 'M', ...]),
    'p1_entry': LabelEncoder(['A', 'DA', 'LL', 'MISSING', 'PR', 'Q', ...]),
    'p1_hand': LabelEncoder(['L', 'MISSING', 'R', 'U']),
    'p1_ioc': LabelEncoder(['ARG', 'AUS', ..., 'USA', 'MISSING']),
    # ... і т.д. для всіх 10 категоріальних
}
```

**Як використовувати:**
```python
encoders = pickle.load(open('label_encoders.pkl', 'rb'))

# Encode
X['surface'] = encoders['surface'].transform(['Hard', 'Clay'])
# Result: [3, 1]

# Decode
encoders['surface'].inverse_transform([3, 1])
# Result: ['Hard', 'Clay']
```

---

### saved models/feature_columns.txt
**Призначення:** Список features у правильному порядку

**Формат:**
```
surface
draw_size
tourney_level
indoor
p1_seed
p1_entry
p1_hand
p1_ht
p1_ioc
p1_age
p1_rank
p1_rank_points
... (48 total)
```

**Використання:**
```python
feature_cols = open('feature_columns.txt').read().splitlines()
X = df[feature_cols]  # Правильний порядок!
```

---

### saved models/model_metrics.txt
**Призначення:** Метрики моделі після тренування

**Формат:**
```
Accuracy: 0.6532
ROC-AUC: 0.6408
Symmetric: True
```

---

## 📓 NOTEBOOKS

### notebooks/Prepare_ETL.ipynb
**Призначення:** Завантаження та очистка сирих даних

**Секції:**

1. **Load CSV Files** (cells 1-10)
   ```python
   csv_files = glob('data/tml/*.csv')
   dfs = [pd.read_csv(f) for f in csv_files]
   df_all = pd.concat(dfs)
   ```

2. **Data Cleaning** (cells 11-30)
   - Remove duplicates
   - Handle NaN values
   - Convert data types
   - Validate match_id

3. **Date Processing** (cells 31-40)
   ```python
   df['tourney_date'] = pd.to_datetime(df['tourney_date'])
   ```

4. **Train/Test Split** (cells 41-50)
   ```python
   train = df[df['tourney_date'].dt.year <= 2024]
   test = df[df['tourney_date'].dt.year == 2025]
   ```

5. **Save** (cells 51-55)
   ```python
   train.to_csv('train_2012_2024.csv')
   test.to_csv('test_2025.csv')
   ```

**Output:**
- ✅ data/processed/train_2012_2024.csv
- ✅ data/processed/test_2025.csv

---

### notebooks/Feature_Engineering.ipynb
**Призначення:** Створення features для ML

**Секції:**

1. **Winner/Loser → P1/P2** (cells 1-20)
   ```python
   # Для кожного матчу створюємо 2 рядки
   row1 = {
       'p1_name': winner_name,
       'p2_name': loser_name,
       'p1_won': 1
   }
   row2 = {
       'p1_name': loser_name,
       'p2_name': winner_name,
       'p1_won': 0
   }
   ```

2. **Rolling Statistics** (cells 21-60)
   ```python
   def calculate_rolling_stats(df, player, stat_col, window=10):
       player_matches = df[df['player'] == player].sort_by('date')
       rolling = player_matches[stat_col].rolling(window).mean()
       return rolling
   
   df['p1_ace_roll10'] = calculate_rolling_stats(df, 'p1_name', 'ace')
   ```

3. **Head-to-Head** (cells 61-80)
   ```python
   def calculate_h2h(df, p1, p2, current_date):
       h2h = df[
           ((df['p1_name']==p1) & (df['p2_name']==p2)) |
           ((df['p1_name']==p2) & (df['p2_name']==p1))
       ]
       h2h = h2h[h2h['date'] < current_date]  # Тільки попередні!
       
       return {
           'h2h_p1_wins': len(h2h[h2h['winner']==p1]),
           'h2h_p2_wins': len(h2h[h2h['winner']==p2]),
           'h2h_total': len(h2h)
       }
   ```

4. **Seed Features** (cells 81-100)
   ```python
   def get_seed_tier(seed):
       if pd.isna(seed): return 'MISSING'
       if seed <= 4: return 'Top4'
       if seed <= 8: return 'Top8'
       if seed <= 16: return 'Top16'
       if seed <= 32: return 'Top32'
       return 'Other'
   
   df['p1_seed_tier'] = df['p1_seed'].apply(get_seed_tier)
   ```

5. **Save** (cells 101-110)
   ```python
   train_features.to_csv('train_features.csv')
   test_features.to_csv('test_features.csv')
   ```

**Output:**
- ✅ data/processed/train_features.csv (73,248 rows)
- ✅ data/processed/test_features.csv (5,822 rows)

---

### notebooks/Model_Training.ipynb
**Призначення:** Тренування та оцінка моделі

**Секції:**

1. **Load Features** (cells 1-10)
   ```python
   train = pd.read_csv('train_features.csv')
   test = pd.read_csv('test_features.csv')
   
   X_train = train[feature_cols]
   y_train = train['p1_won']
   ```

2. **Label Encoding** (cells 11-30)
   ```python
   encoders = {}
   for col in categorical_cols:
       encoder = LabelEncoder()
       encoder.fit(train[col].astype(str).tolist() + ['MISSING'])
       X_train[col] = encoder.transform(X_train[col].fillna('MISSING'))
       encoders[col] = encoder
   ```

3. **Train XGBoost** (cells 31-50)
   ```python
   model = XGBClassifier(...)
   model.fit(X_train, y_train)
   ```

4. **Evaluate** (cells 51-70)
   ```python
   y_pred = model.predict(X_test)
   y_proba = model.predict_proba(X_test)[:, 1]
   
   accuracy = accuracy_score(y_test, y_pred)
   roc_auc = roc_auc_score(y_test, y_proba)
   
   print(f"Accuracy: {accuracy:.4f}")
   print(f"ROC-AUC: {roc_auc:.4f}")
   ```

5. **Feature Importance** (cells 71-90)
   ```python
   importance = pd.DataFrame({
       'feature': feature_cols,
       'importance': model.feature_importances_
   }).sort_values('importance', ascending=False)
   
   plt.barh(importance['feature'][:20], importance['importance'][:20])
   ```

6. **Save Model** (cells 91-100)
   ```python
   pickle.dump(model, 'xgboost_calibrated_model.pkl')
   pickle.dump(encoders, 'label_encoders.pkl')
   ```

**Output:**
- ✅ saved models/xgboost_calibrated_model.pkl
- ✅ saved models/label_encoders.pkl
- ✅ saved models/feature_columns.txt
- ✅ saved models/model_metrics.txt

---

### notebooks/main_eda.ipynb
**Призначення:** Exploratory Data Analysis

**Секції:**
1. Базова статистика
2. Розподіл features
3. Кореляції
4. Візуалізації
5. Insights для feature engineering

**Не впливає на production код** - тільки для аналізу.

---

## 📋 КОНФІГУРАЦІЯ

### requirements.txt
**Призначення:** Python залежності

```
streamlit==1.28.0      # Web framework
pandas==2.0.0          # Data manipulation
numpy==1.24.0          # Arrays
scikit-learn==1.3.0    # ML utilities
xgboost==2.0.0         # Gradient boosting
plotly==5.17.0         # Interactive charts
pickle5==0.0.12        # Serialization
```

### config.py
**Призначення:** Налаштування проекту

```python
# Paths
DATA_DIR = 'data/'
TML_DIR = 'data/tml/'
PROCESSED_DIR = 'data/processed/'
MODELS_DIR = 'saved models/'

# Model params
N_ESTIMATORS = 400
MAX_DEPTH = 4
LEARNING_RATE = 0.03

# Features
CATEGORICAL_FEATURES = [
    'surface', 'tourney_level', 'p1_entry', 'p1_hand', 
    'p1_ioc', 'p2_entry', 'p2_hand', 'p2_ioc',
    'p1_seed_tier', 'p2_seed_tier'
]
```

---

## 🗺️ FLOW ДІАГРАМА

```
┌─────────────────────┐
│   Сирі CSV файли    │
│   data/tml/*.csv    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Prepare_ETL.ipynb   │  ← Очистка + Split
│  - Remove dups      │
│  - Handle NaN       │
│  - Train/Test split │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────────┐
│ train_2012_2024.csv             │
│ test_2025.csv                   │
└──────────┬──────────────────────┘
           │
           ▼
┌─────────────────────┐
│Feature_Engineering  │  ← Створення features
│  - P1/P2 transform  │
│  - Rolling stats    │
│  - H2H              │
│  - Seed features    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────────┐
│ train_features.csv              │
│ test_features.csv               │
└──────────┬──────────────────────┘
           │
           ▼
┌─────────────────────┐
│ fix_asymmetric.py   │  ← Виправлення
│  - Remove асим.     │
│  - Normalize order  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ retrain_model.py    │  ← Тренування
│  - Label encoding   │
│  - XGBoost train    │
│  - Save model       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────────┐
│ xgboost_calibrated_model.pkl    │
│ label_encoders.pkl              │
│ feature_columns.txt             │
└──────────┬──────────────────────┘
           │
           ▼
┌─────────────────────┐
│      app.py         │  ← Production
│  - Load model       │
│  - User interface   │
│  - Predictions      │
└─────────────────────┘
           │
           ▼
┌─────────────────────┐
│    Користувач       │
│  http://localhost   │
│      :8501          │
└─────────────────────┘
```

---

## 🎯 РЕЗЮМЕ

**Файли за важливістю:**

1. **app.py** ⭐⭐⭐⭐⭐ - Головний додаток
2. **fix_asymmetric_features.py** ⭐⭐⭐⭐ - Виправлення моделі
3. **retrain_model.py** ⭐⭐⭐⭐ - Перетренування
4. **train_features.csv** ⭐⭐⭐⭐ - Дані для тренування
5. **test_features.csv** ⭐⭐⭐⭐ - Дані для прогнозів
6. **xgboost_calibrated_model.pkl** ⭐⭐⭐⭐⭐ - Модель
7. **label_encoders.pkl** ⭐⭐⭐⭐⭐ - Encoders
8. **Feature_Engineering.ipynb** ⭐⭐⭐ - Створення features
9. **Prepare_ETL.ipynb** ⭐⭐⭐ - Підготовка даних
10. **Model_Training.ipynb** ⭐⭐ - Тренування (можна замінити retrain_model.py)

**Що потрібно знати для роботи:**
- 🐍 Python basics
- 📊 Pandas for data manipulation
- 🤖 XGBoost basics
- 🌐 Streamlit для UI
- 📈 ML concepts (features, encoding, train/test)
