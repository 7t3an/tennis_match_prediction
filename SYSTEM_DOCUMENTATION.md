# 🎾 ПОВНА ДОКУМЕНТАЦІЯ СИСТЕМИ ПРОГНОЗУВАННЯ ТЕНІСНИХ МАТЧІВ

## 📁 СТРУКТУРА ПРОЕКТУ

```
tennis_match_prediction/
├── 📂 data/                          # Всі дані
│   ├── tml/                          # Сирі дані з Tennis Match Library
│   │   ├── 1968.csv ... 2025.csv   # Історія матчів 1968-2025
│   │   └── ATP_Database.csv         # Повна база ATP
│   └── processed/                    # Оброблені дані готові для ML
│       ├── train_2012_2024.csv      # Тренувальні дані (2012-2024)
│       ├── train_features.csv       # Тренувальні features + target
│       ├── test_2025.csv            # Тестові дані (2025)
│       └── test_features.csv        # Тестові features + target
│
├── 📂 notebooks/                     # Jupyter notebooks для аналізу
│   ├── Prepare_ETL.ipynb            # 1. Завантаження та очистка даних
│   ├── Feature_Engineering.ipynb    # 2. Створення features
│   ├── Model_Training.ipynb         # 3. Тренування моделі
│   └── main_eda.ipynb              # Exploratory Data Analysis
│
├── 📂 saved models/                  # Збережені моделі та артефакти
│   ├── xgboost_calibrated_model.pkl # Натренована XGBoost модель
│   ├── xgboost_symmetric_model.pkl  # Симетрична модель (нова)
│   ├── label_encoders.pkl           # LabelEncoder'и для категорій
│   ├── feature_columns.txt          # Список features (48 штук)
│   └── model_metrics.txt            # Метрики моделі
│
├── 🐍 app.py                         # ГОЛОВНИЙ ФАЙЛ - Streamlit інтерфейс
├── 🐍 fix_asymmetric_features.py     # Скрипт виправлення асиметричних features
├── 🐍 retrain_model.py               # Скрипт перетренування моделі
├── 📋 requirements.txt               # Python залежності
└── 📄 README.md                      # Документація проекту
```

---

## 🔄 ПОВНИЙ ЦИКЛ РОБОТИ СИСТЕМИ

### ЕТАП 1: ПІДГОТОВКА ДАНИХ (notebooks/Prepare_ETL.ipynb)

**Що робить:**
1. Завантажує сирі CSV файли з `data/tml/` (1968-2025)
2. Об'єднує їх в один великий DataFrame
3. Очищує дані:
   - Видаляє дублікати
   - Обробляє NaN значення
   - Конвертує типи даних
   - Валідує формати
4. Створює колонки:
   - `tourney_date` (дата турніру)
   - `winner_name`, `loser_name`
   - `winner_rank`, `loser_rank`
   - Статистика матчу: `w_ace`, `w_df`, `w_svpt`, etc.
5. Зберігає в `data/processed/`:
   - `train_2012_2024.csv` (73,248 матчів)
   - `test_2025.csv` (5,822 матчів)

**Вихідні дані:**
```
Колонки: ~62 штуки
Рядки: 79,070 матчів (1968-2025)
Розмір: ~15 MB
```

---

### ЕТАП 2: FEATURE ENGINEERING (notebooks/Feature_Engineering.ipynb)

**Що робить:**

#### 2.1 Трансформує дані з формату winner/loser → P1/P2

**Було (winner/loser format):**
```csv
match_id, winner_name, loser_name, winner_rank, loser_rank, ...
1, Alcaraz, Draper, 1, 5, ...
```

**Стало (P1/P2 format):**
```csv
match_id, p1_name, p2_name, p1_rank, p2_rank, p1_won
1, Alcaraz, Draper, 1, 5, 1      # P1 виграв
2, Draper, Alcaraz, 5, 1, 0      # P1 програв
```

**Чому? Щоб модель вчилась на обох гравцях рівномірно!**

#### 2.2 Обчислює Rolling Statistics (форма гравця)

Для кожного гравця в кожному матчі обчислює **середнє за останні 10 матчів**:

```python
# Приклад для Carlos Alcaraz
matches = df[df['p1_name'] == 'Alcaraz'].sort_by('date')

for i, match in enumerate(matches):
    previous_10 = matches[i-10:i]  # Попередні 10 матчів
    
    match['p1_ace_roll10'] = previous_10['ace'].mean()
    match['p1_df_roll10'] = previous_10['df'].mean()
    match['p1_1stIn_roll10'] = previous_10['1stIn'].mean()
    # ... і т.д. для 9 статистик
```

**Статистики:**
- `ace_roll10` - к-ть ейсів
- `df_roll10` - подвійні помилки
- `svpt_roll10` - к-ть подач
- `1stIn_roll10` - перша подача в корт
- `1stWon_roll10` - виграно на 1й подачі
- `2ndWon_roll10` - виграно на 2й подачі
- `SvGms_roll10` - сервіс гейми
- `bpSaved_roll10` - врятовані break points
- `bpFaced_roll10` - break points проти

#### 2.3 Обчислює Head-to-Head (H2H)

Для кожного матчу рахує історію зустрічей:

```python
# Для матчу Alcaraz vs Draper
h2h_matches = df[
    ((p1=='Alcaraz') & (p2=='Draper')) | 
    ((p1=='Draper') & (p2=='Alcaraz'))
][:current_match]  # Тільки попередні матчі!

h2h_p1_wins = count(Alcaraz wins)
h2h_p2_wins = count(Draper wins)
h2h_total_matches = len(h2h_matches)
h2h_p1_win_rate = h2h_p1_wins / h2h_total_matches
```

#### 2.4 Створює додаткові features

```python
# Seed features
is_p1_seeded = not pd.isna(p1_seed)
is_p2_seeded = not pd.isna(p2_seed)
p1_seed_tier = 'Top4' if seed <= 4 else 'Top8' if seed <= 8 else ...

# Encoded features (дублікати для backward compatibility)
tourney_level_encoded = tourney_level
surface_encoded = surface
```

**ВАЖЛИВО:** Асиметричні features (`rank_diff`, `rank_ratio`, etc.) були **ВИДАЛЕНІ** в новій версії для симетричності!

#### 2.5 Зберігає результат

```python
# Фінальні файли
train_features.csv:  73,248 rows × 63 columns
test_features.csv:   5,822 rows × 63 columns

# Колонки:
- 10 категоріальних (surface, tourney_level, hand, ioc, entry, seed_tier)
- 43 числових (ranks, points, age, rolling stats, h2h)
- 10 metadata (names, dates, ids)
```

---

### ЕТАП 3: ВИПРАВЛЕННЯ АСИМЕТРІЇ (fix_asymmetric_features.py)

**ПРОБЛЕМА:** Модель давала різні прогнози для "A vs B" і "B vs A"

**ПРИЧИНА:** Features залежали від порядку:
```python
rank_diff = p1_rank - p2_rank  # Alcaraz(1) - Draper(5) = -4
                                # Draper(5) - Alcaraz(1) = +4  ← РІЗНЕ!
```

**РІШЕННЯ:**

1. **Видаляє 5 асиметричних features:**
   - ❌ `rank_diff` (p1_rank - p2_rank)
   - ❌ `rank_ratio` (p1_rank / p2_rank)
   - ❌ `rank_points_diff` (p1_rank_points - p2_rank_points)
   - ❌ `is_p1_favorite` (p1_rank < p2_rank)
   - ❌ `seed_diff` (p1_seed - p2_seed)

2. **Нормалізує дані: завжди P1 = кращий гравець**

```python
def normalize_match_order(df):
    # Якщо P2 має кращий ранг (менше число) - міняємо місцями
    swap_mask = df['p2_rank'] < df['p1_rank']
    
    # Міняємо ВСІ P1/P2 колонки
    df.loc[swap_mask, ['p1_rank', 'p2_rank']] = \
        df.loc[swap_mask, ['p2_rank', 'p1_rank']].values
    
    # Інвертуємо результат
    df.loc[swap_mask, 'p1_won'] = 1 - df.loc[swap_mask, 'p1_won']
```

**Результат:**
- Тепер у ВСІХ матчах: `p1_rank <= p2_rank`
- Модель вчиться передбачати: "Яка ймовірність що КРАЩИЙ гравець виграє?"

3. **Зберігає симетричну модель:**
```
saved models/xgboost_symmetric_model.pkl
Features: 48 (було 53)
Accuracy: 65.3%
ROC-AUC: 0.6408
```

---

### ЕТАП 4: ТРЕНУВАННЯ МОДЕЛІ (retrain_model.py або Model_Training.ipynb)

**Алгоритм тренування:**

```python
# 1. Завантажує дані
X_train = train_features[feature_columns]  # 48 features
y_train = train_features['p1_won']         # 0 або 1

# 2. Encode категоріальні features
categorical_cols = ['surface', 'tourney_level', 'p1_entry', 'p1_hand', ...]

for col in categorical_cols:
    encoder = LabelEncoder()
    encoder.fit(train[col] + ['MISSING'])  # +MISSING для NaN
    
    X_train[col] = encoder.transform(X_train[col].fillna('MISSING'))
    encoders[col] = encoder  # Зберігаємо для app.py

# 3. XGBoost з регуляризацією
model = XGBClassifier(
    n_estimators=400,      # Багато дерев
    max_depth=4,           # Мілкі дерева (менше overfitting)
    learning_rate=0.03,    # Повільне навчання
    subsample=0.7,         # 70% рядків на дерево
    colsample_bytree=0.6,  # 60% features на дерево
    min_child_weight=5,    # Мінімум даних у листі
    gamma=0.3,             # Penalty за складність
    reg_alpha=0.3,         # L1 regularization
    reg_lambda=2.0,        # L2 regularization
)

model.fit(X_train, y_train)

# 4. БЕЗ калібрації (Isotonic давав 100% ймовірності)
# Використовуємо чисту XGBoost модель

# 5. Зберігає
pickle.dump(model, 'saved models/xgboost_calibrated_model.pkl')
pickle.dump(encoders, 'saved models/label_encoders.pkl')
```

**Feature Importance (топ-10):**
```
1. p1_rank            - Ранг гравця 1
2. p2_rank            - Ранг гравця 2
3. h2h_p1_win_rate    - % перемог у H2H
4. p1_rank_points     - Рейтингові очки P1
5. p2_rank_points     - Рейтингові очки P2
6. p1_1stWon_roll10   - Форма на 1й подачі P1
7. p2_1stWon_roll10   - Форма на 1й подачі P2
8. p1_age             - Вік гравця 1
9. p2_age             - Вік гравця 2
10. surface           - Покриття корту
```

---

### ЕТАП 5: STREAMLIT ДОДАТОК (app.py)

**Головний файл інтерфейсу користувача**

#### 5.1 Завантаження при старті

```python
@st.cache_resource
def load_model():
    model = pickle.load('saved models/xgboost_calibrated_model.pkl')
    encoders = pickle.load('saved models/label_encoders.pkl')
    feature_cols = open('saved models/feature_columns.txt').readlines()
    return model, encoders, feature_cols

@st.cache_data
def load_database():
    # Завантажує обидва набори даних
    train = pd.read_csv('data/processed/train_features.csv')
    test = pd.read_csv('data/processed/test_features.csv')
    
    # Об'єднує для повної історії
    full_db = pd.concat([train, test])
    
    return full_db, test  # test - для списку актуальних гравців
```

**Що кешується:**
- ✅ Модель (завантажується 1 раз)
- ✅ Encoders (завантажується 1 раз)
- ✅ База даних (завантажується 1 раз)

#### 5.2 Інтерфейс користувача

**TAB 1: Prediction (Прогноз)**

```python
# Випадаючі списки
p1_name = st.selectbox("Гравець 1", players_list)
p2_name = st.selectbox("Гравець 2", players_list)
surface = st.selectbox("Покриття", ["Hard", "Clay", "Grass"])
tourney_level = st.selectbox("Рівень", ["Grand Slam", "Masters", ...])

# Кнопка
if st.button("ЗРОБИТИ ПРОГНОЗ"):
    # Викликає calculate_features()
    # Показує результат
```

**TAB 2: Player History (Історія гравця)**

```python
# Показує останні 10 матчів обраного гравця
player_matches = db[
    (db['p1_name'] == player) | (db['p2_name'] == player)
].tail(10)

# Таблиця з результатами
```

#### 5.3 Функція calculate_features() - СЕРЦЕ СИСТЕМИ

**Покрокова логіка:**

```python
def calculate_features(db, p1_name, p2_name, surface, tourney_level):
    """
    Створює 48 features для прогнозу БЕЗ введення даних вручну!
    """
    
    # ШАГ 1: Знайти ОСТАННІЙ матч кожного гравця
    p1_matches = db[(db['p1_name'] == p1_name) | (db['p2_name'] == p1_name)]
    p1_latest = p1_matches.iloc[-1]  # Останній рядок
    
    p2_matches = db[(db['p1_name'] == p2_name) | (db['p2_name'] == p2_name)]
    p2_latest = p2_matches.iloc[-1]
    
    # ШАГ 2: Визначити позицію гравця (P1 чи P2 в останньому матчі)
    p1_is_p1 = (p1_latest['p1_name'] == p1_name)  # True якщо був P1
    p2_is_p1 = (p2_latest['p1_name'] == p2_name)
    
    # ШАГ 3: Витягнути features з правильної колонки
    features = {}
    
    # Базові дані P1
    features['p1_rank'] = p1_latest['p1_rank'] if p1_is_p1 else p1_latest['p2_rank']
    features['p1_rank_points'] = p1_latest['p1_rank_points'] if p1_is_p1 else p1_latest['p2_rank_points']
    features['p1_age'] = p1_latest['p1_age'] if p1_is_p1 else p1_latest['p2_age']
    # ... і т.д. для всіх P1 features
    
    # Rolling stats P1
    features['p1_ace_roll10'] = p1_latest['p1_ace_roll10'] if p1_is_p1 else p1_latest['p2_ace_roll10']
    # ... і т.д. для всіх rolling stats
    
    # ТЕ САМЕ ДЛЯ P2
    features['p2_rank'] = p2_latest['p1_rank'] if p2_is_p1 else p2_latest['p2_rank']
    # ...
    
    # ШАГ 4: Head-to-Head
    h2h = db[
        ((db['p1_name']==p1_name) & (db['p2_name']==p2_name)) |
        ((db['p1_name']==p2_name) & (db['p2_name']==p1_name))
    ]
    
    features['h2h_p1_wins'] = len(h2h[h2h['p1_name']==p1_name][h2h['p1_won']==1])
    features['h2h_p2_wins'] = len(h2h[h2h['p2_name']==p2_name][h2h['p1_won']==0])
    features['h2h_total_matches'] = len(h2h)
    features['h2h_p1_win_rate'] = h2h_p1_wins / h2h_total if h2h_total > 0 else 0.5
    
    # ШАГ 5: Seed features
    features['is_p1_seeded'] = not pd.isna(features['p1_seed'])
    features['p1_seed_tier'] = get_seed_tier(features['p1_seed'])
    # ...
    
    # ШАГ 6: User input features
    features['surface'] = surface  # "Hard"
    features['tourney_level'] = tourney_level  # "A"
    features['draw_size'] = 128  # За замовчуванням
    features['indoor'] = 0
    
    # ШАГ 7: НОРМАЛІЗАЦІЯ для симетричності
    needs_swap = features['p2_rank'] < features['p1_rank']
    
    if needs_swap:
        # Міняємо місцями ВСІ P1/P2 features
        for key in list(features.keys()):
            if key.startswith('p1_'):
                p2_key = key.replace('p1_', 'p2_')
                if p2_key in features:
                    features[key], features[p2_key] = features[p2_key], features[key]
        
        # Міняємо H2H
        features['h2h_p1_wins'], features['h2h_p2_wins'] = \
            features['h2h_p2_wins'], features['h2h_p1_wins']
    
    return features, needs_swap
```

#### 5.4 Прогнозування

```python
# Отримуємо features
features_dict, swapped = calculate_features(db, p1_name, p2_name, surface, level)

# Створюємо DataFrame
input_df = pd.DataFrame([features_dict])

# Encode категоріальні features
for col in categorical_features:
    if col in label_encoders:
        input_df[col] = input_df[col].fillna('MISSING')
        input_df[col] = label_encoders[col].transform(input_df[col].astype(str))

# Заповнюємо NaN
input_df = input_df.fillna(0)

# Переупорядковуємо колонки
input_df = input_df[feature_cols]

# ПРОГНОЗ
prob_p1_wins = model.predict_proba(input_df)[0, 1]  # Ймовірність що P1 виграє
prob_p2_wins = 1 - prob_p1_wins

# ІНВЕРСІЯ якщо були поміняні місцями
if swapped:
    prob_p1_wins, prob_p2_wins = prob_p2_wins, prob_p1_wins

# ПОКАЗУЄМО
st.write(f"{original_p1_name}: {prob_p1_wins:.1%}")
st.write(f"{original_p2_name}: {prob_p2_wins:.1%}")
```

---

## 🎯 ЯК ПРАЦЮЄ СИМЕТРИЧНА МОДЕЛЬ

### Приклад: Alcaraz (#1) vs Sinner (#2)

**Варіант 1: Користувач вводить "Alcaraz vs Sinner"**

```python
# 1. calculate_features() отримує:
features = {
    'p1_rank': 1,  # Alcaraz
    'p2_rank': 2,  # Sinner
    ...
}

# 2. Перевірка: чи потрібен swap?
needs_swap = (2 < 1)  # False, бо Alcaraz кращий

# 3. Модель передбачає:
prob_p1_wins = 0.55  # 55% що Alcaraz виграє

# 4. Swap не було, тому результат як є:
Alcaraz: 55%
Sinner: 45%
```

**Варіант 2: Користувач вводить "Sinner vs Alcaraz"**

```python
# 1. calculate_features() отримує:
features = {
    'p1_rank': 2,  # Sinner
    'p2_rank': 1,  # Alcaraz
    ...
}

# 2. Перевірка: чи потрібен swap?
needs_swap = (1 < 2)  # True, бо Alcaraz кращий!

# 3. Міняємо features місцями:
features = {
    'p1_rank': 1,  # Alcaraz (поміняли!)
    'p2_rank': 2,  # Sinner (поміняли!)
    ...
}

# 4. Модель передбачає (ТІ САМІ features що у Варіанті 1):
prob_p1_wins = 0.55  # 55% що Alcaraz виграє

# 5. Але swapped=True, тому інвертуємо для ВІДОБРАЖЕННЯ:
prob_p1_wins, prob_p2_wins = prob_p2_wins, prob_p1_wins
# prob_p1_wins = 45%, prob_p2_wins = 55%

# 6. Показуємо (original_p1 = Sinner, original_p2 = Alcaraz):
Sinner: 45%
Alcaraz: 55%  ✅ ТОЙ САМИЙ РЕЗУЛЬТАТ!
```

---

## 📊 ТИПОВІ ЗНАЧЕННЯ FEATURES

### Базові дані гравця

```python
p1_rank: 1-300          # Ранг ATP (1=найкращий)
p1_rank_points: 0-15000 # Рейтингові очки
p1_age: 16-40           # Вік гравця
p1_ht: 165-211          # Зріст у см
p1_hand: 'R'/'L'/'U'    # Права/Ліва/Невідома рука
p1_ioc: 'ESP'/'USA'/... # Країна (3-літерний код)
p1_seed: 1-32 або NaN   # Сід у турнірі
p1_entry: 'WC'/'Q'/...  # Тип входу
```

### Rolling Statistics (середнє за 10 матчів)

```python
p1_ace_roll10: 0-20       # Ейси за матч
p1_df_roll10: 0-10        # Подвійні помилки
p1_svpt_roll10: 40-150    # Подач за матч
p1_1stIn_roll10: 30-100   # 1х подач в корт
p1_1stWon_roll10: 20-80   # Виграно на 1й
p1_2ndWon_roll10: 10-40   # Виграно на 2й
p1_SvGms_roll10: 8-15     # Сервіс геймів
p1_bpSaved_roll10: 0-5    # Врятовано BP
p1_bpFaced_roll10: 0-10   # BP проти
```

### Head-to-Head

```python
h2h_p1_wins: 0-50         # К-ть перемог P1
h2h_p2_wins: 0-50         # К-ть перемог P2
h2h_total_matches: 0-100  # Всього матчів
h2h_p1_win_rate: 0.0-1.0  # % перемог P1
```

### Категоріальні features

```python
surface:
  - 'Hard'    (0) - Хард
  - 'Clay'    (1) - Грунт
  - 'Grass'   (2) - Трава
  - 'Carpet'  (3) - Килим

tourney_level:
  - 'G'  (0) - Grand Slam
  - 'M'  (1) - Masters 1000
  - 'A'  (2) - ATP 500
  - 'D'  (3) - ATP 250
  - 'F'  (4) - ATP Finals
  - ...

p1_seed_tier:
  - 'Top4'   (0) - Топ-4 сіди
  - 'Top8'   (1) - Топ-8
  - 'Top16'  (2) - Топ-16
  - 'Top32'  (3) - Топ-32
  - 'Other'  (4) - Інші
  - 'MISSING' (5) - Немає сіду
```

---

## 🚀 ЯК ЗАПУСТИТИ ПРОЕКТ

### 1. Встановлення

```bash
# Клонувати репозиторій
git clone https://github.com/7t3an/tennis_match_prediction.git
cd tennis_match_prediction

# Створити віртуальне середовище
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# або
.venv\Scripts\activate     # Windows

# Встановити залежності
pip install -r requirements.txt
```

### 2. Запуск Streamlit

```bash
streamlit run app.py
```

Відкриється браузер: `http://localhost:8501`

### 3. Перетренування моделі (опціонально)

```bash
# Якщо хочете перетренувати з іншими параметрами
python retrain_model.py

# Або виправити асиметрію
python fix_asymmetric_features.py
```

---

## 🔧 НАЛАШТУВАННЯ МОДЕЛІ

### Параметри XGBoost (retrain_model.py)

```python
xgb_model = XGBClassifier(
    n_estimators=400,      # ↑ більше = точніше, але повільніше
    max_depth=4,           # ↓ менше = менше overfitting
    learning_rate=0.03,    # ↓ менше = точніше, але повільніше
    subsample=0.7,         # % рядків на кожне дерево
    colsample_bytree=0.6,  # % features на кожне дерево
    min_child_weight=5,    # ↑ більше = менше overfitting
    gamma=0.3,             # ↑ більше = більша регуляризація
    reg_alpha=0.3,         # L1 regularization
    reg_lambda=2.0,        # L2 regularization
)
```

**Як впливають параметри:**
- `n_estimators` ↑ → accuracy ↑, швидкість ↓
- `max_depth` ↓ → overfitting ↓, bias ↑
- `learning_rate` ↓ → точність ↑, треба більше estimators
- `gamma` ↑ → менше листів у дереві, простіша модель

---

## 📈 МЕТРИКИ МОДЕЛІ

### Поточні результати

```
Симетрична модель:
├── Accuracy: 65.3%       # Правильних прогнозів
├── ROC-AUC: 0.6408       # Якість вірогідностей
├── Features: 48          # Кількість ознак
└── Max prob: 97.3%       # Макс впевненість

Розподіл ймовірностей:
├── >90%: 4.3%            # Дуже впевнені прогнози
├── >95%: 0.5%            # Екстремально впевнені
├── 50-80%: 85%           # Більшість прогнозів
└── Mean: 65.3%           # Середня ймовірність
```

### Що означають метрики

- **Accuracy 65.3%** - модель правильно передбачає 65 з 100 матчів
- **ROC-AUC 0.64** - модель краща за випадкове вгадування (0.5), але не ідеальна (1.0)
- **Max 97.3%** - найвпевненіший прогноз модель дає з ймовірністю 97%

### Реалістичність прогнозів

Порівняння з реальною статистикою:

```
Модель прогнозує:     Реальність:
Rank #1 vs #5: 80%  → 84.2%  ✅ Близько!
Top-3 vs Top-10: 71% → 71.6%  ✅ Дуже точно!
```

---

## ❓ FAQ (Часті Питання)

### Q: Чому модель показує різні результати для різних турнірів?

A: Тому що `tourney_level` впливає на прогноз. На Grand Slam гравці грають краще, ніж на ATP 250.

### Q: Чому треба вводити покриття корту?

A: Деякі гравці краще грають на певних покриттях. Наприклад, Nadal на Clay vs Federer на Grass.

### Q: Звідки беруться rolling stats якщо гравець новачок?

A: Якщо менше 10 матчів - використовуються середні значення по турніру або 0.

### Q: Що робити якщо гравця немає в базі?

A: Додати його матчі в `data/tml/` і перезапустити Feature Engineering.

### Q: Чому видалили rank_diff та інші features?

A: Вони були асиметричними - модель давала різні прогнози для "A vs B" і "B vs A". Симетрична модель завжди дає однакові результати незалежно від порядку.

### Q: Як часто треба оновлювати дані?

A: Після кожного великого турніру (Grand Slam, Masters) - завантажити нові CSV з Tennis Match Library.

---

## 🎓 НАВЧАЛЬНІ МАТЕРІАЛИ

### Що вивчити для розуміння проекту

1. **Python Basics**
   - Pandas (DataFrame, read_csv, merge, groupby)
   - NumPy (arrays, statistics)
   - Pickle (serialization)

2. **Machine Learning**
   - Supervised Learning
   - Classification vs Regression
   - Train/Test split
   - Feature Engineering
   - Label Encoding

3. **XGBoost**
   - Gradient Boosting
   - Decision Trees
   - Regularization
   - Hyperparameter Tuning

4. **Streamlit**
   - st.selectbox, st.button
   - st.cache_resource, st.cache_data
   - Tabs and Layout

---

## 📝 ПІДСУМОК

**Система працює так:**

1. **Завантажує** історичні дані матчів (1968-2025)
2. **Обробляє** їх (очищення, трансформація)
3. **Створює features** (rolling stats, H2H, seed features)
4. **Тренує XGBoost** модель (48 features → prob win)
5. **Нормалізує** дані для симетричності
6. **Streamlit** надає UI для прогнозів
7. **calculate_features()** автоматично витягує дані з останніх матчів
8. **Модель** передбачає ймовірність перемоги
9. **Показує** результат користувачу

**Головна фішка:** Користувач вводить ТІЛЬКИ ІМЕНА, всі дані витягуються автоматично з бази! 🚀
