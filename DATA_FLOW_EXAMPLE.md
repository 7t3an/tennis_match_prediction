# 🔄 ЯК ПРАЦЮЮТЬ ДАНІ: ПОКРОКОВИЙ ПРИКЛАД

## 📊 ПРИКЛАД: Прогноз для "Carlos Alcaraz vs Jack Draper"

### ШАГ 1: Користувач вводить

```
┌──────────────────────────────────┐
│  Streamlit Interface (app.py)    │
├──────────────────────────────────┤
│  Гравець 1: Carlos Alcaraz      │
│  Гравець 2: Jack Draper          │
│  Покриття:  Hard                 │
│  Рівень:    ATP 500              │
│  [🎯 ЗРОБИТИ ПРОГНОЗ]            │
└──────────────────────────────────┘
```

---

### ШАГ 2: calculate_features() шукає дані в БД

**База даних: test_features.csv (2025)**

```python
# Пошук останнього матчу Alcaraz
alcaraz_matches = test_df[
    (test_df['p1_name'] == 'Carlos Alcaraz') | 
    (test_df['p2_name'] == 'Carlos Alcaraz')
]
alcaraz_latest = alcaraz_matches.iloc[-1]  # Останній рядок

# Приклад знайденого рядка:
{
    'match_id': 'M2025-001234',
    'tourney_date': '2025-01-15',
    'p1_name': 'Carlos Alcaraz',     # ← Він був P1
    'p2_name': 'Jannik Sinner',
    'p1_rank': 1,                     # ← Витягуємо ранг
    'p1_rank_points': 11050,          # ← Витягуємо очки
    'p1_age': 22.519,
    'p1_hand': 'R',
    'p1_ht': 183,
    'p1_ioc': 'ESP',
    'p1_seed': 1,
    'p1_entry': 'DA',
    'p1_ace_roll10': 5.3,             # ← Rolling stats
    'p1_df_roll10': 1.8,
    'p1_1stIn_roll10': 46.3,
    # ... і т.д.
    'p1_won': 1  # Виграв той матч
}
```

**ТЕ САМЕ для Jack Draper:**

```python
draper_matches = test_df[
    (test_df['p1_name'] == 'Jack Draper') | 
    (test_df['p2_name'] == 'Jack Draper')
]
draper_latest = draper_matches.iloc[-1]

# Приклад:
{
    'match_id': 'M2025-001189',
    'tourney_date': '2025-01-12',
    'p1_name': 'Jack Draper',        # ← Він був P1
    'p2_name': 'Taylor Fritz',
    'p1_rank': 5,                     # ← Витягуємо ранг
    'p1_rank_points': 4440,
    'p1_age': 23.67,
    'p1_hand': 'L',
    'p1_ht': 193,
    'p1_ioc': 'GBR',
    'p1_seed': 5,
    'p1_entry': 'DA',
    'p1_ace_roll10': 11.67,
    'p1_df_roll10': 3.22,
    # ...
    'p1_won': 0  # Програв той матч
}
```

---

### ШАГ 3: Витягує дані з правильних колонок

**Для Alcaraz (був P1 в останньому матчі):**

```python
p1_is_p1 = (alcaraz_latest['p1_name'] == 'Carlos Alcaraz')  # True

# Тому беремо з p1_* колонок:
alcaraz_rank = alcaraz_latest['p1_rank']           # 1
alcaraz_points = alcaraz_latest['p1_rank_points']  # 11050
alcaraz_age = alcaraz_latest['p1_age']             # 22.519
alcaraz_ace = alcaraz_latest['p1_ace_roll10']      # 5.3
# ...
```

**Для Draper (був P1 в останньому матчі):**

```python
p2_is_p1 = (draper_latest['p1_name'] == 'Jack Draper')  # True

# Тому беремо з p1_* колонок:
draper_rank = draper_latest['p1_rank']           # 5
draper_points = draper_latest['p1_rank_points']  # 4440
draper_age = draper_latest['p1_age']             # 23.67
draper_ace = draper_latest['p1_ace_roll10']      # 11.67
# ...
```

**Якби Draper був P2 в останньому матчі:**

```python
# Припустимо останній матч був:
{
    'p1_name': 'Taylor Fritz',
    'p2_name': 'Jack Draper',      # ← Він P2
    'p2_rank': 5,                   # ← Тоді беремо з p2_*
    'p2_rank_points': 4440,
    'p2_ace_roll10': 11.67,
    # ...
}

p2_is_p1 = False  # Бо він був P2

# Беремо з p2_* колонок:
draper_rank = draper_latest['p2_rank']           # 5
draper_points = draper_latest['p2_rank_points']  # 4440
draper_ace = draper_latest['p2_ace_roll10']      # 11.67
```

---

### ШАГ 4: Створює features dict

```python
features = {
    # Tournament info (від користувача)
    'surface': 'Hard',              # Input
    'tourney_level': 'A',           # ATP 500 = 'A'
    'draw_size': 128,               # За замовчуванням
    'indoor': 0,                    # За замовчуванням
    
    # P1 = Alcaraz (з alcaraz_latest)
    'p1_rank': 1,
    'p1_rank_points': 11050,
    'p1_age': 22.519,
    'p1_hand': 'R',
    'p1_ht': 183,
    'p1_ioc': 'ESP',
    'p1_seed': 1,
    'p1_entry': 'DA',
    'p1_ace_roll10': 5.3,
    'p1_df_roll10': 1.8,
    'p1_svpt_roll10': 70.2,
    'p1_1stIn_roll10': 46.3,
    'p1_1stWon_roll10': 34.1,
    'p1_2ndWon_roll10': 12.8,
    'p1_SvGms_roll10': 11.1,
    'p1_bpSaved_roll10': 2.8,
    'p1_bpFaced_roll10': 3.9,
    
    # P2 = Draper (з draper_latest)
    'p2_rank': 5,
    'p2_rank_points': 4440,
    'p2_age': 23.67,
    'p2_hand': 'L',
    'p2_ht': 193,
    'p2_ioc': 'GBR',
    'p2_seed': 5,
    'p2_entry': 'DA',
    'p2_ace_roll10': 11.67,
    'p2_df_roll10': 3.22,
    'p2_svpt_roll10': 82.56,
    'p2_1stIn_roll10': 53.67,
    'p2_1stWon_roll10': 42.67,
    'p2_2ndWon_roll10': 15.11,
    'p2_SvGms_roll10': 14.44,
    'p2_bpSaved_roll10': 3.0,
    'p2_bpFaced_roll10': 4.56,
    
    # Seed features (обчислені)
    'is_p1_seeded': True,           # seed = 1
    'is_p2_seeded': True,           # seed = 5
    'p1_seed_tier': 'Top4',         # seed <= 4
    'p2_seed_tier': 'Top8',         # seed <= 8
    
    # Encoded features (дублікати)
    'tourney_level_encoded': 'A',
    'surface_encoded': 'Hard',
}
```

**ВАЖЛИВО:** Всього 48 features (було 53, видалили 5 асиметричних)

---

### ШАГ 5: H2H Calculation

```python
# Шукаємо всі попередні матчі між Alcaraz і Draper
h2h_matches = full_db[
    ((full_db['p1_name'] == 'Carlos Alcaraz') & (full_db['p2_name'] == 'Jack Draper')) |
    ((full_db['p1_name'] == 'Jack Draper') & (full_db['p2_name'] == 'Carlos Alcaraz'))
]

# Приклад знайдених матчів:
┌────────────┬──────────────────┬──────────────┬─────────┐
│ tourney    │ p1_name          │ p2_name      │ p1_won  │
├────────────┼──────────────────┼──────────────┼─────────┤
│ 2021-Wim   │ Carlos Alcaraz   │ Jack Draper  │ 1       │ ← Alcaraz виграв
│ 2022-USO   │ Jack Draper      │ Carlos Alcz  │ 0       │ ← Alcaraz виграв (був P2)
│ 2023-AO    │ Carlos Alcaraz   │ Jack Draper  │ 1       │ ← Alcaraz виграв
│ 2023-FO    │ Jack Draper      │ Carlos Alcz  │ 0       │ ← Alcaraz виграв (був P2)
│ 2024-Wim   │ Carlos Alcaraz   │ Jack Draper  │ 1       │ ← Alcaraz виграв
│ ...        │                  │              │         │
└────────────┴──────────────────┴──────────────┴─────────┘

# Рахуємо:
h2h_p1_wins = 0
h2h_p2_wins = 0

for match in h2h_matches:
    if match['p1_name'] == 'Carlos Alcaraz':
        # Alcaraz був P1
        if match['p1_won'] == 1:
            h2h_p1_wins += 1  # Alcaraz виграв
        else:
            h2h_p2_wins += 1  # Draper виграв
    else:
        # Alcaraz був P2
        if match['p1_won'] == 0:
            h2h_p1_wins += 1  # Alcaraz виграв (P2 виграв)
        else:
            h2h_p2_wins += 1  # Draper виграв (P1 виграв)

# Результат:
h2h_p1_wins = 8          # Alcaraz виграв 8 разів
h2h_p2_wins = 4          # Draper виграв 4 рази
h2h_total_matches = 12
h2h_p1_win_rate = 8/12 = 0.6667  # 66.7%

# Додаємо до features:
features['h2h_p1_wins'] = 8
features['h2h_p2_wins'] = 4
features['h2h_total_matches'] = 12
features['h2h_p1_win_rate'] = 0.6667
```

---

### ШАГ 6: Нормалізація (для симетрії)

```python
# Перевірка: чи P2 має кращий rank?
needs_swap = features['p2_rank'] < features['p1_rank']
# needs_swap = 5 < 1 = False  ← Alcaraz кращий, swap не треба

# Якби було навпаки (користувач ввів "Draper vs Alcaraz"):
features_original = {
    'p1_rank': 5,    # Draper
    'p2_rank': 1,    # Alcaraz
    ...
}

needs_swap = 1 < 5 = True  ← Треба поміняти!

# Міняємо ВСІ P1/P2 features:
for key in features.keys():
    if key.startswith('p1_'):
        p2_key = key.replace('p1_', 'p2_')
        features[key], features[p2_key] = features[p2_key], features[key]

# Після swap:
features = {
    'p1_rank': 1,    # Alcaraz (поміняли!)
    'p2_rank': 5,    # Draper (поміняли!)
    'p1_hand': 'R',  # Alcaraz
    'p2_hand': 'L',  # Draper
    ...
}

# Міняємо H2H також:
features['h2h_p1_wins'], features['h2h_p2_wins'] = \
    features['h2h_p2_wins'], features['h2h_p1_wins']

# Повертаємо:
return features, needs_swap  # features, True
```

---

### ШАГ 7: Створення DataFrame

```python
input_df = pd.DataFrame([features])

# Виглядає так:
┌─────────┬────────────┬──────────┬───────┬──────────┬────────────┬─────┐
│ surface │ draw_size  │ indoor   │ p1_r  │ p1_points│ p1_ace_r10 │ ... │
├─────────┼────────────┼──────────┼───────┼──────────┼────────────┼─────┤
│ Hard    │ 128        │ 0        │ 1     │ 11050    │ 5.3        │ ... │
└─────────┴────────────┴──────────┴───────┴──────────┴────────────┴─────┘
```

---

### ШАГ 8: Encoding категоріальних features

```python
# Завантажуємо encoders
encoders = pickle.load('label_encoders.pkl')

# Encode кожну категоріальну колонку:
categorical = ['surface', 'tourney_level', 'p1_entry', 'p1_hand', 
               'p1_ioc', 'p2_entry', 'p2_hand', 'p2_ioc',
               'p1_seed_tier', 'p2_seed_tier']

for col in categorical:
    # Приклад: surface
    input_df[col] = input_df[col].fillna('MISSING')  # 'Hard' → 'Hard'
    input_df[col] = encoders[col].transform(input_df[col])
    
    # encoders['surface'].classes_ = ['Carpet', 'Clay', 'Grass', 'Hard', 'MISSING']
    # 'Hard' → 3
    
# Після encoding:
┌─────────┬────────────┬──────────┬───────┬──────────┬────────────┐
│ surface │ tourney_lv │ p1_hand  │ p1_ioc│ p2_hand  │ p2_ioc     │
├─────────┼────────────┼──────────┼───────┼──────────┼────────────┤
│ 3       │ 2          │ 3        │ 26    │ 1        │ 30         │
│ (Hard)  │ (A)        │ (R)      │ (ESP) │ (L)      │ (GBR)      │
└─────────┴────────────┴──────────┴───────┴──────────┴────────────┘
```

---

### ШАГ 9: Заповнення NaN та впорядкування

```python
# Заповнюємо пропуски нулями
input_df = input_df.fillna(0)

# Переупорядковуємо колонки згідно з feature_columns.txt
feature_cols = open('feature_columns.txt').read().splitlines()
# ['surface', 'draw_size', 'tourney_level', 'indoor', 'p1_seed', ...]

input_df = input_df[feature_cols]  # Правильний порядок!

# Фінальний вигляд (48 колонок):
┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬─────┬─────┬─────┬─────┐
│ 3 │128│ 2 │ 0 │ 1 │ 1 │ 3 │183│26 │22 │11050│ 5.3 │ 1.8 │ ... │
└───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴─────┴─────┴─────┴─────┘
  ↑   ↑   ↑   ↑   ↑   ↑   ↑   ↑   ↑   ↑    ↑     ↑     ↑
  │   │   │   │   │   │   │   │   │   │    │     │     └─ p1_df_roll10
  │   │   │   │   │   │   │   │   │   │    │     └─ p1_ace_roll10
  │   │   │   │   │   │   │   │   │   │    └─ p1_rank_points
  │   │   │   │   │   │   │   │   │   └─ p1_age
  │   │   │   │   │   │   │   │   └─ p1_ioc
  │   │   │   │   │   │   │   └─ p1_ht
  │   │   │   │   │   │   └─ p1_hand
  │   │   │   │   │   └─ p1_entry
  │   │   │   │   └─ p1_seed
  │   │   │   └─ indoor
  │   │   └─ tourney_level
  │   └─ draw_size
  └─ surface
```

---

### ШАГ 10: Prediction (XGBoost)

```python
# Завантажуємо модель
model = pickle.load('xgboost_calibrated_model.pkl')

# Прогноз
probabilities = model.predict_proba(input_df)
# probabilities = [[0.18, 0.82]]
#                   ↑      ↑
#                   P(0)  P(1)
#                   P2    P1
#                   втр   виг

prob_p1_wins = probabilities[0, 1]  # 0.82 = 82%
prob_p2_wins = 1 - prob_p1_wins     # 0.18 = 18%

print(f"P1 (Alcaraz): {prob_p1_wins:.1%}")  # 82.0%
print(f"P2 (Draper): {prob_p2_wins:.1%}")   # 18.0%
```

**ЩО РОБИТЬ МОДЕЛЬ ВСЕРЕДИНІ:**

```python
# XGBoost проходить через 400 decision trees
tree_1_vote = 0.6    # Tree 1: 60% що P1 виграє
tree_2_vote = 0.8    # Tree 2: 80%
tree_3_vote = 0.7    # Tree 3: 70%
...
tree_400_vote = 0.85 # Tree 400: 85%

# Комбінує всі голоси:
final_prob = weighted_average([0.6, 0.8, 0.7, ..., 0.85])
# final_prob = 0.82
```

**Приклад одного дерева:**

```
            [p1_rank <= 2?]
               /        \
            YES          NO
             /            \
    [h2h_win_rate>0.6?] [p2_rank>50?]
        /      \           /      \
      YES      NO        YES      NO
       |        |         |        |
     0.9      0.7       0.3      0.5
     
# Для Alcaraz:
# - p1_rank=1 <= 2? YES → йде ліворуч
# - h2h_win_rate=0.67 > 0.6? YES → йде ліворуч
# - Результат: 0.9 (90% ймовірність)

# Це тільки 1 з 400 дерев!
```

---

### ШАГ 11: Інверсія якщо був swap

```python
# Якщо користувач ввів "Draper vs Alcaraz":
if needs_swap:  # True, бо Alcaraz кращий
    # Модель передбачила для нормалізованого порядку:
    # P1=Alcaraz: 82%, P2=Draper: 18%
    
    # Але користувач хоче бачити:
    # P1=Draper: ???, P2=Alcaraz: ???
    
    # Тому міняємо місцями:
    prob_p1_wins, prob_p2_wins = prob_p2_wins, prob_p1_wins
    # prob_p1_wins = 18%  (Draper)
    # prob_p2_wins = 82%  (Alcaraz)
```

---

### ШАГ 12: Відображення результату

```python
# В Streamlit:
st.markdown(f"""
<div style="text-align: center;">
    <h2>👤 {original_p1_name}</h2>
    <h1>{prob_p1_wins:.1%}</h1>
    <p>Шанс на перемогу</p>
</div>
""")

# Gauge chart
fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=prob_p1_wins * 100,  # 82.0
    title={'text': f"{original_p1_name} Win Probability"},
    gauge={
        'axis': {'range': [0, 100]},
        'bar': {'color': "#667eea"},
        'threshold': {'value': 50}
    }
))

st.plotly_chart(fig)
```

**Фінальний вивід:**

```
┌─────────────────────────────────────┐
│   🎾 Прогноз моделі                 │
├─────────────────────────────────────┤
│                                     │
│   👤 Carlos Alcaraz                 │
│         82.0%                       │
│   Шанс на перемогу                  │
│                                     │
│   Ранг: 1 | Очки: 11050             │
│                                     │
├─────────────────────────────────────┤
│                                     │
│   👤 Jack Draper                    │
│         18.0%                       │
│   Шанс на перемогу                  │
│                                     │
│   Ранг: 5 | Очки: 4440              │
│                                     │
└─────────────────────────────────────┘

📊 Інша інформація:
• H2H: Alcaraz виграв 8 з 12 (67%)
• Фаворит: Carlos Alcaraz (дуже впевнений)
• Rank різниця: 4 позиції
```

---

## 🔄 ПОВНИЙ DATA FLOW

```
USER INPUT:
  Alcaraz, Draper, Hard, ATP 500
         │
         ▼
┌─────────────────────────────┐
│  calculate_features()       │
├─────────────────────────────┤
│  1. Знайти останній матч    │
│     Alcaraz → M2025-001234  │
│     Draper  → M2025-001189  │
│                             │
│  2. Витягти дані:           │
│     - Alcaraz був P1        │
│       → беремо p1_*         │
│     - Draper був P1         │
│       → беремо p1_*         │
│                             │
│  3. H2H calculation:        │
│     - Знайти всі матчі      │
│     - Порахувати перемоги   │
│     - Alcaraz: 8/12         │
│                             │
│  4. Seed features:          │
│     - p1_seed_tier: Top4    │
│     - p2_seed_tier: Top8    │
│                             │
│  5. Normalize:              │
│     - p2_rank < p1_rank?    │
│     - 5 < 1? NO             │
│     - needs_swap = False    │
└─────────────────────────────┘
         │
         ▼
    features dict (48 keys)
         │
         ▼
┌─────────────────────────────┐
│  DataFrame creation         │
│  pd.DataFrame([features])   │
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Label Encoding             │
├─────────────────────────────┤
│  'Hard' → 3                 │
│  'A' → 2                    │
│  'R' → 3                    │
│  'ESP' → 26                 │
│  'L' → 1                    │
│  'GBR' → 30                 │
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Fill NaN & Reorder         │
│  input_df = df[feat_cols]   │
└─────────────────────────────┘
         │
         ▼
    NumPy array (1×48)
    [3, 128, 2, 0, 1, 1, 3, ...]
         │
         ▼
┌─────────────────────────────┐
│  XGBoost Model              │
├─────────────────────────────┤
│  400 Decision Trees         │
│  Tree 1: 0.85               │
│  Tree 2: 0.79               │
│  Tree 3: 0.88               │
│  ...                        │
│  Tree 400: 0.81             │
│                             │
│  Weighted Average:          │
│  → 0.82                     │
└─────────────────────────────┘
         │
         ▼
    prob_p1_wins = 0.82
    prob_p2_wins = 0.18
         │
         ▼
┌─────────────────────────────┐
│  Invert if swapped          │
│  needs_swap? NO             │
│  → No change                │
└─────────────────────────────┘
         │
         ▼
    DISPLAY:
    Alcaraz: 82%
    Draper:  18%
```

---

## 🎯 КЛЮЧОВІ МОМЕНТИ

### 1. Автоматичне витягування даних
**НЕ потрібно вводити вручну:**
- ✅ Ранги
- ✅ Очки
- ✅ Вік
- ✅ Rolling stats
- ✅ H2H

**Все береться з ОСТАННЬОГО матчу гравця!**

### 2. Симетрична модель
**Гарантує однакові результати:**
```
Alcaraz vs Draper = 82% vs 18%
Draper vs Alcaraz = 18% vs 82%
```

### 3. Rolling Statistics = Форма гравця
**Середнє за останні 10 матчів:**
- Якщо гравець у доброй формі → високі stats
- Якщо у поганій → низькі stats

### 4. H2H = Історія зустрічей
**Психологічний фактор:**
- Nadal vs Djokovic на Clay → Nadal often wins
- Навіть якщо rank нижчий

### 5. 48 Features → 1 Number
**Модель комбінує ВСЕ:**
- Rank (найважливіше)
- Points
- Form (rolling stats)
- H2H history
- Court surface preference
- Age/Experience
- Seed advantage

**Результат:** Одна ймовірність 0-100%

---

## 📚 РЕЗЮМЕ

**Система працює просто:**
1. Користувач → Імена + Покриття
2. База даних → Останні матчі
3. Features → 48 чисел
4. Encoding → Категорії → Числа
5. XGBoost → 400 дерев → Ймовірність
6. Display → % для кожного гравця

**Магія в тому що:**
- НЕ треба вводити статистику
- НЕ треба знати ранги
- НЕ треба шукати H2H
- ВСЕ автоматично з бази!

**Користувач тільки:**
```python
select("Alcaraz")
select("Draper")
click("ПРОГНОЗ")
```

**Система робить:**
```python
find_last_match("Alcaraz") → 20+ features
find_last_match("Draper")  → 20+ features
calculate_h2h()            → 4 features
add_user_input()           → 4 features
────────────────────────────────────────
TOTAL: 48 features → XGBoost → 82% vs 18%
```

🎾 **ГОТОВО!**
