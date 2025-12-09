# 🏗️ Архітектура проекту Tennis Match Prediction

## 📁 Структура проекту

```
tennis_match_prediction/
├── 📱 app.py                          # Main Streamlit application
├── 📄 README.md                       # Загальний опис проекту
├── 📋 requirements.txt                # Python dependencies
├── 🐍 .venv/                          # Virtual environment
│
├── 📂 config/                         # Конфігурація
│   ├── __init__.py
│   └── settings.py                    # Налаштування проекту
│
├── 📂 models/                         # ML моделі (раніше: saved models/)
│   ├── xgboost_calibrated_model.pkl   # Основна модель
│   ├── label_encoders.pkl             # Encoders для категорій
│   └── feature_columns.txt            # Список features (48 штук)
│
├── 📂 data/                           # Дані
│   ├── processed/                     # Оброблені CSV
│   │   ├── train_features.csv         # Train set (2012-2024)
│   │   └── test_features.csv          # Test set (2025)
│   └── tml/                           # Історичні сирі дані
│       ├── 1968.csv ... 2025.csv
│       └── ATP_Database.csv
│
├── 📂 notebooks/                      # Jupyter notebooks
│   ├── Feature_Engineering.ipynb      # Створення features
│   ├── Model_Training.ipynb           # Тренування моделі
│   ├── Prepare_ETL.ipynb              # ETL pipeline
│   └── main_eda.ipynb                 # Exploratory Data Analysis
│
├── 📂 scripts/                        # Утиліти та скрипти
│   ├── fix_asymmetric_features.py     # Створення симетричної моделі
│   └── retrain_model.py               # Перетренування моделі
│
├── 📂 src/                            # Вихідний код (модулі)
│   ├── __init__.py
│   ├── data/                          # Data processing
│   │   └── __init__.py
│   ├── features/                      # Feature engineering
│   │   └── __init__.py
│   ├── models/                        # ML models
│   │   └── __init__.py
│   └── utils/                         # Допоміжні функції
│       └── __init__.py
│
├── 📂 tests/                          # Unit tests
│   └── __init__.py
│
├── 📂 docs/                           # Документація
│   └── PROJECT_STRUCTURE.md           # Цей файл
│
└── 📂 assets/                         # Ресурси (зображення, іконки)
    └── photo.png
```

---

## 🔄 Як працює система

### 1. **Запуск програми**
```bash
streamlit run app.py
```

### 2. **app.py завантажує:**
- `models/xgboost_calibrated_model.pkl` - XGBoost модель
- `models/label_encoders.pkl` - Encoders для категорій
- `models/feature_columns.txt` - Список 48 features
- `data/processed/test_features.csv` - Актуальні дані гравців (2025)
- `data/processed/train_features.csv` - Історія для H2H

### 3. **Користувач вибирає:**
- Гравець 1
- Гравець 2
- Покриття (Hard/Clay/Grass)
- Рівень турніру (Grand Slam/Masters/ATP 500/250)

### 4. **Система:**
1. Витягує статистику з CSV
2. Створює 48 features
3. Нормалізує (P1 = кращий гравець)
4. Прогнозує ймовірність
5. Показує результат

---

## 🛠️ Для розробників

### **Структура коду в app.py:**

```python
# Завантаження (кешується)
load_model()        # Модель + encoders + features
load_database()     # Train + Test CSV

# Feature engineering
calculate_features()  # Створює 48 features з бази даних
                     # Виконує normalization (P1 = кращий)

# Prediction
model.predict_proba() # XGBoost прогноз
                      # Інверсія якщо був swap
```

### **Скрипти:**

**retrain_model.py:**
- Перетренування моделі
- Запуск: `cd scripts && python retrain_model.py`
- Зберігає в `../models/`

**fix_asymmetric_features.py:**
- Створення симетричної моделі
- Одноразовий скрипт (вже виконано)

---

## 📊 Дані

### **Train set:** `data/processed/train_features.csv`
- 73,248 матчів (2012-2024)
- 53 колонки
- Використовується для тренування та H2H

### **Test set:** `data/processed/test_features.csv`
- 5,822 матчі (2025)
- 53 колонки
- Використовується для актуальної статистики гравців

### **Features (48 штук):**
1. Tournament info (4): surface, level, draw_size, indoor
2. P1 features (20): rank, points, age, hand, seed, rolling stats...
3. P2 features (20): те саме
4. H2H (4): wins, total, win_rate

---

## 🎯 Важливі файли

| Файл | Призначення | Критичність |
|------|-------------|-------------|
| `app.py` | Головна програма | ⭐⭐⭐⭐⭐ |
| `models/*.pkl` | ML моделі | ⭐⭐⭐⭐⭐ |
| `data/processed/*.csv` | База даних | ⭐⭐⭐⭐⭐ |
| `scripts/retrain_model.py` | Перетренування | ⭐⭐⭐⭐ |
| `notebooks/` | Розробка | ⭐⭐⭐ |
| `config/settings.py` | Налаштування | ⭐⭐ |

---

## 🚀 Майбутній розвиток

### **Планові покращення:**
1. Розбити `app.py` на модулі в `src/`
2. Додати unit tests в `tests/`
3. FastAPI для REST API
4. Docker deployment
5. CI/CD з GitHub Actions

### **Поточний статус:**
✅ Симетрична модель (65.3% accuracy)
✅ Streamlit UI
✅ 58 років даних
✅ Професійна структура папок
⏳ Модульна архітектура (в процесі)
⏳ Tests (потрібно додати)
