# 🎾 Tennis Match Prediction

Система прогнозування результатів тенісних матчів на основі Machine Learning.

## 📊 Особливості

- **Симетрична модель** - однакові прогнози незалежно від порядку гравців
- **XGBoost** з 65.3% accuracy
- **48 features** - rank, форма, H2H, статистика
- **58 років даних** (1968-2025)
- **Красивий UI** на Streamlit

## 🚀 Швидкий старт

### 1. Клонувати репозиторій
```bash
git clone https://github.com/7t3an/tennis_match_prediction.git
cd tennis_match_prediction
```

### 2. Створити віртуальне середовище
```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# або
.venv\Scripts\activate     # Windows
```

### 3. Встановити залежності
```bash
pip install -r requirements.txt
```

### 4. Запустити програму
```bash
streamlit run app.py
```

Відкрийте браузер: http://localhost:8501

## 📁 Структура проекту

```
tennis_match_prediction/
├── app.py                    # Main Streamlit app
├── models/                   # ML моделі
├── data/                     # Дані (train/test CSV)
├── scripts/                  # Утиліти (retrain, fix)
├── notebooks/                # Jupyter notebooks
├── config/                   # Конфігурація
├── src/                      # Вихідний код (модулі)
├── tests/                    # Unit tests
├── docs/                     # Документація
└── assets/                   # Зображення
```

Детально: [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)

## 🎯 Як користуватися

1. Вибрати двох гравців
2. Вибрати покриття (Hard/Clay/Grass)
3. Вибрати рівень турніру
4. Натиснути "🎯 ЗРОБИТИ ПРОГНОЗ"

Система автоматично витягне статистику та покаже ймовірності перемоги.

## 🔧 Для розробників

### Перетренування моделі
```bash
cd scripts
python retrain_model.py
```

### Структура коду
- `app.py` - Streamlit UI + логіка прогнозування
- `models/` - XGBoost модель + encoders
- `data/processed/` - Train/Test CSV
- `scripts/` - Утиліти для тренування

## 📈 Метрики

- **Accuracy:** 65.3%
- **ROC-AUC:** 0.64
- **Features:** 48
- **Dataset:** 73K train + 5.8K test

## 🎓 Технології

- Python 3.11
- XGBoost
- Streamlit
- Pandas, NumPy
- Scikit-learn
- Plotly

## 📝 Ліцензія

MIT License

## 👨‍💻 Автор

GitHub: [@7t3an](https://github.com/7t3an)