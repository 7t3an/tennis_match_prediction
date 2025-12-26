# Tennis Match Prediction - ML Pipeline Documentation

## Overview

Продакшн ML пайплайн для прогнозування результатів тенісних матчів ATP Tour.

### Метрики моделі (Test 2025)

| Метрика | Значення | Ціль | Статус |
|---------|----------|------|--------|
| **ROC-AUC** | 0.7087 | ~0.70 | ✅ Досягнуто |
| **Recall** | 0.6513 | 0.55-0.58 | ✅ Перевищено |
| **Accuracy** | 0.6517 | - | ✅ Стабільно |
| **F1-Score** | 0.6507 | - | ✅ |
| **Log Loss** | 0.6898 | 0.55-0.58 | ⚠️ Вище цілі |

### Performance by Confidence Level

| Рівень впевненості | Матчі | Accuracy |
|--------------------|-------|----------|
| Very Confident (>70%) | 1,440 | **72.15%** |
| Confident (60-70%) | 487 | 62.22% |
| Moderate (55-60%) | 348 | 56.03% |
| Uncertain (50-55%) | 593 | 53.29% |

## Архітектура

```
src/pipeline/
├── __init__.py              # Exports
├── data_processor.py        # DataProcessor: завантаження та очистка даних
├── feature_engineer.py      # FeatureEngineer: створення фіч без витоку даних
├── model_trainer.py         # ModelTrainer: тренування XGBoost + калібрація
└── pipeline.py              # TennisPredictionPipeline: оркестрація
```

## Фічі (NO DATA LEAKAGE!)

### 1. Rank-based Features
- `rank_diff` - різниця рейтингів (P1 - P2)
- `rank_ratio` - відношення рейтингів
- `rank_points_diff` - різниця рейтингових очок
- `is_p1_favorite` - чи P1 фаворит за рейтингом

### 2. Rolling Statistics (10 останніх матчів)
**КРИТИЧНО**: використовуємо `shift(1)` щоб не включати поточний матч!

- `p1_ace_roll10`, `p2_ace_roll10` - середні ейси
- `p1_df_roll10`, `p2_df_roll10` - середні подвійні помилки
- `p1_1stWon_roll10`, `p2_1stWon_roll10` - відсоток виграних перших подач
- ... (всього 18 rolling features)

### 3. Head-to-Head Features
- `h2h_p1_wins`, `h2h_p2_wins` - перемоги у попередніх зустрічах
- `h2h_total_matches` - загальна кількість зустрічей
- `h2h_p1_win_rate`, `h2h_p2_win_rate` - відсоток перемог

### 4. Context Features
- `surface` - покриття (Hard, Clay, Grass)
- `tourney_level` - рівень турніру (G, M, A, D, F)
- `indoor` - indoor/outdoor

### 5. Player Metadata
- `p1_ht`, `p2_ht` - зріст
- `p1_age`, `p2_age` - вік
- `p1_hand`, `p2_hand` - робоча рука (R, L, U)
- `p1_ioc`, `p2_ioc` - країна

## ⚠️ Seed Features - ВИМКНЕНО!

Seed features (`is_seeded`, `seed_diff`, `seed_tier`) **не використовуються** бо:
1. Вони сильно корелюють з результатом (сіяний гравець часто виграє)
2. Це призводить до штучно завищеного AUC (~0.92 vs 0.71)
3. Для real-world prediction seed може бути невідомий

## Використання

### Повний пайплайн
```bash
cd tennis_match_prediction
source .venv/bin/activate
python scripts/run_pipeline.py --full
```

### Тільки тренування (без оновлення даних)
```bash
python scripts/run_pipeline.py --train --start-year 2012 --test-year 2025
```

### Тільки тестування
```bash
python scripts/run_pipeline.py --test
```

### Streamlit App
```bash
streamlit run app.py
```

## Автоматизація

`scripts/run_pipeline.py` автоматично:
1. Витягує нові дані з git repository (`data/tml/`)
2. Обробляє дані (2012-2025)
3. Створює фічі без витоку даних
4. Тренує XGBoost з early stopping
5. Калібрує ймовірності (Isotonic)
6. Зберігає модель до `models/`

## Валідація - NO DATA LEAKAGE

### Temporal Split
- **Train**: 2012-2024 (36,624 матчі)
- **Test**: 2025 (2,926 матчів)

### Cross-Validation
- TimeSeriesSplit з 5 фолдами
- Кожен фолд - хронологічний період

### Перевірки
1. Rolling stats використовують `shift(1)` - тільки минулі матчі
2. H2H рахується тільки з попередніх зустрічей
3. Player IDs **НЕ використовуються** як фічі (модель б запам'ятала сильних гравців)
4. Результати матчу (score, minutes) видалені

## Збережені файли

```
models/
├── xgboost_calibrated_model.pkl   # Модель
├── feature_columns.txt             # Список фіч
├── label_encoders.pkl              # Encoders для категоріальних фіч
└── model_metrics.txt               # Метрики

data/processed/
├── train_features.csv              # Тренувальні дані з фічами
└── test_features.csv               # Тестові дані з фічами
```

## Найважливіші фічі

| Feature | Importance |
|---------|------------|
| rank_ratio | 0.2131 |
| rank_points_diff | 0.0815 |
| rank_diff | 0.0707 |
| is_p1_favorite | 0.0285 |
| p2_rank_points | 0.0215 |
| h2h_p2_win_rate | 0.0193 |
| p1_rank_points | 0.0190 |
| p1_ht | 0.0186 |
| h2h_p1_win_rate | 0.0172 |
| p1_bpFaced_roll10 | 0.0163 |

## Порівняння з різними конфігураціями

| Конфігурація | AUC | Accuracy | Примітки |
|--------------|-----|----------|----------|
| З seed features | 0.92 | 81.5% | ⚠️ Занадто високо - можливий витік |
| Без seed features | 0.71 | 65.2% | ✅ Реалістично |
| Data duplication | 1.00 | 100% | ❌ Критичний витік |

## Рекомендації

1. **Не вмикайте seed features** - вони дають нереалістичні метрики
2. **Регулярно оновлюйте дані** - запускайте `--full` для отримання нових матчів
3. **Ретреніруйте щомісяця** - гравці змінюють форму
4. **Використовуйте "Very Confident" прогнози** - 72% accuracy на >70% confidence
