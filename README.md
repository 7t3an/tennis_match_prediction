# Tennis Match Prediction

Machine learning system for predicting ATP tennis match outcomes using XGBoost.

## Model Performance

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.7039 |
| Accuracy | 65.39% |
| Precision | 65.25% |
| Recall | 64.61% |
| F1-Score | 64.93% |

## Features

- **No Data Leakage**: All features available before match starts
- **Temporal Validation**: Train on past, test on future
- **61 Engineered Features**: Rolling stats, H2H, rankings
- **Calibrated Probabilities**: Isotonic calibration

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train model
python scripts/train_model.py

# Run app
streamlit run app.py
```

## Project Structure

```
tennis_match_prediction/
├── app.py                  # Streamlit app
├── tml-data/              # Raw data (2012-2026)
├── data/processed/        # Processed features
├── models/                # Saved models
├── scripts/train_model.py # Training script
└── src/                   # Source code
    ├── data/              # Data loading
    ├── features/          # Feature calculator
    ├── models/            # Predictor
    └── pipeline/          # ML pipeline
```

## Data Split

- **Train**: 2012-2024 (36,624 matches)
- **Test**: 2025-2026 (3,077 matches)
- **Final**: All data for production model

## Author

Vladyslav Romaniuk
