# Tennis Match Prediction System

A machine learning system for predicting ATP tennis match outcomes using XGBoost with temporal validation and probability calibration.

## Key Features

- **XGBoost Classifier** with isotonic probability calibration
- **Temporal Validation** - no data leakage from future to past
- **60 Engineered Features** including rolling statistics, head-to-head records, and derived performance metrics
- **Interactive Web Interface** built with Streamlit
- **Automated Pipeline** for data updates and model retraining

## Model Performance

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.71 |
| Accuracy | 65.2% |
| Log Loss | 0.63 |
| F1 Score | 0.65 |

### Performance by Confidence Level

| Confidence | Matches | Accuracy |
|------------|---------|----------|
| Very High (>70%) | 1,440 | 72.15% |
| High (60-70%) | 487 | 62.22% |
| Medium (55-60%) | 348 | 56.03% |
| Low (50-55%) | 593 | 53.29% |

## Quick Start

### Prerequisites

- Python 3.11+
- pip or conda

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/tennis_match_prediction.git
cd tennis_match_prediction

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
# Start Streamlit web interface
streamlit run app.py
```

Open http://localhost:8501 in your browser.

### Training the Model

```bash
# Full pipeline (data processing + training)
python scripts/run_pipeline.py --full

# Production mode (train on all data)
python scripts/run_pipeline.py --full --production

# Training only (skip data update)
python scripts/run_pipeline.py --train
```

## Project Structure

```
tennis_match_prediction/
├── app.py                      # Streamlit web application
├── requirements.txt            # Python dependencies
├── config/                     # Configuration settings
│   └── settings.py
├── data/
│   ├── processed/              # Processed CSV files
│   └── tml/                    # Raw ATP match data (1968-2025)
├── docs/                       # Documentation
│   └── THESIS.md               # Thesis documentation
├── models/                     # Trained models and encoders
│   ├── xgboost_calibrated_model.pkl
│   ├── label_encoders.pkl
│   └── feature_columns.txt
├── notebooks/                  # Jupyter notebooks for analysis
│   ├── Feature_Engineering.ipynb
│   ├── Model_Training.ipynb
│   └── main_eda.ipynb
├── scripts/                    # Automation scripts
│   ├── run_pipeline.py
│   └── retrain_model.py
├── src/                        # Source code modules
│   ├── data/                   # Data loading and processing
│   ├── features/               # Feature engineering
│   ├── models/                 # Model prediction logic
│   └── pipeline/               # ML pipeline components
└── tests/                      # Unit tests
```

## Feature Engineering

### Rank-Based Features
- `rank_diff` - ATP ranking difference between players
- `rank_ratio` - Ratio of player rankings
- `rank_points_diff` - Difference in ranking points

### Rolling Statistics (10-match window)
- Aces, double faults, serve points
- First serve percentage and win rate
- Break points saved/faced

### Head-to-Head Features
- Total previous encounters
- Win rate for each player
- Recent H2H performance

### Context Features
- Court surface (Hard, Clay, Grass)
- Tournament level (Grand Slam, Masters, ATP 500/250)
- Indoor/outdoor indicator

## Data Sources

Match data sourced from [Jeff Sackmann's Tennis Abstract](https://github.com/JeffSackmann/tennis_atp), covering ATP matches from 1968 to present.

## Technology Stack

- **Python 3.11**
- **XGBoost** - Gradient boosting classifier
- **Scikit-learn** - Model calibration and evaluation
- **Pandas** - Data manipulation
- **Streamlit** - Web interface
- **Plotly** - Interactive visualizations

## License

MIT License

## Author

Vladyslav Romaniuk
