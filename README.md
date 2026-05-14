# Tennis Match Prediction

Machine learning system that predicts ATP tennis match outcomes using XGBoost.  
Trained on 14 years of official ATP data (2012–2026), calibrated with isotonic regression and temperature scaling.

**Live demo →** https://tennismatchprediction-qyr6ukbtkw5to7n4kzyjta.streamlit.app/

---

## Model Performance

Evaluated on 3,077 unseen matches (2025–2026 test set):

| Metric | Value |
|---|---|
| ROC-AUC | 0.7039 |
| Accuracy | 65.39% |
| F1-Score | 64.93% |
| Log Loss | 0.622 |
| Brier Score | 0.218 |

Accuracy by confidence bracket:

| Confidence | Accuracy | Matches |
|---|---|---|
| > 70% | 73.68% | high-confidence calls |
| 60–70% | 62.51% | |
| 55–60% | 58.55% | |
| < 55% | 52.23% | near-50/50 |

---

## Benchmark - Australian Open 2026

Compared against [Predix Sport](https://www.predixsport.com/tennis_predictions) (commercial platform, 500+ features) across R32–Final:

| Metric | This model | Predix Sport |
|---|---|---|
| Accuracy | **81.3%** | 78.7% |
| Log Loss | **−28%** vs Predix | baseline |
| Brier Score | **−30%** vs Predix | baseline |
| Largest accuracy gap | R16: **75%** vs 62.5% | |
| Features used | **60** | 500+ |

> Better calibration (Temperature Scaling T=1.262, clipping to [5%–95%]) — not more data — drives the Log Loss and Brier Score advantage.

---

## Features (60 total)

| Group | Examples |
|---|---|
| Ranking | `rank_diff`, `log_rank_ratio`, `rank_points_diff` |
| Rolling stats (last 10 matches) | `ace_roll10`, `df_roll10`, `1stWon_roll10`, `bpSaved_roll10` |
| Serve form (derived) | `serve_efficiency_diff`, `bp_save_rate_diff` |
| Head-to-head | `h2h_p1_wins`, `h2h_p1_win_rate` |
| Context | `surface_encoded`, `tourney_level_encoded` |
| Physical | `age_diff`, `height_diff`, `hand` |

**No data leakage:** rolling stats use `shift(1).rolling()` - current match is excluded. Enforced by an automated leakage validator that aborts the pipeline on violation.

---

## Data

Source: [JeffSackmann/tennis_atp](https://github.com/JeffSackmann/tennis_atp) - the open-source standard for historical ATP match data.

| Split | Period | Matches |
|---|---|---|
| Train | 2012–2024 | 36,624 |
| Test | 2025–2026 | 3,077 |

Temporal split only — no random shuffle, no k-fold across time.

---

## Run Locally

```bash
git clone https://github.com/7t3an/tennis_match_prediction.git
cd tennis_match_prediction
pip install -r requirements.txt
streamlit run app.py
```

---

## Deploy (Streamlit Community Cloud)

1. Fork this repo (or push your own copy)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select repo `tennis_match_prediction`, branch `main`, file `app.py`
4. Click **Deploy** - free, public URL in ~2 minutes

---

## Docker

```bash
docker compose up --build
# open http://localhost:8501
```

---

## Project Structure

```
tennis_match_prediction/
├── app.py                          # Streamlit app (Prediction + Player History)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
│
├── tml-data/                       # Raw ATP CSVs, 2012–2026
├── data/processed/                 # Engineered feature CSVs (train / test / full)
├── models/                         # Trained model, encoders, feature list
├── scripts/train_model.py          # Full training entry point
├── notebooks/                      # EDA, feature engineering, model selection
│
└── src/
    ├── data/          # DB loading + player stats (Streamlit-cached)
    ├── features/      # On-the-fly feature computation for inference
    ├── models/        # Model loading, temperature scaling, prediction
    └── pipeline/      # Training pipeline: DataProcessor, FeatureEngineer, ModelTrainer
```

---

## Tech Stack

`Python 3.11` · `XGBoost` · `scikit-learn` · `Streamlit` · `Plotly` · `pandas` · `Docker`

---

*Vladyslav Antoniuk — Bachelor's thesis project, 2026*
