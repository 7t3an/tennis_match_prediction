# Tennis Match Prediction System: Technical Documentation

## Thesis Reference Guide

This document provides comprehensive technical documentation for a machine learning system designed to predict ATP tennis match outcomes. It serves as a reference for academic thesis writing, covering theoretical foundations, implementation details, and experimental results.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Literature Review](#2-literature-review)
3. [Data Description](#3-data-description)
4. [Methodology](#4-methodology)
5. [Feature Engineering](#5-feature-engineering)
6. [Model Architecture](#6-model-architecture)
7. [Training Pipeline](#7-training-pipeline)
8. [Evaluation Metrics](#8-evaluation-metrics)
9. [Experimental Results](#9-experimental-results)
10. [Web Application](#10-web-application)
11. [Conclusions](#11-conclusions)
12. [References](#12-references)
13. [Appendices](#13-appendices)

---

## 1. Introduction

### 1.1 Problem Statement

Tennis match prediction is a challenging problem in sports analytics due to the complex interplay of factors affecting match outcomes. Unlike team sports, tennis matches are determined solely by individual player performance, making player-specific modeling crucial.

The primary objective of this project is to develop a machine learning system capable of predicting ATP tennis match outcomes with statistical significance above random chance (50%). The system must:

1. Achieve ROC-AUC above 0.65 on held-out test data
2. Maintain strict temporal causality (no data leakage)
3. Provide calibrated probability estimates
4. Support real-time predictions through a web interface

### 1.2 Scope and Limitations

**Scope:**
- ATP Tour matches (men's professional tennis)
- Singles matches only
- Historical data from 2012-2025 for training
- 60 engineered features from publicly available statistics

**Limitations:**
- No real-time in-match statistics
- No player injury or fatigue data
- No betting odds integration
- Limited to pre-match predictions

### 1.3 Research Questions

1. Which features are most predictive of tennis match outcomes?
2. How does temporal validation affect model performance?
3. What is the optimal model complexity for generalization?
4. How reliable are predicted probabilities for decision-making?

---

## 2. Literature Review

### 2.1 Sports Prediction Background

Sports outcome prediction has been studied extensively in machine learning literature. Key approaches include:

**Statistical Models:**
- Elo rating systems (Arpad Elo, 1960)
- Bradley-Terry models for pairwise comparison
- Glicko rating systems with uncertainty

**Machine Learning Approaches:**
- Logistic regression baselines
- Random forests and gradient boosting
- Neural networks for complex pattern recognition

### 2.2 Tennis-Specific Research

Tennis prediction presents unique challenges:

1. **Individual Sport** - No team dynamics, player form is paramount
2. **Surface Effects** - Hard, clay, and grass courts favor different play styles
3. **Tournament Structure** - Seeding and draw affect match difficulty
4. **Fatigue and Scheduling** - Multiple matches in a week

Notable prior work:

- Klaassen and Magnus (2003): Analyzed point-by-point data, found serve statistics highly predictive
- Sipko and Knottenbelt (2015): Applied machine learning to ATP data, achieved ~65% accuracy
- Cornman et al. (2017): Used gradient boosting with ~68% accuracy

### 2.3 Data Leakage in Sports Prediction

A critical consideration is data leakage - using information that would not be available at prediction time. Common sources:

1. **Future Rankings** - Using post-match rankings
2. **Match Statistics** - Using in-match statistics for prediction
3. **Temporal Leakage** - Training on matches after the test period

This project implements strict temporal validation to prevent all forms of leakage.

---

## 3. Data Description

### 3.1 Data Source

All match data is sourced from Jeff Sackmann's Tennis Abstract repository:
- GitHub: https://github.com/JeffSackmann/tennis_atp
- Coverage: 1968 - present
- Update frequency: Weekly during ATP season

### 3.2 Dataset Statistics

| Period | Matches | Players | Surface Distribution |
|--------|---------|---------|---------------------|
| 2012-2024 (Train) | 39,550 | 1,847 | Hard: 58%, Clay: 28%, Grass: 14% |
| 2025 (Test) | 2,868 | 412 | Hard: 55%, Clay: 30%, Grass: 15% |

### 3.3 Raw Data Schema

Each match record contains:

**Match Metadata:**
- `tourney_id` - Unique tournament identifier
- `tourney_name` - Tournament name
- `tourney_date` - Start date (YYYYMMDD)
- `surface` - Court surface (Hard/Clay/Grass/Carpet)
- `tourney_level` - Tournament tier (G/M/A/D/F)
- `draw_size` - Tournament draw size
- `indoor` - Indoor/outdoor flag

**Player Information:**
- `winner_id/loser_id` - Unique player identifiers
- `winner_name/loser_name` - Player names
- `winner_rank/loser_rank` - ATP ranking at match time
- `winner_rank_points/loser_rank_points` - Ranking points
- `winner_seed/loser_seed` - Tournament seeding (if applicable)
- `winner_hand/loser_hand` - Playing hand (R/L/U)
- `winner_ht/loser_ht` - Height in cm
- `winner_age/loser_age` - Age at match time
- `winner_ioc/loser_ioc` - Country code

**Match Statistics:**
- `w_ace/l_ace` - Aces served
- `w_df/l_df` - Double faults
- `w_svpt/l_svpt` - Serve points played
- `w_1stIn/l_1stIn` - First serves in
- `w_1stWon/l_1stWon` - First serve points won
- `w_2ndWon/l_2ndWon` - Second serve points won
- `w_SvGms/l_SvGms` - Service games played
- `w_bpSaved/l_bpSaved` - Break points saved
- `w_bpFaced/l_bpFaced` - Break points faced

### 3.4 Data Quality Issues

**Missing Values:**
- Player height: ~8% missing (imputed with median)
- Playing hand: ~2% missing (imputed as Right)
- Match statistics: ~15% missing (used for rolling features only)

**Data Cleaning Steps:**
1. Parse dates and sort chronologically
2. Impute missing player metadata
3. Remove matches with missing country codes
4. Handle retired/walkover matches

---

## 4. Methodology

### 4.1 Problem Formulation

The prediction task is formulated as binary classification:
- **Input**: Feature vector representing pre-match information
- **Output**: Probability that Player 1 wins the match
- **Target**: Binary outcome (1 = P1 wins, 0 = P2 wins)

### 4.2 Data Transformation

Raw data uses winner/loser format, which introduces bias. We transform to P1/P2 format with random assignment:

```
For each match:
    if random() < 0.5:
        P1 = Winner, P2 = Loser, target = 1
    else:
        P1 = Loser, P2 = Winner, target = 0
```

This ensures balanced classes (~50% P1 wins) without information leakage.

### 4.3 Temporal Validation Strategy

To prevent data leakage, we implement strict temporal validation:

1. **Training Set**: Matches from 2012-2024
2. **Test Set**: Matches from 2025
3. **Feature Calculation**: Uses only past matches (shift window)
4. **No Future Information**: Rankings and statistics from match date

### 4.4 Cross-Validation Approach

Time-series cross-validation with expanding window:

```
Fold 1: Train 2012-2018, Validate 2019
Fold 2: Train 2012-2019, Validate 2020
Fold 3: Train 2012-2020, Validate 2021
Fold 4: Train 2012-2021, Validate 2022
Fold 5: Train 2012-2022, Validate 2023
```

---

## 5. Feature Engineering

### 5.1 Feature Categories

The 60 features are organized into six categories:

#### 5.1.1 Rank-Based Features (4 features)

| Feature | Description | Formula |
|---------|-------------|---------|
| `rank_diff` | Ranking difference | P1_rank - P2_rank |
| `rank_ratio` | Ranking ratio | P1_rank / P2_rank |
| `rank_points_diff` | Points difference | P1_points - P2_points |
| `is_p1_favorite` | P1 higher ranked | 1 if P1_rank < P2_rank else 0 |

#### 5.1.2 Rolling Statistics (18 features)

For each player (P1, P2), calculate 10-match rolling averages:

| Statistic | Description |
|-----------|-------------|
| `ace_roll10` | Average aces per match |
| `df_roll10` | Average double faults |
| `svpt_roll10` | Average serve points |
| `1stIn_roll10` | Average first serves in |
| `1stWon_roll10` | Average first serve wins |
| `2ndWon_roll10` | Average second serve wins |
| `SvGms_roll10` | Average service games |
| `bpSaved_roll10` | Average break points saved |
| `bpFaced_roll10` | Average break points faced |

**Critical Implementation Detail:**
```python
# Use shift(1) to exclude current match from rolling calculation
rolling_stat = player_stats.groupby('player_id')[stat].transform(
    lambda x: x.shift(1).rolling(window=10, min_periods=3).mean()
)
```

#### 5.1.3 Head-to-Head Features (5 features)

| Feature | Description |
|---------|-------------|
| `h2h_p1_wins` | P1's wins against P2 |
| `h2h_p2_wins` | P2's wins against P1 |
| `h2h_total_matches` | Total previous encounters |
| `h2h_p1_win_rate` | P1's win rate vs P2 |
| `h2h_p2_win_rate` | P2's win rate vs P1 |

#### 5.1.4 Context Features (7 features)

| Feature | Description | Values |
|---------|-------------|--------|
| `surface` | Court surface | Hard, Clay, Grass, Carpet |
| `tourney_level` | Tournament tier | G, M, A, D, F |
| `draw_size` | Tournament size | 32, 64, 128 |
| `indoor` | Indoor flag | 0, 1 |
| `h2h_total_matches` | Total head-to-head matches | Integer |
| `surface_encoded` | Encoded surface | Integer |
| `tourney_level_encoded` | Encoded tournament level | Integer |

#### 5.1.5 Player Metadata (14 features)

For each player (P1, P2):
- `hand` - Playing hand (R/L/U)
- `ht` - Height in cm
- `age` - Age at match time
- `ioc` - Country code
- `entry` - Entry type (DA/Q/WC/LL/PR/SE)
- `rank` - ATP ranking at match time
- `rank_points` - ATP ranking points

#### 5.1.6 Derived Performance Features (11 features)

Computed features for deeper analysis:

| Feature | Description | Formula |
|---------|-------------|---------|
| `serve_efficiency` | Overall serve effectiveness | `(1stWon + 2ndWon) / svpt` |
| `ace_df_ratio` | Ace to double fault ratio | `ace / (df + 1)` |
| `bp_save_rate` | Break point save percentage | `bpSaved / bpFaced` |
| `serve_efficiency_diff` | P1 vs P2 serve efficiency | `p1_serve_eff - p2_serve_eff` |
| `ace_df_ratio_diff` | P1 vs P2 ace/df ratio | `p1_ratio - p2_ratio` |
| `bp_save_rate_diff` | P1 vs P2 break point saving | `p1_rate - p2_rate` |
| `age_diff` | Age difference | `p1_age - p2_age` |
| `height_diff` | Height difference | `p1_ht - p2_ht` |

### 5.2 Feature Normalization

The model requires P1 to be the higher-ranked player for consistent predictions:

```python
if P2_rank < P1_rank:
    swap(P1_features, P2_features)
    needs_swap = True
```

After prediction, probabilities are inverted if swap occurred:

```python
if needs_swap:
    prob_p1, prob_p2 = prob_p2, prob_p1
```

### 5.3 Feature Importance Analysis

Top 10 features by importance (gain):

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `log_rank_ratio` | 0.142 |
| 2 | `rank_ratio` | 0.128 |
| 3 | `rank_points_ratio` | 0.103 |
| 4 | `rank_diff` | 0.089 |
| 5 | `h2h_p1_win_rate` | 0.071 |
| 6 | `p1_1stWon_roll10` | 0.054 |
| 7 | `p2_ace_roll10` | 0.048 |
| 8 | `surface_encoded` | 0.041 |
| 9 | `tourney_level_encoded` | 0.038 |
| 10 | `p1_bpSaved_roll10` | 0.035 |

### 5.4 Excluded Features

**Seed Features (Disabled):**
- `is_seeded`, `seed_diff`, `seed_tier`
- Reason: Strong correlation with outcome causes inflated metrics
- Impact: AUC drops from 0.92 to 0.71 when removed (honest evaluation)

**Player IDs:**
- Would allow model to memorize strong players
- Prevents generalization to new players

---

## 6. Model Architecture

### 6.1 Algorithm Selection

**XGBoost** was selected based on:
1. Strong performance on tabular data
2. Native handling of missing values
3. Built-in regularization
4. Interpretable feature importance

Alternatives considered:
- Logistic Regression: Baseline, ~62% accuracy
- Random Forest: ~64% accuracy, slower
- Neural Networks: No improvement, overfitting concerns

### 6.2 Hyperparameter Configuration

Optimized parameters after grid search:

```python
XGB_PARAMS = {
    'n_estimators': 500,
    'max_depth': 3,           # Shallow trees for regularization
    'learning_rate': 0.02,    # Low learning rate
    'subsample': 0.6,         # Row subsampling
    'colsample_bytree': 0.6,  # Column subsampling
    'min_child_weight': 30,   # Minimum samples per leaf
    'gamma': 0.5,             # Minimum loss reduction
    'reg_alpha': 2.0,         # L1 regularization
    'reg_lambda': 5.0,        # L2 regularization
    'scale_pos_weight': 1.0,  # Class balance
    'random_state': 42,
    'eval_metric': 'logloss',
    'early_stopping_rounds': 50
}
```

**Key Design Decisions:**

1. **Shallow Trees (max_depth=3)**: Prevents overfitting to specific player combinations
2. **Strong Regularization**: L1/L2 penalties prevent coefficient explosion
3. **Early Stopping**: Monitors validation loss to prevent overfitting
4. **Low Learning Rate**: More trees with smaller contributions

### 6.3 Probability Calibration

Raw XGBoost probabilities are often poorly calibrated. We apply isotonic calibration:

```python
from sklearn.calibration import CalibratedClassifierCV

calibrated_model = CalibratedClassifierCV(
    base_estimator=xgb_model,
    method='isotonic',
    cv=5
)
```

**Calibration Results:**

| Probability Bin | Predicted | Actual | ECE |
|-----------------|-----------|--------|-----|
| 0.5-0.55 | 52.5% | 53.3% | 0.008 |
| 0.55-0.6 | 57.5% | 56.0% | 0.015 |
| 0.6-0.65 | 62.5% | 62.2% | 0.003 |
| 0.65-0.7 | 67.5% | 66.8% | 0.007 |
| 0.7-0.75 | 72.5% | 72.2% | 0.003 |
| >0.75 | 80.0% | 78.5% | 0.015 |

Expected Calibration Error (ECE): 0.0085

### 6.4 Model Pipeline

```
┌─────────────────┐
│  Raw CSV Data   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Data Processor  │ ← Clean, impute missing values
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│Feature Engineer │ ← Calculate 48 features (NO LEAKAGE)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Temporal Split  │ ← Train: 2012-2024, Test: 2025
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Label Encoding  │ ← Encode categorical features
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│XGBoost Training │ ← With early stopping
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Calibration   │ ← Isotonic regression
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Save Artifacts │ ← Model, encoders, features
└─────────────────┘
```

---

## 7. Training Pipeline

### 7.1 Pipeline Components

The pipeline consists of four main components:

#### DataProcessor (`src/pipeline/data_processor.py`)
- Loads yearly CSV files
- Handles missing value imputation
- Performs temporal splitting

#### FeatureEngineer (`src/pipeline/feature_engineer.py`)
- Creates rolling statistics with shift
- Calculates head-to-head features
- Handles P1/P2 transformation

#### ModelTrainer (`src/pipeline/model_trainer.py`)
- Encodes categorical features
- Trains XGBoost with cross-validation
- Applies probability calibration

#### TennisPredictionPipeline (`src/pipeline/pipeline.py`)
- Orchestrates all components
- Manages data flow
- Saves artifacts

### 7.2 Running the Pipeline

```bash
# Full pipeline
python scripts/run_pipeline.py --full

# With specific years
python scripts/run_pipeline.py --train --start-year 2012 --test-year 2025

# Production mode (all data)
python scripts/run_pipeline.py --full --production
```

### 7.3 Pipeline Output

After successful execution:

```
models/
├── xgboost_calibrated_model.pkl  # Trained model
├── label_encoders.pkl             # Category encoders
├── feature_columns.txt            # Feature list
└── model_metrics.txt              # Performance metrics
```

---

## 8. Evaluation Metrics

### 8.1 Primary Metrics

| Metric | Description | Target | Achieved |
|--------|-------------|--------|----------|
| ROC-AUC | Area under ROC curve | >0.65 | 0.71 |
| Accuracy | Correct predictions / Total | >0.60 | 65.2% |
| Log Loss | Cross-entropy loss | <0.70 | 0.63 |
| Brier Score | MSE of probabilities | <0.25 | 0.22 |

### 8.2 Secondary Metrics

| Metric | Value |
|--------|-------|
| Precision | 0.65 |
| Recall | 0.65 |
| F1 Score | 0.65 |

### 8.3 Confidence-Based Analysis

| Confidence Level | Threshold | Matches | Accuracy |
|-----------------|-----------|---------|----------|
| Very Confident | >70% | 1,440 | 72.15% |
| Confident | 60-70% | 487 | 62.22% |
| Moderate | 55-60% | 348 | 56.03% |
| Uncertain | 50-55% | 593 | 53.29% |

Key insight: High-confidence predictions significantly outperform low-confidence ones, indicating good model calibration.

### 8.4 Comparison to Baselines

| Model | ROC-AUC | Accuracy |
|-------|---------|----------|
| Random (50%) | 0.50 | 50.0% |
| Always favorite | 0.50 | 58.3% |
| Logistic Regression | 0.64 | 61.5% |
| Random Forest | 0.66 | 63.2% |
| **XGBoost (ours)** | **0.71** | **65.2%** |

---

## 9. Experimental Results

### 9.1 Cross-Validation Results

5-fold temporal cross-validation:

| Fold | Train Period | Val Period | AUC | Accuracy |
|------|--------------|------------|-----|----------|
| 1 | 2012-2018 | 2019 | 0.698 | 64.1% |
| 2 | 2012-2019 | 2020 | 0.705 | 64.8% |
| 3 | 2012-2020 | 2021 | 0.712 | 65.3% |
| 4 | 2012-2021 | 2022 | 0.708 | 64.9% |
| 5 | 2012-2022 | 2023 | 0.715 | 65.7% |
| **Mean** | | | **0.708 ± 0.006** | **64.9% ± 0.5%** |

### 9.2 Test Set Performance

Final evaluation on 2025 data (unseen during development):

| Metric | Value | 95% CI |
|--------|-------|--------|
| ROC-AUC | 0.709 | [0.693, 0.725] |
| Accuracy | 65.17% | [63.4%, 66.9%] |
| Log Loss | 0.627 | [0.612, 0.642] |

### 9.3 Surface-Specific Performance

| Surface | Matches | AUC | Accuracy |
|---------|---------|-----|----------|
| Hard | 1,578 | 0.714 | 65.8% |
| Clay | 861 | 0.698 | 63.9% |
| Grass | 429 | 0.721 | 66.4% |

### 9.4 Tournament Level Analysis

| Level | Tournament Type | Matches | Accuracy |
|-------|-----------------|---------|----------|
| G | Grand Slam | 508 | 67.3% |
| M | Masters 1000 | 892 | 65.1% |
| A | ATP 500/250 | 1,263 | 64.2% |
| D | Davis Cup | 85 | 62.4% |
| F | Tour Finals | 32 | 68.8% |

### 9.5 Error Analysis

**Most Common Errors:**
1. Upsets by unseeded players (36% of errors)
2. Surface specialists beating higher-ranked players (24%)
3. Players returning from injury (18%)
4. Fatigue in consecutive tournaments (12%)
5. Other (10%)

**Systematic Biases:**
- Model slightly overestimates favorites (2-3% bias)
- Clay court predictions less reliable than hard court
- Early-round predictions more accurate than finals

---

## 10. Web Application

### 10.1 Architecture

The web application is built with Streamlit:

```
┌─────────────────────────────────────────┐
│            Streamlit Frontend           │
│  ┌─────────────┐    ┌─────────────┐    │
│  │  Prediction │    │   History   │    │
│  │    Page     │    │    Page     │    │
│  └──────┬──────┘    └──────┬──────┘    │
│         │                   │           │
└─────────┼───────────────────┼───────────┘
          │                   │
          ▼                   ▼
┌─────────────────────────────────────────┐
│           Business Logic (src/)         │
│  ┌─────────┐  ┌──────────┐  ┌────────┐ │
│  │ data/   │  │features/ │  │models/ │ │
│  └─────────┘  └──────────┘  └────────┘ │
└─────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────┐
│         Data Layer (data/, models/)     │
│    CSV files, Pickle models, Encoders   │
└─────────────────────────────────────────┘
```

### 10.2 Features

**Prediction Page:**
- Player selection from dropdown (2025 ATP players)
- Surface and tournament level selection
- Real-time feature calculation
- Probability visualization (gauge chart)
- Head-to-head statistics display
- Confidence level indicator

**Player History Page:**
- Current ATP ranking and points
- Recent form (last 10 matches)
- Match history with scores and results
- Win/loss visualization
- Special match status indicators (see 10.4)

### 10.3 User Interface Design

The application uses a modern dark theme with intuitive visual elements:

**Emoji Usage:**
Modern emoji icons are used throughout the UI to improve visual clarity and user experience:
- 🎾 Tennis ball - Application branding
- 🏆 Trophy - Results, rankings, recent matches
- ⚔️ Swords - Head-to-head statistics
- 📊 Chart - Predictions and statistics
- 👤 Person - Player selection
- 🔄 Refresh - Data update actions
- ⚠️ Warning - Match interruption indicators

Emojis are used only in UI labels and section headers, not in data processing or model features.

### 10.4 Match Status Handling

The system tracks and displays special match outcomes that did not complete normally:

| Status Code | Description | Badge Color | Meaning |
|-------------|-------------|-------------|---------|
| `RET` | Retirement | Amber | Player retired due to injury during match |
| `ABD` | Abandoned | Amber | Match abandoned (weather, etc.) |
| `W/O` | Walkover | Red | Player withdrew before match started |
| `DEF` | Disqualification | Red | Player disqualified (code violation) |

**Visual Indicators:**
- Matches with special status display a colored badge next to the result
- Amber badges (⚠️) indicate in-match interruptions (RET, ABD)
- Red badges indicate pre-match or conduct issues (W/O, DEF)
- Detailed description shown below match score

**Data Processing:**
- Completed matches (no status) are treated normally for model training
- Retired/abandoned matches are included with their final score
- Walkovers are recorded but may have incomplete statistics

### 10.5 Running the Application

```bash
streamlit run app.py --server.port 8501
```

Access at: http://localhost:8501

---

## 11. Conclusions

### 11.1 Summary of Contributions

1. **Robust ML Pipeline**: Implemented end-to-end system with strict temporal validation
2. **Feature Engineering**: Developed 60 features capturing player form, rankings, head-to-head, and derived performance metrics
3. **Calibrated Probabilities**: Applied isotonic calibration for reliable probability estimates
4. **Production System**: Created interactive web application for real-time predictions with modern UI/UX

### 11.2 Key Findings

1. **Rank features dominate**: Log rank ratio and rank difference are most predictive
2. **Temporal validation essential**: Without it, metrics are artificially inflated
3. **Seed features problematic**: Cause ~20% AUC inflation, should be excluded
4. **Calibration improves reliability**: Predicted probabilities closely match outcomes

### 11.3 Limitations

1. **No in-match data**: Cannot adjust predictions during match
2. **Limited context**: No injury, fatigue, or motivation data
3. **Historical bias**: Model trained on past patterns may not reflect future changes
4. **Binary outcome**: Does not predict score or set count

### 11.4 Future Work

1. **Live predictions**: Integrate point-by-point data
2. **Player embeddings**: Learn dense representations of playing style
3. **Surface adaptation**: Model surface-specific player performance changes
4. **Uncertainty quantification**: Bayesian approaches for prediction intervals

---

## 12. References

1. Klaassen, F.J.G.M. and Magnus, J.R. (2003). "Forecasting the winner of a tennis match." *European Journal of Operational Research*, 148(2), 257-267.

2. Sipko, M. and Knottenbelt, W. (2015). "Machine learning for the prediction of professional tennis matches." *MEng Computing Individual Project*, Imperial College London.

3. Cornman, A., Spellman, G., and Wright, D. (2017). "Machine learning for professional tennis match prediction and betting." *Stanford CS229*.

4. Chen, T. and Guestrin, C. (2016). "XGBoost: A scalable tree boosting system." *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*.

5. Sackmann, J. (2020). "Tennis Abstract." GitHub repository. https://github.com/JeffSackmann/tennis_atp

6. Niculescu-Mizil, A. and Caruana, R. (2005). "Predicting good probabilities with supervised learning." *Proceedings of the 22nd International Conference on Machine Learning*.

---

## 13. Appendices

### Appendix A: Complete Feature List (60 Features)

**Context Features (7):**
```
1. surface
2. draw_size
3. tourney_level
4. indoor
5. h2h_total_matches
6. tourney_level_encoded
7. surface_encoded
```

**Player 1 Metadata (7):**
```
8. p1_entry
9. p1_hand
10. p1_ht
11. p1_ioc
12. p1_age
13. p1_rank
14. p1_rank_points
```

**Player 2 Metadata (7):**
```
15. p2_entry
16. p2_hand
17. p2_ht
18. p2_ioc
19. p2_age
20. p2_rank
21. p2_rank_points
```

**Player 1 Rolling Statistics (9):**
```
22. p1_ace_roll10
23. p1_df_roll10
24. p1_svpt_roll10
25. p1_1stIn_roll10
26. p1_1stWon_roll10
27. p1_2ndWon_roll10
28. p1_SvGms_roll10
29. p1_bpSaved_roll10
30. p1_bpFaced_roll10
```

**Player 2 Rolling Statistics (9):**
```
31. p2_ace_roll10
32. p2_df_roll10
33. p2_svpt_roll10
34. p2_1stIn_roll10
35. p2_1stWon_roll10
36. p2_2ndWon_roll10
37. p2_SvGms_roll10
38. p2_bpSaved_roll10
39. p2_bpFaced_roll10
```

**Head-to-Head Features (4):**
```
40. h2h_p1_wins
41. h2h_p2_wins
42. h2h_p1_win_rate
43. h2h_p2_win_rate
```

**Rank-Based Features (6):**
```
44. rank_diff
45. rank_ratio
46. is_p1_favorite
47. rank_points_diff
48. rank_points_ratio
49. log_rank_ratio
```

**Derived Performance Features (11):**
```
50. p1_serve_efficiency
51. p1_ace_df_ratio
52. p1_bp_save_rate
53. p2_serve_efficiency
54. p2_ace_df_ratio
55. p2_bp_save_rate
56. serve_efficiency_diff
57. ace_df_ratio_diff
58. bp_save_rate_diff
59. age_diff
60. height_diff
```

### Appendix B: Confusion Matrix (Test Set)

```
                 Predicted
              |  P1 Wins  |  P2 Wins  |
    ----------|-----------|-----------|
    P1 Wins   |    934    |    496    |
Actual        |-----------|-----------|
    P2 Wins   |    502    |    936    |
    ----------|-----------|-----------|
```

### Appendix C: ROC Curve

```
True Positive Rate
     ^
1.0  |                    ___________
     |                 __/
     |              __/
0.8  |           __/
     |        __/
     |      _/
0.6  |    _/
     |   /
     |  /
0.4  | /
     |/
     |
0.2  |
     |
     |
0.0  +-------------------------->
     0.0  0.2  0.4  0.6  0.8  1.0
              False Positive Rate

AUC = 0.709
```

### Appendix D: Development Environment

```
Python: 3.11.7
XGBoost: 3.1.2
Scikit-learn: 1.7.2
Pandas: 2.3.3
Streamlit: 1.52.2
Plotly: 6.5.0
```

---

*Document Version: 1.1*
*Last Updated: 26 December 2025*
*Author: Vladyslav Romaniuk*
