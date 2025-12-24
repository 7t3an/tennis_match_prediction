# Refactoring Architecture Changes

## Date: December 9, 2025

## Overview
Refactored `app.py` from monolithic 735-line file to clean modular architecture following separation of concerns principle.

## New Project Structure

```
src/
├── __init__.py
├── data/
│   ├── __init__.py
│   ├── data_loader.py       # Database loading, player extraction
│   └── player_stats.py      # Player statistics, match history
├── features/
│   ├── __init__.py
│   └── feature_calculator.py # Feature engineering (48 features)
└── models/
    ├── __init__.py
    └── predictor.py          # Model loading, prediction logic
```

## Module Breakdown

### `src/data/data_loader.py`
- **`load_database()`** - Loads train/test CSV files, caches with @st.cache_data
- **`get_unique_players()`** - Extracts unique players sorted by ATP rank

### `src/data/player_stats.py`
- **`get_player_stats()`** - Retrieves latest player statistics (rank, points, rolling stats)
- **`get_last_10_matches()`** - Formats recent match history for UI display

### `src/features/feature_calculator.py`
- **`calculate_features()`** - Computes all 48 features for match prediction
  - Seed features (is_seeded, seed_tier, seed_diff)
  - Rolling statistics (10-match window: aces, df, serve stats)
  - Head-to-head statistics
  - **Normalization**: Swaps P1/P2 so P1 = better ranked player (model requirement)
  - Returns: `(features_dict, needs_swap)`

### `src/models/predictor.py`
- **`load_model()`** - Loads XGBoost model + feature_columns.txt + label_encoders.pkl
- **`prepare_features_for_prediction()`** - Handles missing features, encodes categoricals
- **`predict_match()`** - Runs model.predict_proba(), inverts probabilities if swapped

### `app.py` (Refactored)
**Before**: 735 lines (UI + all business logic)  
**After**: ~400 lines (pure Streamlit UI only)

Imports:
```python
from src.data import load_database, get_unique_players, get_player_stats, get_last_10_matches
from src.features import calculate_features
from src.models import load_model, prepare_features_for_prediction, predict_match
```

## Benefits

1. **Separation of Concerns**: UI code separated from business logic
2. **Testability**: Each module can be unit tested independently
3. **Maintainability**: Easier to locate and modify specific functionality
4. **Reusability**: Modules can be imported by other scripts (e.g., `retrain_model.py`)
5. **Readability**: Clear module boundaries with docstrings

## Key Design Patterns

### Feature Normalization
Model trained with constraint: **P1 rank < P2 rank** (lower number = better)

**Example**: User selects Federer (rank=5) vs Nadal (rank=2)
1. `calculate_features()` detects P2 has better rank
2. Sets `needs_swap = True` and swaps all p1_/p2_ features
3. Model receives: P1=Nadal, P2=Federer
4. `predict_match()` inverts probabilities back to original order
5. UI shows: Federer=35%, Nadal=65%

### Caching Strategy
- `@st.cache_resource` for model (loaded once, never expires)
- `@st.cache_data` for database and players (refreshes on file change)

## Migration Notes

### No Logic Changes
- All formulas, calculations, and ML logic **unchanged**
- Only **organizational** refactoring
- Model metrics remain: 68.88% accuracy, 0.76 ROC-AUC

### Breaking Changes
- None (all imports handled internally)
- `app.py` remains entry point: `streamlit run app.py`

## Future Enhancements

1. Add `src/utils/` for shared helpers (e.g., confidence_level calculation)
2. Create `tests/` with unit tests for each module
3. Add type hints (Python 3.10+ type annotations)
4. Extract debug visualization to separate module
5. Consider FastAPI backend + React frontend for production

## Testing

Verified functionality:
- ✅ Streamlit app launches successfully
- ✅ Player selection works
- ✅ Feature calculation maintains same results
- ✅ Predictions match original implementation
- ✅ No import errors or missing dependencies
