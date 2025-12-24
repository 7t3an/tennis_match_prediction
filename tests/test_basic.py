"""Basic tests for Tennis Match Prediction project."""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_imports():
    """Test that all main modules can be imported."""
    from src.data import load_database, get_unique_players, get_player_stats, get_last_10_matches
    from src.features import calculate_features
    from src.models import load_model, prepare_features_for_prediction, predict_match
    
    assert callable(load_database)
    assert callable(get_unique_players)
    assert callable(get_player_stats)
    assert callable(get_last_10_matches)
    assert callable(calculate_features)
    assert callable(load_model)
    assert callable(prepare_features_for_prediction)
    assert callable(predict_match)


def test_config_imports():
    """Test that config module can be imported."""
    from config.settings import (
        BASE_DIR,
        DATA_DIR,
        github_config,
        db_config,
        model_config,
    )
    
    assert BASE_DIR.exists()
    assert DATA_DIR.exists()
    assert github_config.repo_owner == 'JeffSackmann'
    assert model_config.random_state == 42


def test_data_files_exist():
    """Test that required data files exist."""
    data_dir = PROJECT_ROOT / 'data' / 'processed'
    
    assert data_dir.exists(), f"Data directory not found: {data_dir}"
    
    required_files = [
        'train_features.csv',
        'test_features.csv',
    ]
    
    for filename in required_files:
        filepath = data_dir / filename
        assert filepath.exists(), f"Required file not found: {filepath}"


def test_model_files_exist():
    """Test that model files exist (may fail if model not trained yet)."""
    models_dir = PROJECT_ROOT / 'models'
    
    required_files = [
        'feature_columns.txt',
    ]
    
    for filename in required_files:
        filepath = models_dir / filename
        assert filepath.exists(), f"Model file not found: {filepath}"


def test_feature_columns_valid():
    """Test that feature columns file is valid."""
    features_path = PROJECT_ROOT / 'models' / 'feature_columns.txt'
    
    if features_path.exists():
        with open(features_path, 'r') as f:
            columns = [line.strip() for line in f.readlines() if line.strip()]
        
        assert len(columns) > 0, "Feature columns file is empty"
        assert 'p1_rank' in columns, "Expected 'p1_rank' in feature columns"
        assert 'p2_rank' in columns, "Expected 'p2_rank' in feature columns"
        assert 'surface' in columns, "Expected 'surface' in feature columns"


if __name__ == '__main__':
    print("Running basic tests...")
    
    try:
        test_imports()
        print("✓ test_imports passed")
    except Exception as e:
        print(f"✗ test_imports failed: {e}")
    
    try:
        test_config_imports()
        print("✓ test_config_imports passed")
    except Exception as e:
        print(f"✗ test_config_imports failed: {e}")
    
    try:
        test_data_files_exist()
        print("✓ test_data_files_exist passed")
    except Exception as e:
        print(f"✗ test_data_files_exist failed: {e}")
    
    try:
        test_model_files_exist()
        print("✓ test_model_files_exist passed")
    except Exception as e:
        print(f"✗ test_model_files_exist failed: {e}")
    
    try:
        test_feature_columns_valid()
        print("✓ test_feature_columns_valid passed")
    except Exception as e:
        print(f"✗ test_feature_columns_valid failed: {e}")
    
    print("\nDone!")
