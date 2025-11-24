import os
from pathlib import Path
from dataclasses import dataclass

# Базові шляхи
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
MODELS_DIR = DATA_DIR / 'models'

# GitHub dataset
GITHUB_REPO = 'JeffSackmann/tennis_atp'
GITHUB_RAW_URL = 'https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/'

# База даних
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///tennis_predictions.db')

# MLflow
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'sqlite:///mlflow.db')
MLFLOW_EXPERIMENT_NAME = 'tennis_prediction'

# Параметри моделей
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Airflow
AIRFLOW_HOME = os.getenv('AIRFLOW_HOME', BASE_DIR / 'airflow')

