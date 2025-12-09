"""
Конфігураційний файл для Tennis Prediction System
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict
from datetime import datetime

# === БАЗОВІ ШЛЯХИ ===
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
PREDICTIONS_DIR = DATA_DIR / 'predictions'
MODELS_DIR = DATA_DIR / 'models'

# Створити директорії якщо не існують
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, PREDICTIONS_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


# === GITHUB ДАТАСЕТ ===
@dataclass
class GitHubConfig:
    """Конфігурація для завантаження з GitHub"""
    repo_owner: str = 'JeffSackmann'
    repo_name: str = 'tennis_atp'
    branch: str = 'master'
    
    # Файли для завантаження
    match_files_pattern: str = 'atp_matches_{year}.csv'
    rankings_files_pattern: str = 'atp_rankings_{date}.csv'
    players_file: str = 'atp_players.csv'
    
    # Роки для завантаження
    start_year: int = 2010
    end_year: int = datetime.now().year
    
    @property
    def base_url(self) -> str:
        return f'https://raw.githubusercontent.com/{self.repo_owner}/{self.repo_name}/{self.branch}/'
    
    def get_matches_url(self, year: int) -> str:
        return f'{self.base_url}{self.match_files_pattern.format(year=year)}'
    
    def get_players_url(self) -> str:
        return f'{self.base_url}{self.players_file}'


# === БАЗА ДАНИХ ===
@dataclass
class DatabaseConfig:
    """Конфігурація бази даних"""
    db_type: str = os.getenv('DB_TYPE', 'sqlite')  # sqlite, postgresql
    db_name: str = os.getenv('DB_NAME', 'tennis_predictions.db')
    db_host: str = os.getenv('DB_HOST', 'localhost')
    db_port: int = int(os.getenv('DB_PORT', '5432'))
    db_user: str = os.getenv('DB_USER', '')
    db_password: str = os.getenv('DB_PASSWORD', '')
    
    @property
    def connection_string(self) -> str:
        if self.db_type == 'sqlite':
            return f'sqlite:///{BASE_DIR / self.db_name}'
        elif self.db_type == 'postgresql':
            return f'postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}'
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")


# === ETL КОНФІГУРАЦІЯ ===
@dataclass
class ETLConfig:
    """Конфігурація ETL процесу"""
    # Параметри обробки
    chunk_size: int = 10000
    min_matches_per_player: int = 5
    
    # Роки для обробки
    training_start_year: int = 2010
    training_end_year: int = 2024
    prediction_year: int = 2025
    
    # Колонки для обробки
    required_columns: List[str] = field(default_factory=lambda: [
        'tourney_id', 'tourney_name', 'surface', 'tourney_date',
        'winner_id', 'winner_name', 'winner_rank', 'winner_age',
        'loser_id', 'loser_name', 'loser_rank', 'loser_age',
        'score', 'round'
    ])
    
    # Валідні значення
    valid_surfaces: List[str] = field(default_factory=lambda: ['Hard', 'Clay', 'Grass', 'Carpet'])
    valid_rounds: List[str] = field(default_factory=lambda: ['F', 'SF', 'QF', 'R16', 'R32', 'R64', 'R128', 'RR'])


# === FEATURE ENGINEERING ===
@dataclass
class FeatureConfig:
    """Конфігурація створення фічей"""
    # ELO параметри
    elo_k_factor: float = 32.0
    elo_k_factor_grand_slam: float = 40.0
    elo_k_factor_masters: float = 36.0
    initial_elo: float = 1500.0
    
    # Форма гравця
    recent_matches_count: int = 10
    form_weight_decay: float = 0.9  # Більша вага останнім матчам
    
    # Head-to-head
    h2h_min_matches: int = 2
    h2h_years_back: int = 5
    
    # Статистика по покриттях
    surface_min_matches: int = 5
    
    # Фізична форма
    rest_days_optimal: int = 7
    rest_days_penalty_start: int = 30


# === ML МОДЕЛІ ===
@dataclass
class ModelConfig:
    """Конфігурація ML моделей"""
    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    
    # Моделі для навчання
    models_to_train: List[str] = field(default_factory=lambda: ['xgboost', 'catboost', 'lightgbm'])
    
    # XGBoost параметри
    xgboost_params: Dict = field(default_factory=lambda: {
        'n_estimators': 500,
        'max_depth': 7,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.1,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'random_state': 42
    })
    
    # CatBoost параметри
    catboost_params: Dict = field(default_factory=lambda: {
        'iterations': 1000,
        'depth': 6,
        'learning_rate': 0.05,
        'loss_function': 'Logloss',
        'eval_metric': 'Logloss',
        'random_seed': 42,
        'verbose': False
    })
    
    # LightGBM параметри
    lightgbm_params: Dict = field(default_factory=lambda: {
        'n_estimators': 500,
        'max_depth': 7,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'binary',
        'metric': 'binary_logloss',
        'random_state': 42,
        'verbose': -1
    })
    
    # Важливі фічі (приблизний порядок важливості)
    important_features: List[str] = field(default_factory=lambda: [
        'p1_elo', 'p2_elo', 'p1_rank', 'p2_rank',
        'p1_recent_form', 'p2_recent_form',
        'p1_surface_winrate', 'p2_surface_winrate',
        'h2h_p1_wins', 'h2h_total',
        'p1_age', 'p2_age',
        'p1_rest_days', 'p2_rest_days'
    ])


# === MLFLOW ===
@dataclass
class MLflowConfig:
    """Конфігурація MLflow"""
    tracking_uri: str = os.getenv('MLFLOW_TRACKING_URI', f'sqlite:///{BASE_DIR / "mlflow.db"}')
    experiment_name: str = 'tennis_prediction'
    artifact_location: str = str(BASE_DIR / 'mlruns')
    
    # Метрики для логування
    metrics_to_log: List[str] = field(default_factory=lambda: [
        'accuracy', 'precision', 'recall', 'f1',
        'roc_auc', 'log_loss', 'brier_score'
    ])


# === AIRFLOW ===
@dataclass
class AirflowConfig:
    """Конфігурація Airflow"""
    airflow_home: Path = BASE_DIR / 'airflow'
    dags_folder: Path = airflow_home / 'dags'
    
    # Розклад DAGs
    daily_update_schedule: str = '0 6 * * *'  # Щодня о 6:00
    weekly_retrain_schedule: str = '0 2 * * 0'  # Щонеділі о 2:00
    
    # Email налаштування (опціонально)
    email_on_failure: bool = True
    email_on_retry: bool = False
    email: str = os.getenv('AIRFLOW_EMAIL', '')


# === WEB SCRAPING (опціонально) ===
@dataclass
class ScraperConfig:
    """Конфігурація web scraping"""
    enabled: bool = False  # За замовчуванням вимкнено
    
    # ATP сайт
    atp_base_url: str = 'https://www.atptour.com'
    atp_rankings_url: str = f'{atp_base_url}/en/rankings/singles'
    
    # Затримки
    request_delay: float = 1.0  # Секунди між запитами
    timeout: int = 30
    
    # User agent
    user_agent: str = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'


# === STREAMLIT ===
@dataclass
class StreamlitConfig:
    """Конфігурація Streamlit додатку"""
    app_title: str = "🎾 Tennis Match Predictor"
    page_icon: str = "🎾"
    layout: str = "wide"
    
    # Кількість гравців для відображення
    top_players_count: int = 100
    recent_matches_display: int = 10
    
    # Графіки
    chart_height: int = 400
    chart_theme: str = "streamlit"
    
    # Кеш
    cache_ttl: int = 3600  # 1 година


# === ЛОГУВАННЯ ===
@dataclass
class LogConfig:
    """Конфігурація логування"""
    log_level: str = os.getenv('LOG_LEVEL', 'INFO')
    log_file: Path = BASE_DIR / 'logs' / 'tennis_prediction.log'
    log_format: str = '{time:YYYY-MM-DD HH:mm:ss} | {level} | {module}:{function}:{line} | {message}'
    rotation: str = '10 MB'
    retention: str = '1 month'


# === ГЛОБАЛЬНІ ІНСТАНСИ ===
github_config = GitHubConfig()
db_config = DatabaseConfig()
etl_config = ETLConfig()
feature_config = FeatureConfig()
model_config = ModelConfig()
mlflow_config = MLflowConfig()
airflow_config = AirflowConfig()
scraper_config = ScraperConfig()
streamlit_config = StreamlitConfig()
log_config = LogConfig()


# === ДОПОМІЖНІ ФУНКЦІЇ ===
def get_year_range() -> range:
    """Повертає діапазон років для обробки"""
    return range(etl_config.training_start_year, etl_config.training_end_year + 1)


def is_grand_slam(tourney_name: str) -> bool:
    """Перевіряє чи турнір є Grand Slam"""
    grand_slams = ['Australian Open', 'Roland Garros', 'Wimbledon', 'US Open']
    return any(gs in tourney_name for gs in grand_slams)


def is_masters(tourney_name: str) -> bool:
    """Перевіряє чи турнір є Masters"""
    masters = ['Indian Wells', 'Miami', 'Monte Carlo', 'Madrid', 'Rome', 
               'Canada', 'Cincinnati', 'Shanghai', 'Paris']
    return any(m in tourney_name for m in masters)


def get_k_factor(tourney_name: str) -> float:
    """Повертає K-factor для ELO в залежності від турніру"""
    if is_grand_slam(tourney_name):
        return feature_config.elo_k_factor_grand_slam
    elif is_masters(tourney_name):
        return feature_config.elo_k_factor_masters
    else:
        return feature_config.elo_k_factor


if __name__ == '__main__':
    # Тестування конфігурації
    print("=== Tennis Prediction System Configuration ===\n")
    print(f"Base Directory: {BASE_DIR}")
    print(f"Database: {db_config.connection_string}")
    print(f"GitHub Repo: {github_config.base_url}")
    print(f"MLflow: {mlflow_config.tracking_uri}")
    print(f"Training Years: {etl_config.training_start_year}-{etl_config.training_end_year}")
    print(f"Models: {', '.join(model_config.models_to_train)}")
    print("\n✓ Configuration loaded successfully!")