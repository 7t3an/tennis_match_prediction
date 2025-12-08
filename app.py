import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import plotly.graph_objects as go
from datetime import datetime

# Налаштування сторінки
st.set_page_config(
    page_title="🎾 Tennis Match Predictor",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Кастомні стилі
st.markdown("""
    <style>
    .main {
        padding: 1rem 2rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 18px;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 10px;
        border: none;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    .player-card {
        padding: 1.5rem;
        border-radius: 12px;
        background-color: #f8f9fa;
        border: 2px solid #e9ecef;
        margin-bottom: 1rem;
    }
    .match-card {
        padding: 1rem;
        border-radius: 8px;
        background-color: white;
        border-left: 4px solid #667eea;
        margin-bottom: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stats-box {
        padding: 1rem;
        border-radius: 8px;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# ========== ЗАВАНТАЖЕННЯ ДАНИХ ==========

@st.cache_resource
def load_model():
    """Завантаження моделі, features та label encoders"""
    try:
        model_path = Path('saved models/xgboost_calibrated_model.pkl')
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        features_path = Path('saved models/feature_columns.txt')
        with open(features_path, 'r') as f:
            feature_cols = [line.strip() for line in f.readlines()]
        
        # Завантажуємо LabelEncoders
        encoders_path = Path('saved models/label_encoders.pkl')
        with open(encoders_path, 'rb') as f:
            label_encoders = pickle.load(f)
        
        return model, feature_cols, label_encoders
    except Exception as e:
        st.error(f"❌ Помилка завантаження моделі: {e}")
        return None, None, None

@st.cache_data
def load_database():
    """Завантаження бази даних для прогнозування"""
    try:
        # Для PRODUCTION використовуємо TEST 2025 (найсвіжіші дані)
        # Це дає актуальні ранги, форму, статистику гравців
        test_df = pd.read_csv('data/processed/test_features.csv')
        
        # Для H2H та історії також додаємо train (минулі матчі)
        train_df = pd.read_csv('data/processed/train_features.csv')
        
        # Об'єднуємо: спочатку старі дані (train), потім нові (test)
        # Це важливо щоб при сортуванні останні рядки були найновіші
        full_df = pd.concat([train_df, test_df], ignore_index=True)
        
        return full_df, test_df
    except Exception as e:
        st.error(f"❌ Помилка завантаження даних: {e}")
        return None, None

@st.cache_data
def get_unique_players(_df):
    """Отримання унікальних гравців з актуальними даними"""
    # Беремо тільки з TEST 2025 (найсвіжіші дані)
    # Сортуємо за індексом (останні рядки = найновіші матчі)
    _df_sorted = _df.sort_index(ascending=False)
    
    # Гравці P1
    p1_players = _df_sorted[['p1_name', 'p1_rank', 'p1_rank_points']].rename(
        columns={'p1_name': 'name', 'p1_rank': 'rank', 'p1_rank_points': 'points'}
    )
    
    # Гравці P2
    p2_players = _df_sorted[['p2_name', 'p2_rank', 'p2_rank_points']].rename(
        columns={'p2_name': 'name', 'p2_rank': 'rank', 'p2_rank_points': 'points'}
    )
    
    # Об'єднуємо
    all_players = pd.concat([p1_players, p2_players])
    
    # Беремо ПЕРШИЙ запис для кожного гравця (це буде ОСТАННІЙ матч, бо відсортовано)
    latest_players = all_players.drop_duplicates('name', keep='first').reset_index(drop=True)
    latest_players = latest_players.sort_values('rank').reset_index(drop=True)
    
    return latest_players

def get_player_stats(_df, player_name):
    """Отримання АКТУАЛЬНОЇ статистики гравця з ОСТАННЬОГО матчу"""
    # Знаходимо всі матчі гравця
    player_matches = _df[
        (_df['p1_name'] == player_name) | (_df['p2_name'] == player_name)
    ].copy()
    
    if len(player_matches) == 0:
        return None
    
    # Сортуємо за індексом (останні рядки = найновіші дані в нашій базі)
    # В test_features.csv останні рядки - це матчі 2025 року
    player_matches = player_matches.sort_index(ascending=False)
    
    # Беремо ОСТАННІЙ матч для актуальних даних
    latest_match = player_matches.iloc[0]
    
    # Визначаємо чи це P1 або P2
    is_p1 = latest_match['p1_name'] == player_name
    
    stats = {
        'name': player_name,
        'rank': latest_match['p1_rank'] if is_p1 else latest_match['p2_rank'],
        'rank_points': latest_match['p1_rank_points'] if is_p1 else latest_match['p2_rank_points'],
        'recent_matches': player_matches.head(10),  # Останні 10 матчів
    }
    
    # Додаємо rolling статистику з ОСТАННЬОГО матчу
    rolling_cols = [col for col in player_matches.columns if '_roll10' in col]
    if rolling_cols:
        prefix = 'p1_' if is_p1 else 'p2_'
        for col in rolling_cols:
            if col.startswith(prefix):
                feature_name = col  # Зберігаємо повну назву з префіксом
                stats[feature_name] = latest_match[col]
    
    return stats

def get_last_10_matches(_df, player_name):
    """Отримання останніх 10 матчів гравця для відображення"""
    player_matches = _df[
        (_df['p1_name'] == player_name) | (_df['p2_name'] == player_name)
    ].copy()
    
    if len(player_matches) == 0:
        return pd.DataFrame()
    
    # Сортуємо за датою
    if 'tourney_date' in player_matches.columns:
        player_matches = player_matches.sort_values('tourney_date', ascending=False)
    
    # Беремо останні 10
    recent_matches = player_matches.head(10).copy()
    
    # Форматуємо для відображення
    formatted_matches = []
    for _, match in recent_matches.iterrows():
        is_p1 = match['p1_name'] == player_name
        opponent = match['p2_name'] if is_p1 else match['p1_name']
        
        # Визначаємо результат
        if is_p1:
            won = match['p1_won'] == 1
        else:
            won = match['p1_won'] == 0
        
        formatted_matches.append({
            'Date': match.get('tourney_date', 'N/A'),
            'Tournament': match.get('tourney_name', 'N/A'),
            'Surface': match.get('surface', 'N/A'),
            'Opponent': opponent,
            'Result': '✅ WIN' if won else '❌ LOSS',
            'Score': match.get('score', 'N/A'),
        })
    
    return pd.DataFrame(formatted_matches)

def calculate_features(_df, p1_name, p2_name, surface='Hard', tourney_level='A'):
    """
    Автоматичне обчислення ВСІХ 53 features для моделі з АКТУАЛЬНИХ даних
    
    ⚠️ ВАЖЛИВО: НЕ міняємо гравців місцями!
    Модель приймає будь-які два гравці і повертає ймовірність що P1 виграє.
    
    Повертає: features_dict (словник з 53 features)
    """
    
    # Отримуємо статистику гравців з ОСТАННЬОГО матчу
    p1_stats = get_player_stats(_df, p1_name)
    p2_stats = get_player_stats(_df, p2_name)
    
    if p1_stats is None or p2_stats is None:
        return None
    
    # Знаходимо останні матчі для отримання всіх даних
    p1_matches = _df[(_df['p1_name'] == p1_name) | (_df['p2_name'] == p1_name)].sort_index(ascending=False)
    p2_matches = _df[(_df['p1_name'] == p2_name) | (_df['p2_name'] == p2_name)].sort_index(ascending=False)
    
    p1_latest = p1_matches.iloc[0]
    p2_latest = p2_matches.iloc[0]
    
    # Визначаємо позицію в матчі
    p1_is_p1 = p1_latest['p1_name'] == p1_name
    p2_is_p1 = p2_latest['p1_name'] == p2_name
    
    # === БАЗОВІ FEATURES ===
    features = {}
    
    # Surface та tournament - ЗБЕРІГАЄМО ЯК ТЕКСТ (як в тренувальних даних!)
    features['surface'] = surface  # 'Hard', 'Clay', 'Grass', 'Carpet'
    features['tourney_level'] = tourney_level  # 'G', 'M', 'A', 'D', 'F', '250', '500'
    features['draw_size'] = 128  # За замовчуванням
    features['indoor'] = 0  # За замовчуванням outdoor
    
    # === P1 FEATURES ===
    features['p1_rank'] = p1_latest['p1_rank'] if p1_is_p1 else p1_latest['p2_rank']
    features['p1_rank_points'] = p1_latest['p1_rank_points'] if p1_is_p1 else p1_latest['p2_rank_points']
    features['p1_seed'] = p1_latest.get('p1_seed', np.nan) if p1_is_p1 else p1_latest.get('p2_seed', np.nan)
    features['p1_entry'] = p1_latest.get('p1_entry', 'DA') if p1_is_p1 else p1_latest.get('p2_entry', 'DA')
    features['p1_hand'] = p1_latest.get('p1_hand', 'R') if p1_is_p1 else p1_latest.get('p2_hand', 'R')
    features['p1_ht'] = p1_latest.get('p1_ht', 180) if p1_is_p1 else p1_latest.get('p2_ht', 180)
    features['p1_ioc'] = p1_latest.get('p1_ioc', 'ESP') if p1_is_p1 else p1_latest.get('p2_ioc', 'ESP')  # ЗБЕРІГАЄМО ЯК ТЕКСТ
    features['p1_age'] = p1_latest.get('p1_age', 25) if p1_is_p1 else p1_latest.get('p2_age', 25)
    
    # === P2 FEATURES ===
    features['p2_rank'] = p2_latest['p1_rank'] if p2_is_p1 else p2_latest['p2_rank']
    features['p2_rank_points'] = p2_latest['p1_rank_points'] if p2_is_p1 else p2_latest['p2_rank_points']
    features['p2_seed'] = p2_latest.get('p1_seed', np.nan) if p2_is_p1 else p2_latest.get('p2_seed', np.nan)
    features['p2_entry'] = p2_latest.get('p1_entry', 'DA') if p2_is_p1 else p2_latest.get('p2_entry', 'DA')
    features['p2_hand'] = p2_latest.get('p1_hand', 'R') if p2_is_p1 else p2_latest.get('p2_hand', 'R')
    features['p2_ht'] = p2_latest.get('p1_ht', 180) if p2_is_p1 else p2_latest.get('p2_ht', 180)
    features['p2_ioc'] = p2_latest.get('p1_ioc', 'ESP') if p2_is_p1 else p2_latest.get('p2_ioc', 'ESP')  # ЗБЕРІГАЄМО ЯК ТЕКСТ
    features['p2_age'] = p2_latest.get('p1_age', 25) if p2_is_p1 else p2_latest.get('p2_age', 25)
    
    # === SEED FEATURES ===
    features['is_p1_seeded'] = not pd.isna(features['p1_seed']) and features['p1_seed'] > 0
    features['is_p2_seeded'] = not pd.isna(features['p2_seed']) and features['p2_seed'] > 0
    
    p1_seed_val = features['p1_seed'] if features['is_p1_seeded'] else 999
    p2_seed_val = features['p2_seed'] if features['is_p2_seeded'] else 999
    features['seed_diff'] = p1_seed_val - p2_seed_val
    
    # Seed tiers - ЯК ТЕКСТ (як в тренувальних даних!)
    def get_seed_tier(seed_val):
        if seed_val <= 4:
            return 'Top4'
        elif seed_val <= 8:
            return 'Top8'
        elif seed_val <= 16:
            return 'Top16'
        elif seed_val <= 32:
            return 'Top32+'
        else:
            return 'Not_Seeded'
    
    features['p1_seed_tier'] = get_seed_tier(p1_seed_val)
    features['p2_seed_tier'] = get_seed_tier(p2_seed_val)
    
    # === ROLLING FEATURES ===
    # P1 rolling stats
    for stat in ['ace', 'df', 'svpt', '1stIn', '1stWon', '2ndWon', 'SvGms', 'bpSaved', 'bpFaced']:
        col_name = f'p1_{stat}_roll10'
        if p1_is_p1:
            features[col_name] = p1_latest.get(col_name, 0)
        else:
            col_name_p2 = f'p2_{stat}_roll10'
            features[col_name] = p1_latest.get(col_name_p2, 0)
    
    # P2 rolling stats
    for stat in ['ace', 'df', 'svpt', '1stIn', '1stWon', '2ndWon', 'SvGms', 'bpSaved', 'bpFaced']:
        col_name = f'p2_{stat}_roll10'
        if p2_is_p1:
            col_name_p1 = f'p1_{stat}_roll10'
            features[col_name] = p2_latest.get(col_name_p1, 0)
        else:
            features[col_name] = p2_latest.get(col_name, 0)
    
    # === H2H FEATURES ===
    h2h_matches = _df[
        ((_df['p1_name'] == p1_name) & (_df['p2_name'] == p2_name)) |
        ((_df['p1_name'] == p2_name) & (_df['p2_name'] == p1_name))
    ]
    
    if len(h2h_matches) > 0:
        p1_wins = 0
        p2_wins = 0
        
        for _, match in h2h_matches.iterrows():
            if match['p1_name'] == p1_name:
                if match['p1_won'] == 1:
                    p1_wins += 1
                else:
                    p2_wins += 1
            else:
                if match['p1_won'] == 0:
                    p1_wins += 1
                else:
                    p2_wins += 1
        
        features['h2h_p1_wins'] = p1_wins
        features['h2h_p2_wins'] = p2_wins
        features['h2h_total_matches'] = len(h2h_matches)
        features['h2h_p1_win_rate'] = p1_wins / len(h2h_matches)
    else:
        features['h2h_p1_wins'] = 0
        features['h2h_p2_wins'] = 0
        features['h2h_total_matches'] = 0
        features['h2h_p1_win_rate'] = 0.5
    
    # === RANK FEATURES ВИДАЛЕНО ===
    # Асиметричні features (rank_diff, rank_ratio, rank_points_diff, is_p1_favorite, seed_diff)
    # були видалені з моделі для симетричності
    
    # === ENCODED FEATURES ===
    # Ці features будуть дублікатами після pd.Categorical().codes, 
    # але модель їх очікує, тому додаємо як текст (будуть закодовані пізніше)
    features['tourney_level_encoded'] = features['tourney_level']
    features['surface_encoded'] = features['surface']
    
    # 🔄 НОРМАЛІЗАЦІЯ: завжди P1 = кращий гравець
    # Модель тренувалась так що P1 завжди має кращий або рівний rank
    needs_swap = features['p2_rank'] < features['p1_rank']
    
    if needs_swap:
        # Міняємо місцями ВСІ P1/P2 features
        for key in list(features.keys()):
            if key.startswith('p1_'):
                p2_key = key.replace('p1_', 'p2_')
                if p2_key in features:
                    features[key], features[p2_key] = features[p2_key], features[key]
        
        # Міняємо H2H
        if 'h2h_p1_wins' in features and 'h2h_p2_wins' in features:
            features['h2h_p1_wins'], features['h2h_p2_wins'] = features['h2h_p2_wins'], features['h2h_p1_wins']
    
    return features, needs_swap  # Повертаємо чи були поміняні місцями

# ========== ІНТЕРФЕЙС ==========

# Завантаження
model, feature_cols, label_encoders = load_model()
db_result = load_database()

if db_result is None:
    st.error("❌ Не вдалося завантажити базу даних")
    st.stop()

db, test_db = db_result

if model is None or db is None or feature_cols is None or label_encoders is None:
    st.error("❌ Не вдалося завантажити модель або базу даних")
    st.stop()

# Для списку гравців використовуємо TEST 2025 (актуальні дані)
players_df = get_unique_players(test_db)

# Tabs (Сторінки)
tab1, tab2 = st.tabs(["🎯 Prediction", "📊 Player History"])

# ========== СТОРІНКА 1: PREDICTION ==========
with tab1:
    st.title("🎾 Tennis Match Prediction")
    st.markdown("### Виберіть двох гравців для прогнозу")
    st.markdown("---")
    
    # Форма вводу
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 👤 Player 1")
        p1_name = st.selectbox(
            "Оберіть гравця:",
            options=[''] + sorted(players_df['name'].tolist()),
            key='p1_name',
            format_func=lambda x: "-- Оберіть гравця --" if x == '' else x
        )
        
        if p1_name and p1_name != '':
            p1_stats = get_player_stats(db, p1_name)
            if p1_stats:
                st.markdown(f"""
                <div class="stats-box">
                    <h4>📊 Актуальні дані</h4>
                    <p><strong>Ранг ATP:</strong> {int(p1_stats['rank'])}</p>
                    <p><strong>Рейтингові очки:</strong> {int(p1_stats['rank_points'])}</p>
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 👤 Player 2")
        p2_name = st.selectbox(
            "Оберіть гравця:",
            options=[''] + sorted(players_df['name'].tolist()),
            key='p2_name',
            format_func=lambda x: "-- Оберіть гравця --" if x == '' else x
        )
        
        if p2_name and p2_name != '':
            p2_stats = get_player_stats(db, p2_name)
            if p2_stats:
                st.markdown(f"""
                <div class="stats-box">
                    <h4>📊 Актуальні дані</h4>
                    <p><strong>Ранг ATP:</strong> {int(p2_stats['rank'])}</p>
                    <p><strong>Рейтингові очки:</strong> {int(p2_stats['rank_points'])}</p>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Додаткові параметри (опціонально)
    st.markdown("### ⚙️ Додаткові параметри (опціонально)")
    
    col3, col4 = st.columns(2)
    
    with col3:
        surface = st.selectbox(
            "Покриття:",
            options=['Hard', 'Clay', 'Grass', 'Carpet'],
            index=0
        )
    
    with col4:
        tourney_level = st.selectbox(
            "Рівень турніру:",
            options=['G', 'M', 'A', 'D', 'F'],
            index=2,
            format_func=lambda x: {
                'G': 'Grand Slam',
                'M': 'Masters 1000',
                'A': 'ATP 500/250',
                'D': 'Davis Cup',
                'F': 'Tour Finals'
            }[x]
        )
    
    st.markdown("---")
    
    # Кнопка прогнозу
    if st.button("🎯 ЗРОБИТИ ПРОГНОЗ", type="primary"):
        if not p1_name or not p2_name or p1_name == '' or p2_name == '':
            st.error("❌ Оберіть обох гравців!")
        elif p1_name == p2_name:
            st.error("❌ Оберіть різних гравців!")
        else:
            with st.spinner('🔄 Витягуємо актуальні дані з бази та створюємо прогноз...'):
                # Зберігаємо оригінальні імена для відображення
                original_p1_name = p1_name
                original_p2_name = p2_name
                
                # Обчислюємо features автоматично з АКТУАЛЬНИХ даних
                result = calculate_features(db, p1_name, p2_name, surface, tourney_level)
                
                if result is None or result[0] is None:
                    st.error("❌ Не вдалося отримати дані для одного з гравців")
                else:
                    features_dict, swapped = result
                    
                    # Якщо поміняли місцями - запам'ятовуємо для інверсії результату
                    if swapped:
                        st.info(f"ℹ️ Для симетричності моделі: порівнюємо {p2_name} vs {p1_name}")
                    
                    # DEBUG: Показуємо які дані використовує модель
                    with st.expander("🔍 Debug: Дані які модель використовує", expanded=False):
                        st.write(f"**Swapped: {swapped}**")
                        st.write("**📊 БАЗОВІ ДАНІ (як бачить модель):**")
                        col_d1, col_d2 = st.columns(2)
                        with col_d1:
                            st.write(f"**P1: {p1_name}**")
                            st.write(f"- Ранг: {int(features_dict['p1_rank'])}")
                            st.write(f"- Очки: {int(features_dict['p1_rank_points'])}")
                            st.write(f"- Вік: {features_dict.get('p1_age', 'N/A')}")
                            st.write(f"- Рука: {features_dict.get('p1_hand', 'N/A')}")
                        with col_d2:
                            st.write(f"**{p2_name}:**")
                            st.write(f"- Ранг: {int(features_dict['p2_rank'])}")
                            st.write(f"- Очки: {int(features_dict['p2_rank_points'])}")
                            st.write(f"- Вік: {features_dict.get('p2_age', 'N/A')}")
                            st.write(f"- Рука: {features_dict.get('p2_hand', 'N/A')}")
                        
                        st.write(f"\n**🎯 RANK INFO:**")
                        st.write(f"- P1 rank: **{int(features_dict['p1_rank'])}**")
                        st.write(f"- P2 rank: **{int(features_dict['p2_rank'])}**")
                        st.write(f"- Кращий гравець: **{'P1' if features_dict['p1_rank'] < features_dict['p2_rank'] else 'P2' if features_dict['p2_rank'] < features_dict['p1_rank'] else 'Рівні'}**")
                        
                        st.write(f"\n**🤝 H2H:**")
                        st.write(f"- Всього матчів: {features_dict['h2h_total_matches']}")
                        if features_dict['h2h_total_matches'] > 0:
                            # Враховуємо що features могли бути поміняні
                            p1_display_name = original_p2_name if swapped else original_p1_name
                            p2_display_name = original_p1_name if swapped else original_p2_name
                            st.write(f"- {p1_display_name}: {features_dict['h2h_p1_wins']} перемог ({features_dict['h2h_p1_win_rate']:.0%})")
                            st.write(f"- {p2_display_name}: {features_dict['h2h_p2_wins']} перемог")
                        else:
                            st.write("- Немає попередніх зустрічей")
                    
                    # Створюємо DataFrame для моделі
                    input_df = pd.DataFrame([features_dict])
                    
                    # Додаємо відсутні features зі значеннями за замовчуванням
                    for col in feature_cols:
                        if col not in input_df.columns:
                            # Визначаємо тип даних за назвою колонки
                            if 'hand' in col:
                                input_df[col] = 'R'  # За замовчуванням правша
                            elif 'entry' in col:
                                input_df[col] = 'DA'  # За замовчуванням Direct Acceptance
                            elif 'ioc' in col:
                                input_df[col] = 'ESP'  # За замовчуванням Іспанія
                            else:
                                input_df[col] = 0
                    
                    # 🔥 КРИТИЧНО: Encode ВСІ categorical features використовуючи LabelEncoders!
                    categorical_features = input_df.select_dtypes(include=['object']).columns.tolist()
                    
                    if label_encoders is not None:  # Type guard для Pylance
                        for col in categorical_features:
                            if col in feature_cols:
                                # Для _encoded features використовуємо encoder базової колонки
                                encoder_col = col.replace('_encoded', '') if '_encoded' in col else col
                                
                                if encoder_col in label_encoders:
                                    encoder = label_encoders[encoder_col]  # Зберігаємо посилання
                                    # Заповнюємо NaN
                                    input_df[col] = input_df[col].fillna('MISSING').astype(str)
                                    # Handle unseen labels
                                    input_df[col] = input_df[col].apply(
                                        lambda x: x if x in encoder.classes_ else 'MISSING'
                                    )
                                    # Transform using LabelEncoder
                                    input_df[col] = encoder.transform(input_df[col])
                    
                    # Заповнюємо NaN у числових features
                    input_df = input_df.fillna(0)
                    
                    # Переупорядковуємо колонки згідно з тим що очікує модель
                    input_df = input_df[feature_cols]
                    
                    # DEBUG: Показуємо ВСІ features
                    st.write(f"**DEBUG: ALL {len(feature_cols)} FEATURES:**")
                    st.dataframe(input_df.T, use_container_width=True)
                    
                    # Прогноз
                    prob_p1_wins = model.predict_proba(input_df)[0, 1]
                    prob_p2_wins = 1 - prob_p1_wins
                    
                    # 🔄 ІНВЕРСІЯ якщо були поміняні місцями
                    if swapped:
                        # Модель передбачила для P1 (кращий), але це original_p2
                        # Тому міняємо місцями
                        prob_p1_wins, prob_p2_wins = prob_p2_wins, prob_p1_wins
                    
                    # DEBUG вивід
                    st.write(f"🔍 DEBUG: prob({original_p1_name})={prob_p1_wins:.3f}, prob({original_p2_name})={prob_p2_wins:.3f}, swapped={swapped}")
                    st.write(f"🔍 {original_p1_name}={prob_p1_wins:.3f}, {original_p2_name}={prob_p2_wins:.3f}")
                    
                    # Візуалізація
                    st.markdown("---")
                    st.markdown("## 📊 Результати прогнозу")
                    
                    # Gauge chart для ПЕРШОГО ОБРАНОГО гравця
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = prob_p1_wins * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': f"{original_p1_name} Win Probability", 'font': {'size': 24}},
                        delta = {'reference': 50, 'increasing': {'color': "green"}},
                        gauge = {
                            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                            'bar': {'color': "#667eea"},
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'bordercolor': "gray",
                            'steps': [
                                {'range': [0, 50], 'color': '#ffebee'},
                                {'range': [50, 100], 'color': '#e8eaf6'}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 50
                            }
                        }
                    ))
                    
                    fig.update_layout(
                        height=400,
                        margin=dict(l=20, r=20, t=80, b=20),
                        paper_bgcolor="white",
                        font={'color': "darkblue", 'family': "Arial"}
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Результати
                    col_res1, col_res2 = st.columns(2)
                    
                    # Отримуємо дані гравців з features
                    original_p1_rank = features_dict['p1_rank']
                    original_p1_points = features_dict['p1_rank_points']
                    original_p2_rank = features_dict['p2_rank']
                    original_p2_points = features_dict['p2_rank_points']
                    
                    with col_res1:
                        st.markdown(f"""
                        <div style="padding: 2rem; border-radius: 15px; 
                             background: linear-gradient(135deg, {'#4CAF50' if prob_p1_wins > 0.5 else '#9E9E9E'}, 
                                                                 {'#45a049' if prob_p1_wins > 0.5 else '#757575'});
                             color: white; text-align: center;">
                            <h2>👤 {original_p1_name}</h2>
                            <h1 style="font-size: 3rem; margin: 1rem 0;">{prob_p1_wins:.1%}</h1>
                            <p style="font-size: 1.2rem;">Шанс на перемогу</p>
                            <p style="font-size: 0.9rem; margin-top: 1rem;">
                                Ранг: {int(original_p1_rank)} | Очки: {int(original_p1_points)}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_res2:
                        st.markdown(f"""
                        <div style="padding: 2rem; border-radius: 15px; 
                             background: linear-gradient(135deg, {'#4CAF50' if prob_p2_wins > 0.5 else '#9E9E9E'}, 
                                                                 {'#45a049' if prob_p2_wins > 0.5 else '#757575'});
                             color: white; text-align: center;">
                            <h2>👤 {original_p2_name}</h2>
                            <h1 style="font-size: 3rem; margin: 1rem 0;">{prob_p2_wins:.1%}</h1>
                            <p style="font-size: 1.2rem;">Шанс на перемогу</p>
                            <p style="font-size: 0.9rem; margin-top: 1rem;">
                                Ранг: {int(original_p2_rank)} | Очки: {int(original_p2_points)}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Переможець
                    st.markdown("---")
                    
                    # DEBUG: показуємо значення перед визначенням фаворита
                    st.write(f"**DEBUG VALUES:**")
                    st.write(f"original_p1_name = {original_p1_name}")
                    st.write(f"original_p2_name = {original_p2_name}")
                    st.write(f"prob_p1_wins = {prob_p1_wins}")
                    st.write(f"prob_p2_wins = {prob_p2_wins}")
                    
                    favorite = original_p1_name if prob_p1_wins > prob_p2_wins else original_p2_name
                    favorite_prob = max(prob_p1_wins, prob_p2_wins)
                    
                    st.write(f"favorite = {favorite}")
                    st.write(f"favorite_prob = {favorite_prob}")
                    st.markdown("---")
                    
                    confidence_level = "дуже впевнений" if favorite_prob > 0.7 else \
                                     "впевнений" if favorite_prob > 0.6 else \
                                     "помірно впевнений" if favorite_prob > 0.55 else \
                                     "невпевнений"
                    
                    confidence_emoji = "🔥" if favorite_prob > 0.7 else \
                                      "✅" if favorite_prob > 0.6 else \
                                      "⚠️" if favorite_prob > 0.55 else "❓"
                    
                    st.markdown(f"""
                    <div style="padding: 2rem; border-radius: 15px; 
                         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                         color: white; text-align: center;">
                        <h2>{confidence_emoji} Прогноз моделі</h2>
                        <h1 style="font-size: 2.5rem; margin: 1rem 0;">
                            {favorite}
                        </h1>
                        <p style="font-size: 1.3rem;">
                            Модель {confidence_level} у перемозі<br/>
                            з ймовірністю <strong>{favorite_prob:.1%}</strong>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Деталі
                    st.markdown("---")
                    st.markdown("### 📈 Деталі аналізу")
                    
                    col_det1, col_det2, col_det3 = st.columns(3)
                    
                    with col_det1:
                        rank_diff = abs(features_dict['p1_rank'] - features_dict['p2_rank'])
                        st.metric(
                            "Різниця рангів",
                            f"{rank_diff}",
                            f"{'P1' if features_dict['p1_rank'] < features_dict['p2_rank'] else 'P2'} вище"
                        )
                    
                    with col_det2:
                        st.metric(
                            "H2H матчі",
                            f"{features_dict['h2h_total_matches']}",
                            f"P1: {features_dict['h2h_p1_win_rate']:.0%}" if features_dict['h2h_total_matches'] > 0 else "Немає даних"
                        )
                    
                    with col_det3:
                        expected_accuracy = 73.68 if favorite_prob > 0.7 else \
                                          62.51 if favorite_prob > 0.6 else \
                                          58.55 if favorite_prob > 0.55 else \
                                          52.23
                        
                        st.metric(
                            "Очікувана точність",
                            f"{expected_accuracy:.1f}%",
                            f"{confidence_level}"
                        )

# ========== СТОРІНКА 2: PLAYER HISTORY ==========
with tab2:
    st.title("📊 Player Match History")
    st.markdown("### Перегляньте останні матчі будь-якого гравця")
    st.markdown("---")
    
    # Вибір гравця
    selected_player = st.selectbox(
        "🔍 Оберіть гравця:",
        options=[''] + sorted(players_df['name'].tolist()),
        key='history_player',
        format_func=lambda x: "-- Оберіть гравця --" if x == '' else x
    )
    
    if selected_player and selected_player != '':
        # Отримуємо статистику
        player_stats = get_player_stats(db, selected_player)
        
        if player_stats:
            # Інфо картка
            col_info1, col_info2, col_info3 = st.columns(3)
            
            with col_info1:
                st.markdown(f"""
                <div class="stats-box">
                    <h3>🏆 Ранг ATP</h3>
                    <h1 style="font-size: 3rem; margin: 0.5rem 0;">{int(player_stats['rank'])}</h1>
                </div>
                """, unsafe_allow_html=True)
            
            with col_info2:
                st.markdown(f"""
                <div class="stats-box">
                    <h3>⭐ Рейтингові очки</h3>
                    <h1 style="font-size: 3rem; margin: 0.5rem 0;">{int(player_stats['rank_points'])}</h1>
                </div>
                """, unsafe_allow_html=True)
            
            with col_info3:
                recent = player_stats['recent_matches']
                wins = 0
                for _, match in recent.iterrows():
                    is_p1 = match['p1_name'] == selected_player
                    if is_p1:
                        wins += match['p1_won']
                    else:
                        wins += (1 - match['p1_won'])
                
                win_rate = (wins / len(recent) * 100) if len(recent) > 0 else 0
                
                st.markdown(f"""
                <div class="stats-box">
                    <h3>📈 Форма (10 матчів)</h3>
                    <h1 style="font-size: 3rem; margin: 0.5rem 0;">{win_rate:.0f}%</h1>
                    <p style="margin: 0;">{int(wins)}-{len(recent)-int(wins)}</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Останні 10 матчів
            st.markdown("### 🎾 Останні 10 матчів")
            
            last_matches = get_last_10_matches(db, selected_player)
            
            if not last_matches.empty:
                for idx, match in last_matches.iterrows():
                    result_color = "#d4edda" if "WIN" in match['Result'] else "#f8d7da"
                    
                    st.markdown(f"""
                    <div class="match-card" style="background-color: {result_color};">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <h4 style="margin: 0;">{match['Tournament']}</h4>
                                <p style="margin: 0.2rem 0; color: #666;">
                                    {match['Date']} | {match['Surface']}
                                </p>
                            </div>
                            <div style="text-align: right;">
                                <h3 style="margin: 0;">{match['Result']}</h3>
                                <p style="margin: 0.2rem 0;">vs {match['Opponent']}</p>
                                <p style="margin: 0; font-size: 0.9rem; color: #666;">{match['Score']}</p>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("📭 Немає даних про матчі цього гравця")
        else:
            st.error("❌ Не вдалося отримати дані про гравця")
