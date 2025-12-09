"""Tennis Match Prediction System

XGBoost-based prediction model with symmetric architecture.
Accuracy: 65.3%, ROC-AUC: 0.64
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import plotly.graph_objects as go
from datetime import datetime


st.set_page_config(
    page_title="Tennis Match Predictor",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    .main { padding: 1rem 2rem; }
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

@st.cache_resource
def load_model():
    """Load XGBoost model, feature columns and label encoders."""
    try:
        model_path = Path('models/xgboost_calibrated_model.pkl')
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        features_path = Path('models/feature_columns.txt')
        with open(features_path, 'r') as f:
            feature_cols = [line.strip() for line in f.readlines()]
        
        encoders_path = Path('models/label_encoders.pkl')
        with open(encoders_path, 'rb') as f:
            label_encoders = pickle.load(f)
        
        return model, feature_cols, label_encoders
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

@st.cache_data
def load_database():
    """Load combined train and test datasets."""
    try:
        test_df = pd.read_csv('data/processed/test_features.csv')
        train_df = pd.read_csv('data/processed/train_features.csv')
        full_df = pd.concat([train_df, test_df], ignore_index=True)
        
        return full_df, test_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

@st.cache_data
def get_unique_players(_df):
    """Extract unique players with latest statistics."""
    _df_sorted = _df.sort_index(ascending=False)
    
    p1_players = _df_sorted[['p1_name', 'p1_rank', 'p1_rank_points']].rename(
        columns={'p1_name': 'name', 'p1_rank': 'rank', 'p1_rank_points': 'points'}
    )
    
    p2_players = _df_sorted[['p2_name', 'p2_rank', 'p2_rank_points']].rename(
        columns={'p2_name': 'name', 'p2_rank': 'rank', 'p2_rank_points': 'points'}
    )
    
    all_players = pd.concat([p1_players, p2_players])
    latest_players = all_players.drop_duplicates('name', keep='first').reset_index(drop=True)
    latest_players = latest_players.sort_values('rank').reset_index(drop=True)
    
    return latest_players

def get_player_stats(_df, player_name):
    """Get player's latest statistics from most recent match."""
    player_matches = _df[
        (_df['p1_name'] == player_name) | (_df['p2_name'] == player_name)
    ].copy()
    
    if len(player_matches) == 0:
        return None
    
    player_matches = player_matches.sort_index(ascending=False)
    latest_match = player_matches.iloc[0]
    is_p1 = latest_match['p1_name'] == player_name
    
    stats = {
        'name': player_name,
        'rank': latest_match['p1_rank'] if is_p1 else latest_match['p2_rank'],
        'rank_points': latest_match['p1_rank_points'] if is_p1 else latest_match['p2_rank_points'],
        'recent_matches': player_matches.head(10),
    }
    
    rolling_cols = [col for col in player_matches.columns if '_roll10' in col]
    if rolling_cols:
        prefix = 'p1_' if is_p1 else 'p2_'
        for col in rolling_cols:
            if col.startswith(prefix):
                feature_name = col
                stats[feature_name] = latest_match[col]
    
    return stats

def get_last_10_matches(_df, player_name):
    """Get player's last 10 matches for display."""
    player_matches = _df[
        (_df['p1_name'] == player_name) | (_df['p2_name'] == player_name)
    ].copy()
    
    if len(player_matches) == 0:
        return pd.DataFrame()
    
    if 'tourney_date' in player_matches.columns:
        player_matches = player_matches.sort_values('tourney_date', ascending=False)
    
    recent_matches = player_matches.head(10).copy()
    
    formatted_matches = []
    for _, match in recent_matches.iterrows():
        is_p1 = match['p1_name'] == player_name
        opponent = match['p2_name'] if is_p1 else match['p1_name']
        
        if is_p1:
            won = match['p1_won'] == 1
        else:
            won = match['p1_won'] == 0
        
        formatted_matches.append({
            'Date': match.get('tourney_date', 'N/A'),
            'Tournament': match.get('tourney_name', 'N/A'),
            'Surface': match.get('surface', 'N/A'),
            'Opponent': opponent,
            'Result': 'WIN' if won else 'LOSS',
            'Score': match.get('score', 'N/A'),
        })
    
    return pd.DataFrame(formatted_matches)

def calculate_features(_df, p1_name, p2_name, surface='Hard', tourney_level='A'):
    """Calculate all 53 features from latest player statistics.
    
    Note: Model predicts probability that P1 wins. Player order matters.
    
    Returns: features_dict (dict with 53 features)
    """
    
    p1_stats = get_player_stats(_df, p1_name)
    p2_stats = get_player_stats(_df, p2_name)
    
    if p1_stats is None or p2_stats is None:
        return None
    
    p1_matches = _df[(_df['p1_name'] == p1_name) | (_df['p2_name'] == p1_name)].sort_index(ascending=False)
    p2_matches = _df[(_df['p1_name'] == p2_name) | (_df['p2_name'] == p2_name)].sort_index(ascending=False)
    
    p1_latest = p1_matches.iloc[0]
    p2_latest = p2_matches.iloc[0]
    
    p1_is_p1 = p1_latest['p1_name'] == p1_name
    p2_is_p1 = p2_latest['p1_name'] == p2_name
    
    features = {}
    
    # Surface and tournament metadata
    features['surface'] = surface
    features['tourney_level'] = tourney_level
    features['draw_size'] = 128
    features['indoor'] = 0
    
    # P1 player features
    features['p1_rank'] = p1_latest['p1_rank'] if p1_is_p1 else p1_latest['p2_rank']
    features['p1_rank_points'] = p1_latest['p1_rank_points'] if p1_is_p1 else p1_latest['p2_rank_points']
    features['p1_seed'] = p1_latest.get('p1_seed', np.nan) if p1_is_p1 else p1_latest.get('p2_seed', np.nan)
    features['p1_entry'] = p1_latest.get('p1_entry', 'DA') if p1_is_p1 else p1_latest.get('p2_entry', 'DA')
    features['p1_hand'] = p1_latest.get('p1_hand', 'R') if p1_is_p1 else p1_latest.get('p2_hand', 'R')
    features['p1_ht'] = p1_latest.get('p1_ht', 180) if p1_is_p1 else p1_latest.get('p2_ht', 180)
    features['p1_ioc'] = p1_latest.get('p1_ioc', 'ESP') if p1_is_p1 else p1_latest.get('p2_ioc', 'ESP')
    features['p1_age'] = p1_latest.get('p1_age', 25) if p1_is_p1 else p1_latest.get('p2_age', 25)
    
    # P2 player features
    features['p2_rank'] = p2_latest['p1_rank'] if p2_is_p1 else p2_latest['p2_rank']
    features['p2_rank_points'] = p2_latest['p1_rank_points'] if p2_is_p1 else p2_latest['p2_rank_points']
    features['p2_seed'] = p2_latest.get('p1_seed', np.nan) if p2_is_p1 else p2_latest.get('p2_seed', np.nan)
    features['p2_entry'] = p2_latest.get('p1_entry', 'DA') if p2_is_p1 else p2_latest.get('p2_entry', 'DA')
    features['p2_hand'] = p2_latest.get('p1_hand', 'R') if p2_is_p1 else p2_latest.get('p2_hand', 'R')
    features['p2_ht'] = p2_latest.get('p1_ht', 180) if p2_is_p1 else p2_latest.get('p2_ht', 180)
    features['p2_ioc'] = p2_latest.get('p1_ioc', 'ESP') if p2_is_p1 else p2_latest.get('p2_ioc', 'ESP')
    features['p2_age'] = p2_latest.get('p1_age', 25) if p2_is_p1 else p2_latest.get('p2_age', 25)
    
    # Seed features
    features['is_p1_seeded'] = not pd.isna(features['p1_seed']) and features['p1_seed'] > 0
    features['is_p2_seeded'] = not pd.isna(features['p2_seed']) and features['p2_seed'] > 0
    
    p1_seed_val = features['p1_seed'] if features['is_p1_seeded'] else 999
    p2_seed_val = features['p2_seed'] if features['is_p2_seeded'] else 999
    features['seed_diff'] = p1_seed_val - p2_seed_val
    
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
    
    # Rolling statistics (last 10 matches)
    for stat in ['ace', 'df', 'svpt', '1stIn', '1stWon', '2ndWon', 'SvGms', 'bpSaved', 'bpFaced']:
        col_name = f'p1_{stat}_roll10'
        if p1_is_p1:
            features[col_name] = p1_latest.get(col_name, 0)
        else:
            col_name_p2 = f'p2_{stat}_roll10'
            features[col_name] = p1_latest.get(col_name_p2, 0)
    
    for stat in ['ace', 'df', 'svpt', '1stIn', '1stWon', '2ndWon', 'SvGms', 'bpSaved', 'bpFaced']:
        col_name = f'p2_{stat}_roll10'
        if p2_is_p1:
            col_name_p1 = f'p1_{stat}_roll10'
            features[col_name] = p2_latest.get(col_name_p1, 0)
        else:
            features[col_name] = p2_latest.get(col_name, 0)
    
    # Head-to-head statistics
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
    
    # Encoded features (will be processed by label encoders)
    features['tourney_level_encoded'] = features['tourney_level']
    features['surface_encoded'] = features['surface']
    
    # Normalize: always P1 = better ranked player (model trained this way)
    needs_swap = features['p2_rank'] < features['p1_rank']
    
    if needs_swap:
        for key in list(features.keys()):
            if key.startswith('p1_'):
                p2_key = key.replace('p1_', 'p2_')
                if p2_key in features:
                    features[key], features[p2_key] = features[p2_key], features[key]
        
        if 'h2h_p1_wins' in features and 'h2h_p2_wins' in features:
            features['h2h_p1_wins'], features['h2h_p2_wins'] = features['h2h_p2_wins'], features['h2h_p1_wins']
    
    return features, needs_swap

# Main interface
model, feature_cols, label_encoders = load_model()
db_result = load_database()

if db_result is None:
    st.error("Failed to load database")
    st.stop()

db, test_db = db_result

if model is None or db is None or feature_cols is None or label_encoders is None:
    st.error("Failed to load model or database")
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
        st.markdown("### Player 1")
        p1_name = st.selectbox(
            "Select player:",
            options=[''] + sorted(players_df['name'].tolist()),
            key='p1_name',
            format_func=lambda x: "-- Select player --" if x == '' else x
        )
        
        if p1_name and p1_name != '':
            p1_stats = get_player_stats(db, p1_name)
            if p1_stats:
                st.markdown(f"""
                <div class="stats-box">
                    <h4>Current Stats</h4>
                    <p><strong>ATP Rank:</strong> {int(p1_stats['rank'])}</p>
                    <p><strong>Ranking Points:</strong> {int(p1_stats['rank_points'])}</p>
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Player 2")
        p2_name = st.selectbox(
            "Select player:",
            options=[''] + sorted(players_df['name'].tolist()),
            key='p2_name',
            format_func=lambda x: "-- Select player --" if x == '' else x
        )
        
        if p2_name and p2_name != '':
            p2_stats = get_player_stats(db, p2_name)
            if p2_stats:
                st.markdown(f"""
                <div class="stats-box">
                    <h4>Current Stats</h4>
                    <p><strong>ATP Rank:</strong> {int(p2_stats['rank'])}</p>
                    <p><strong>Ranking Points:</strong> {int(p2_stats['rank_points'])}</p>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### Match Parameters (Optional)")
    
    col3, col4 = st.columns(2)
    
    with col3:
        surface = st.selectbox(
            "Surface:",
            options=['Hard', 'Clay', 'Grass', 'Carpet'],
            index=0
        )
    
    with col4:
        tourney_level = st.selectbox(
            "Tournament Level:",
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
    
    if st.button("MAKE PREDICTION", type="primary"):
        if not p1_name or not p2_name or p1_name == '' or p2_name == '':
            st.error("Please select both players!")
        elif p1_name == p2_name:
            st.error("Please select different players!")
        else:
            with st.spinner('Loading latest data and generating prediction...'):
                original_p1_name = p1_name
                original_p2_name = p2_name
                
                result = calculate_features(db, p1_name, p2_name, surface, tourney_level)
                
                if result is None or result[0] is None:
                    st.error("Failed to retrieve player data")
                else:
                    features_dict, swapped = result
                    
                    if swapped:
                        st.info(f"Model normalized: comparing {p2_name} vs {p1_name}")
                    
                    with st.expander("Debug: Model Input Data", expanded=False):
                        st.write(f"**Swapped: {swapped}**")
                        st.write("**Base Statistics (as seen by model):**")
                        col_d1, col_d2 = st.columns(2)
                        with col_d1:
                            st.write(f"**P1: {p1_name}**")
                            st.write(f"- Rank: {int(features_dict['p1_rank'])}")
                            st.write(f"- Points: {int(features_dict['p1_rank_points'])}")
                            st.write(f"- Age: {features_dict.get('p1_age', 'N/A')}")
                            st.write(f"- Hand: {features_dict.get('p1_hand', 'N/A')}")
                        with col_d2:
                            st.write(f"**{p2_name}:**")
                            st.write(f"- Rank: {int(features_dict['p2_rank'])}")
                            st.write(f"- Points: {int(features_dict['p2_rank_points'])}")
                            st.write(f"- Age: {features_dict.get('p2_age', 'N/A')}")
                            st.write(f"- Hand: {features_dict.get('p2_hand', 'N/A')}")
                        
                        st.write(f"\n**Rank Info:**")
                        st.write(f"- P1 rank: **{int(features_dict['p1_rank'])}**")
                        st.write(f"- P2 rank: **{int(features_dict['p2_rank'])}**")
                        st.write(f"- Better player: **{'P1' if features_dict['p1_rank'] < features_dict['p2_rank'] else 'P2' if features_dict['p2_rank'] < features_dict['p1_rank'] else 'Equal'}**")
                        
                        st.write(f"\n**H2H:**")
                        st.write(f"- Total matches: {features_dict['h2h_total_matches']}")
                        if features_dict['h2h_total_matches'] > 0:
                            p1_display_name = original_p2_name if swapped else original_p1_name
                            p2_display_name = original_p1_name if swapped else original_p2_name
                            st.write(f"- {p1_display_name}: {features_dict['h2h_p1_wins']} wins ({features_dict['h2h_p1_win_rate']:.0%})")
                            st.write(f"- {p2_display_name}: {features_dict['h2h_p2_wins']} wins")
                        else:
                            st.write("- No previous meetings")
                    
                    input_df = pd.DataFrame([features_dict])
                    
                    # Add missing features with defaults
                    for col in feature_cols:
                        if col not in input_df.columns:
                            if 'hand' in col:
                                input_df[col] = 'R'
                            elif 'entry' in col:
                                input_df[col] = 'DA'
                            elif 'ioc' in col:
                                input_df[col] = 'ESP'
                            else:
                                input_df[col] = 0
                    
                    # Encode categorical features using LabelEncoders
                    categorical_features = input_df.select_dtypes(include=['object']).columns.tolist()
                    
                    if label_encoders is not None:
                        for col in categorical_features:
                            if col in feature_cols:
                                encoder_col = col.replace('_encoded', '') if '_encoded' in col else col
                                
                                if encoder_col in label_encoders:
                                    encoder = label_encoders[encoder_col]
                                    input_df[col] = input_df[col].fillna('MISSING').astype(str)
                                    input_df[col] = input_df[col].apply(
                                        lambda x: x if x in encoder.classes_ else 'MISSING'
                                    )
                                    input_df[col] = encoder.transform(input_df[col])
                    
                    input_df = input_df.fillna(0)
                    input_df = input_df[feature_cols]
                    
                    st.write(f"**DEBUG: ALL {len(feature_cols)} FEATURES:**")
                    st.dataframe(input_df.T, use_container_width=True)
                    
                    prob_p1_wins = model.predict_proba(input_df)[0, 1]
                    prob_p2_wins = 1 - prob_p1_wins
                    
                    # Invert probabilities if players were swapped
                    if swapped:
                        prob_p1_wins, prob_p2_wins = prob_p2_wins, prob_p1_wins
                    
                    st.markdown("---")
                    st.markdown("## Prediction Results")
                    
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
                    
                    # Results
                    col_res1, col_res2 = st.columns(2)
                    
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
                            <h2>{original_p1_name}</h2>
                            <h1 style="font-size: 3rem; margin: 1rem 0;">{prob_p1_wins:.1%}</h1>
                            <p style="font-size: 1.2rem;">Win Probability</p>
                            <p style="font-size: 0.9rem; margin-top: 1rem;">
                                Rank: {int(original_p1_rank)} | Points: {int(original_p1_points)}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_res2:
                        st.markdown(f"""
                        <div style="padding: 2rem; border-radius: 15px; 
                             background: linear-gradient(135deg, {'#4CAF50' if prob_p2_wins > 0.5 else '#9E9E9E'}, 
                                                                 {'#45a049' if prob_p2_wins > 0.5 else '#757575'});
                             color: white; text-align: center;">
                            <h2>{original_p2_name}</h2>
                            <h1 style="font-size: 3rem; margin: 1rem 0;">{prob_p2_wins:.1%}</h1>
                            <p style="font-size: 1.2rem;">Win Probability</p>
                            <p style="font-size: 0.9rem; margin-top: 1rem;">
                                Rank: {int(original_p2_rank)} | Points: {int(original_p2_points)}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    
                    st.markdown("---")
                    
                    favorite = original_p1_name if prob_p1_wins > prob_p2_wins else original_p2_name
                    favorite_prob = max(prob_p1_wins, prob_p2_wins)
                    
                    confidence_level = "very confident" if favorite_prob > 0.7 else \
                                     "confident" if favorite_prob > 0.6 else \
                                     "moderately confident" if favorite_prob > 0.55 else \
                                     "uncertain"
                    
                    st.markdown(f"""
                    <div style="padding: 2rem; border-radius: 15px; 
                         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                         color: white; text-align: center;">
                        <h2>Model Prediction</h2>
                        <h1 style="font-size: 2.5rem; margin: 1rem 0;">
                            {favorite}
                        </h1>
                        <p style="font-size: 1.3rem;">
                            Model is {confidence_level}<br/>
                            with probability <strong>{favorite_prob:.1%}</strong>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    st.markdown("### Analysis Details")
                    
                    col_det1, col_det2, col_det3 = st.columns(3)
                    
                    with col_det1:
                        rank_diff = abs(features_dict['p1_rank'] - features_dict['p2_rank'])
                        st.metric(
                            "Rank Difference",
                            f"{rank_diff}",
                            f"{'P1' if features_dict['p1_rank'] < features_dict['p2_rank'] else 'P2'} higher"
                        )
                    
                    with col_det2:
                        st.metric(
                            "H2H Matches",
                            f"{features_dict['h2h_total_matches']}",
                            f"P1: {features_dict['h2h_p1_win_rate']:.0%}" if features_dict['h2h_total_matches'] > 0 else "No data"
                        )
                    
                    with col_det3:
                        expected_accuracy = 73.68 if favorite_prob > 0.7 else \
                                          62.51 if favorite_prob > 0.6 else \
                                          58.55 if favorite_prob > 0.55 else \
                                          52.23
                        
                        st.metric(
                            "Expected Accuracy",
                            f"{expected_accuracy:.1f}%",
                            f"{confidence_level}"
                        )

# Player History Tab
with tab2:
    st.title("Player Match History")
    st.markdown("### View recent matches for any player")
    st.markdown("---")
    
    selected_player = st.selectbox(
        "Select player:",
        options=[''] + sorted(players_df['name'].tolist()),
        key='history_player',
        format_func=lambda x: "-- Select player --" if x == '' else x
    )
    
    if selected_player and selected_player != '':
        player_stats = get_player_stats(db, selected_player)
        
        if player_stats:
            col_info1, col_info2, col_info3 = st.columns(3)
            
            with col_info1:
                st.markdown(f"""
                <div class="stats-box">
                    <h3>ATP Rank</h3>
                    <h1 style="font-size: 3rem; margin: 0.5rem 0;">{int(player_stats['rank'])}</h1>
                </div>
                """, unsafe_allow_html=True)
            
            with col_info2:
                st.markdown(f"""
                <div class="stats-box">
                    <h3>Ranking Points</h3>
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
                    <h3>Form (10 matches)</h3>
                    <h1 style="font-size: 3rem; margin: 0.5rem 0;">{win_rate:.0f}%</h1>
                    <p style="margin: 0;">{int(wins)}-{len(recent)-int(wins)}</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            st.markdown("### Last 10 Matches")
            
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
                st.warning("No match data available for this player")
        else:
            st.error("Failed to retrieve player data")
