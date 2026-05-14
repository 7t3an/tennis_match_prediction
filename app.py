"""Tennis Match Prediction System.
A machine learning application for predicting ATP tennis match outcomes.
Built with XGBoost and Streamlit for interactive predictions.

Model Performance (Latest):
    - Metrics loaded from models/model_metrics.txt
    - No data leakage - all features available before match

Author: Vladyslav Romaniuk
"""

import os
from datetime import datetime
import subprocess
import sys

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.data import (
    load_database,
    get_unique_players,
    get_player_stats,
    get_last_10_matches
)
from src.features import calculate_features
from src.models import load_model, prepare_features_for_prediction, predict_match


# Page Configuration

st.set_page_config(
    page_title="Tennis Match Predictor",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# Custom CSS Styles

st.markdown("""
    <style>
    /* Color Palette */
    :root {
        --primary: #3b82f6;
        --primary-dark: #1d4ed8;
        --secondary: #22c55e;
        --danger: #ef4444;
        --warning: #f59e0b;
        --bg: #0b1220;
        --panel: #0a0f1e;
        --panel-alt: #0f172a;
        --border: #1f2937;
        --text: #e5e7eb;
        --text-muted: #94a3b8;
    }

    /* Global Styles */
    .main { 
        padding: 1.5rem 3rem; 
        background: var(--bg);
        color: var(--text);
    }
    body {
        background: var(--bg);
        color: var(--text);
    }
    
    /* Sidebar Styles */
    section[data-testid="stSidebar"] {
        background: var(--panel);
        border-right: 2px solid var(--border);
        box-shadow: 4px 0 12px rgba(0,0,0,0.45);
    }
    section[data-testid="stSidebar"] * { color: var(--text); }

    /* Button Styles */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
        color: white;
        font-size: 16px;
        font-weight: 600;
        padding: 0.875rem 1.5rem;
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #1e40af 0%, #1d4ed8 100%);
        box-shadow: 0 8px 12px -2px rgba(37, 99, 235, 0.3);
        transform: translateY(-1px);
    }
    
    /* Stats Card Styles */
    .stats-box {
        padding: 1.5rem;
        border-radius: 16px;
        background: var(--panel-alt);
        border: 2px solid var(--border);
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.35);
    }
    .stats-box h4 {
        color: var(--text-muted);
        font-size: 0.875rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    .stats-box p {
        color: var(--text);
        font-size: 1rem;
        font-weight: 600;
        margin: 0.25rem 0;
    }
    .stats-box strong {
        color: var(--text-muted);
        font-weight: 500;
    }
    
    /* Match History Card Styles */
    .match-card {
        padding: 1rem;
        border-radius: 12px;
        background: var(--panel-alt);
        border: 2px solid var(--border);
        margin-bottom: 0.75rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.35);
        transition: all 0.2s ease;
    }
    .match-card:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.45);
        transform: translateY(-2px);
    }
    .match-card h4 { color: var(--text); font-size: 1rem; font-weight: 600; margin: 0; }
    .match-card p { color: var(--text-muted); margin: 0.25rem 0; font-size: 0.9rem; }
    .match-card h3 { color: var(--text); font-size: 1.25rem; font-weight: 700; margin: 0; }
    
    /* Badge Styles */
    .win-badge { background: #064e3b; border: 2px solid #10b981; color: #a7f3d0; }
    .loss-badge { background: #3f0e12; border: 2px solid #ef4444; color: #fecaca; }
    .pill { 
        background: var(--border); 
        color: white; 
        padding: 0.25rem 0.75rem; 
        border-radius: 6px; 
        font-weight: 700; 
        font-size: 0.875rem; 
    }
    .pill.win { background: #10b981; }
    .pill.loss { background: #ef4444; }
    .badge { 
        padding: 0.25rem 0.5rem; 
        border-radius: 6px; 
        font-weight: 700; 
        font-size: 0.75rem; 
        margin-left: 0.5rem; 
    }
    .badge.ret, .badge.abd { background: #f59e0b; color: #0b1220; }
    .badge.wo, .badge.def { background: #ef4444; color: #0b1220; }

    /* Match Card State Modifiers */
    .match-card.win { border-left: 4px solid #10b981; background: rgba(34,197,94,0.08); }
    .match-card.loss { border-left: 4px solid #ef4444; background: rgba(239,68,68,0.08); }

    /* Match Card Layout */
    .match-card .top { display:flex; justify-content:space-between; align-items:flex-start; }
    .match-card .right { text-align:right; }
    .match-card .meta { margin:0; font-size:0.8rem; color: var(--text-muted); }
    .match-card .bottom { margin-top:0.75rem; padding-top:0.75rem; border-top:1px solid var(--border); }
    .match-card .opponent { margin:0; font-weight:600; font-size:0.9rem; color:var(--text); }
    .match-card .score { 
        margin:0.25rem 0 0 0; 
        color:var(--text); 
        font-size:0.95rem; 
        font-family: Menlo, Consolas, monospace; 
        letter-spacing: 0.02em; 
    }
    
    /* Section Headers */
    .section-header {
        color: var(--text);
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid var(--primary);
    }
    
    /* Selectbox Styles */
    .stSelectbox label {
        color: var(--text) !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }
    </style>
""", unsafe_allow_html=True)


# Data Loading

model, feature_cols, label_encoders = load_model()
db_result = load_database()

if db_result is None:
    st.error("Failed to load database. Please check data files in tml-data/")
    st.stop()

db, test_db = db_result

# Check if test_db is not None
if test_db is None or len(test_db) == 0:
    st.warning("No test data available")
    test_db = db  # Use all data if no test data available

model_available = all([
    model is not None,
    feature_cols is not None,
    label_encoders is not None
])

if not model_available:
    st.warning(
        "Prediction model not found. The 'Prediction' tab is temporarily unavailable. "
        "Run `python scripts/train_model.py` to train the model."
    )

players_df = get_unique_players(test_db)


# Sidebar Navigation

options = ["Player History"] + (["Prediction"] if model_available else [])
page = st.sidebar.radio(
    "Navigation",
    options=options,
    index=(1 if model_available else 0),
    help="Switch between pages using this menu"
)



# Prediction Page

if page == "Prediction" and model_available:
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0 1rem 0;">
        <h1 style="color: var(--text); font-size: 2.5rem; margin-bottom: 0.5rem;">
            🎾 Tennis Match Predictor
        </h1>
        <p style="color: var(--text-muted); font-size: 1.1rem;">
            Select two players to predict match outcome
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Player Selection
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("#### 🟢 Player 1")
        p1_name = st.selectbox(
            "Select player:",
            options=[''] + sorted(players_df['name'].tolist()),
            key='p1_name',
            format_func=lambda x: "-- Select player --" if x == '' else x,
            label_visibility="collapsed"
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
        st.markdown("#### 🔵 Player 2")
        p2_name = st.selectbox(
            "Select player:",
            options=[''] + sorted(players_df['name'].tolist()),
            key='p2_name',
            format_func=lambda x: "-- Select player --" if x == '' else x,
            label_visibility="collapsed"
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
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### ⚙️ Match Parameters")
    
    col3, col4 = st.columns(2, gap="large")
    
    with col3:
        surface = st.selectbox(
            "Court Surface",
            options=['Hard', 'Clay', 'Grass', 'Carpet'],
            index=0
        )
    
    with col4:
        tourney_level = st.selectbox(
            "Tournament Level",
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
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🎯 PREDICT MATCH", type="primary"):
        if not p1_name or not p2_name or p1_name == '' or p2_name == '':
            st.error("Please select both players!")
        elif p1_name == p2_name:
            st.error("Please select different players!")
        else:
            with st.spinner('Generating prediction...'):
                original_p1_name = p1_name
                original_p2_name = p2_name
                
                p1_orig_stats = get_player_stats(db, p1_name)
                p2_orig_stats = get_player_stats(db, p2_name)
                
                result = calculate_features(db, p1_name, p2_name, surface, tourney_level)
                
                if result is None or result[0] is None:
                    st.error("Failed to retrieve player data")
                else:
                    features_dict, swapped = result
                    
                    # Store original H2H (features may be swapped for model)
                    if swapped:
                        orig_h2h_p1_wins = features_dict['h2h_p2_wins']
                        orig_h2h_p2_wins = features_dict['h2h_p1_wins']
                        orig_h2h_p1_win_rate = features_dict['h2h_p2_win_rate']
                    else:
                        orig_h2h_p1_wins = features_dict['h2h_p1_wins']
                        orig_h2h_p2_wins = features_dict['h2h_p2_wins']
                        orig_h2h_p1_win_rate = features_dict['h2h_p1_win_rate']
                    
                    input_df = prepare_features_for_prediction(
                        features_dict, feature_cols, label_encoders
                    )
                    prob_p1_wins, prob_p2_wins = predict_match(model, input_df, swapped)
                    
                    st.markdown("---")
                    st.markdown("## 📊 Prediction Results")
                    
                    # Gauge Chart - centered
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=prob_p1_wins * 100,
                        number={'suffix': '%', 'font': {'size': 48, 'color': '#e5e7eb'}},
                        delta={'reference': 50, 'increasing': {'color': '#22c55e'}, 'decreasing': {'color': '#ef4444'}},
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={
                            'text': f"<b>{original_p1_name}</b><br><span style='font-size:14px;color:#94a3b8'>Win Probability</span>",
                            'font': {'size': 24, 'color': '#e5e7eb'}
                        },
                        gauge={
                            'axis': {
                                'range': [0, 100], 
                                'tickwidth': 2, 
                                'tickcolor': "#475569",
                                'tickfont': {'color': '#94a3b8', 'size': 12}
                            },
                            'bar': {'color': "#3b82f6", 'thickness': 0.7},
                            'bgcolor': "#1e293b",
                            'borderwidth': 3,
                            'bordercolor': "#334155",
                            'steps': [
                                {'range': [0, 35], 'color': 'rgba(239,68,68,0.3)'},
                                {'range': [35, 50], 'color': 'rgba(251,146,60,0.2)'},
                                {'range': [50, 65], 'color': 'rgba(250,204,21,0.2)'},
                                {'range': [65, 100], 'color': 'rgba(34,197,94,0.3)'}
                            ],
                            'threshold': {
                                'line': {'color': "#f8fafc", 'width': 4},
                                'thickness': 0.8,
                                'value': 50
                            }
                        }
                    ))
                    
                    fig.update_layout(
                        height=380,
                        margin=dict(l=40, r=40, t=100, b=20),
                        paper_bgcolor="rgba(0,0,0,0)",
                        font={'color': "#e5e7eb", 'family': "Inter, Arial, sans-serif"}
                    )
                    
                    # Center the gauge chart
                    col_gauge_l, col_gauge_c, col_gauge_r = st.columns([1, 2, 1])
                    with col_gauge_c:
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Results Cards
                    col_res1, col_res2 = st.columns(2, gap="large")
                    
                    original_p1_rank = (
                        p1_orig_stats['rank'] if p1_orig_stats 
                        else features_dict['p1_rank']
                    )
                    original_p1_points = (
                        p1_orig_stats['rank_points'] if p1_orig_stats 
                        else features_dict['p1_rank_points']
                    )
                    original_p2_rank = (
                        p2_orig_stats['rank'] if p2_orig_stats 
                        else features_dict['p2_rank']
                    )
                    original_p2_points = (
                        p2_orig_stats['rank_points'] if p2_orig_stats 
                        else features_dict['p2_rank_points']
                    )
                    
                    with col_res1:
                        win_color = "#10b981" if prob_p1_wins > 0.5 else "#64748b"
                        st.markdown(f"""
                        <div style="padding: 2rem; border-radius: 16px; 
                             background: var(--panel-alt);
                             border: 3px solid {win_color};
                             text-align: center;
                             box-shadow: 0 4px 6px rgba(0,0,0,0.35);">
                            <h2 style="color: var(--text); margin: 0;">{original_p1_name}</h2>
                            <h1 style="font-size: 3.5rem; margin: 1rem 0; color: {win_color};">
                                {prob_p1_wins:.1%}
                            </h1>
                            <p style="font-size: 1.1rem; color: var(--text-muted);">Win Probability</p>
                            <div style="margin-top: 1.5rem; padding-top: 1rem; 
                                        border-top: 2px solid var(--border);">
                                <p style="color: var(--text-muted); margin: 0.25rem 0;">
                                    <strong>Rank:</strong> #{int(original_p1_rank)}
                                </p>
                                <p style="color: var(--text-muted); margin: 0.25rem 0;">
                                    <strong>Points:</strong> {int(original_p1_points):,}
                                </p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_res2:
                        win_color = "#10b981" if prob_p2_wins > 0.5 else "#64748b"
                        st.markdown(f"""
                        <div style="padding: 2rem; border-radius: 16px; 
                             background: var(--panel-alt);
                             border: 3px solid {win_color};
                             text-align: center;
                             box-shadow: 0 4px 6px rgba(0,0,0,0.35);">
                            <h2 style="color: var(--text); margin: 0;">{original_p2_name}</h2>
                            <h1 style="font-size: 3.5rem; margin: 1rem 0; color: {win_color};">
                                {prob_p2_wins:.1%}
                            </h1>
                            <p style="font-size: 1.1rem; color: var(--text-muted);">Win Probability</p>
                            <div style="margin-top: 1.5rem; padding-top: 1rem; 
                                        border-top: 2px solid var(--border);">
                                <p style="color: var(--text-muted); margin: 0.25rem 0;">
                                    <strong>Rank:</strong> #{int(original_p2_rank)}
                                </p>
                                <p style="color: var(--text-muted); margin: 0.25rem 0;">
                                    <strong>Points:</strong> {int(original_p2_points):,}
                                </p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    favorite = (
                        original_p1_name if prob_p1_wins > prob_p2_wins 
                        else original_p2_name
                    )
                    favorite_prob = max(prob_p1_wins, prob_p2_wins)
                    
                    if favorite_prob > 0.7:
                        confidence_level = "Very Confident"
                    elif favorite_prob > 0.6:
                        confidence_level = "Confident"
                    elif favorite_prob > 0.55:
                        confidence_level = "Moderately Confident"
                    else:
                        confidence_level = "Uncertain"
                    
                    confidence_emoji = '🔥' if favorite_prob > 0.7 else '💪' if favorite_prob > 0.6 else '🤔' if favorite_prob > 0.55 else '⚖️'
                    st.markdown(f"""
                    <div style="padding: 2rem; border-radius: 16px; 
                         background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
                         color: white; text-align: center;
                         box-shadow: 0 8px 16px rgba(37, 99, 235, 0.2);">
                        <h2 style="margin: 0.5rem 0;">🏆 Model Prediction</h2>
                        <h1 style="font-size: 2.5rem; margin: 1rem 0;">{favorite}</h1>
                        <p style="font-size: 1.2rem; opacity: 0.95;">
                            {confidence_emoji} {confidence_level} - {favorite_prob:.1%}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("### 📈 Match Analysis")
                    
                    col_det1, col_det2, col_det3 = st.columns(3)
                    
                    with col_det1:
                        rank_diff = abs(int(original_p1_rank) - int(original_p2_rank))
                        better_player = (
                            original_p1_name if original_p1_rank < original_p2_rank 
                            else original_p2_name
                        )
                        st.markdown(f"""
                        <div style="padding: 1.5rem; background: var(--panel-alt); 
                                    border-radius: 12px; border: 2px solid var(--border); 
                                    text-align: center;">
                            <p style="color: var(--text-muted); font-size: 0.875rem; margin: 0;">
                                Rank Difference
                            </p>
                            <h2 style="color: var(--text); margin: 0.5rem 0;">{rank_diff}</h2>
                            <p style="color: var(--text-muted); font-size: 0.875rem; margin: 0;">
                                {better_player} higher ranked
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_det2:
                        h2h_info = (
                            f"{original_p1_name}: {orig_h2h_p1_win_rate:.0%}" 
                            if features_dict['h2h_total_matches'] > 0 
                            else "No data"
                        )
                        st.markdown(f"""
                        <div style="padding: 1.5rem; background: var(--panel-alt); 
                                    border-radius: 12px; border: 2px solid var(--border); 
                                    text-align: center;">
                            <p style="color: var(--text-muted); font-size: 0.875rem; margin: 0;">
                                Head-to-Head
                            </p>
                            <h2 style="color: var(--text); margin: 0.5rem 0;">
                                {features_dict['h2h_total_matches']}
                            </h2>
                            <p style="color: var(--text-muted); font-size: 0.875rem; margin: 0;">
                                {h2h_info}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_det3:
                        if favorite_prob > 0.7:
                            expected_accuracy = 73.68
                        elif favorite_prob > 0.6:
                            expected_accuracy = 62.51
                        elif favorite_prob > 0.55:
                            expected_accuracy = 58.55
                        else:
                            expected_accuracy = 52.23
                        
                        st.markdown(f"""
                        <div style="padding: 1.5rem; background: var(--panel-alt); 
                                    border-radius: 12px; border: 2px solid var(--border); 
                                    text-align: center;">
                            <p style="color: var(--text-muted); font-size: 0.875rem; margin: 0;">
                                Expected Accuracy
                            </p>
                            <h2 style="color: var(--text); margin: 0.5rem 0;">
                                {expected_accuracy:.1f}%
                            </h2>
                            <p style="color: var(--text-muted); font-size: 0.875rem; margin: 0;">
                                {confidence_level}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)


# Player History Page

elif page == "Player History":
    st.title("👤 Player Statistics")
    st.markdown("### 📋 View detailed player performance")
    st.markdown("---")
    
    selected_player = st.selectbox(
        "Select player:",
        options=[''] + sorted(players_df['name'].tolist()),
        key='history_player',
        format_func=lambda x: "-- Select player --" if x == '' else x
    )
    
    if selected_player and selected_player != '':
        with st.spinner("Loading player data..."):
            player_stats = get_player_stats(db, selected_player)
        
        if player_stats:
            col_info1, col_info2, col_info3 = st.columns(3, gap="medium")
            
            with col_info1:
                st.markdown(f"""
                <div style="padding: 1.5rem; background: var(--panel-alt); 
                            border-radius: 12px; border: 2px solid var(--primary); 
                            text-align: center; box-shadow: 0 2px 6px rgba(0,0,0,0.35);">
                    <p style="color: var(--text-muted); font-size: 0.875rem; 
                              font-weight: 600; margin: 0; text-transform: uppercase;">
                        ATP Rank
                    </p>
                    <h1 style="font-size: 2.5rem; margin: 0.5rem 0; color: var(--text);">
                        #{int(player_stats['rank'])}
                    </h1>
                </div>
                """, unsafe_allow_html=True)
            
            with col_info2:
                st.markdown(f"""
                <div style="padding: 1.5rem; background: var(--panel-alt); 
                            border-radius: 12px; border: 2px solid var(--secondary); 
                            text-align: center; box-shadow: 0 2px 6px rgba(0,0,0,0.35);">
                    <p style="color: var(--text-muted); font-size: 0.875rem; 
                              font-weight: 600; margin: 0; text-transform: uppercase;">
                        Ranking Points
                    </p>
                    <h1 style="font-size: 2.5rem; margin: 0.5rem 0; color: var(--text);">
                        {int(player_stats['rank_points']):,}
                    </h1>
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
                
                if win_rate >= 60:
                    form_color = "#22c55e"
                elif win_rate >= 40:
                    form_color = "#f59e0b"
                else:
                    form_color = "#ef4444"
                
                st.markdown(f"""
                <div style="padding: 1.5rem; background: var(--panel-alt); 
                            border-radius: 12px; border: 2px solid {form_color}; 
                            text-align: center; box-shadow: 0 2px 6px rgba(0,0,0,0.35);">
                    <p style="color: var(--text-muted); font-size: 0.875rem; 
                              font-weight: 600; margin: 0; text-transform: uppercase;">
                        Recent Form (L10)
                    </p>
                    <h1 style="font-size: 2.5rem; margin: 0.5rem 0; color: var(--text);">
                        {win_rate:.0f}%
                    </h1>
                    <p style="margin: 0; color: var(--text-muted); font-weight: 600;">
                        {int(wins)}W - {len(recent)-int(wins)}L
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 🏆 Recent Matches")
            
            with st.spinner("Loading match history..."):
                last_matches = get_last_10_matches(db, selected_player)
            
            if not last_matches.empty:
                try:
                    last_matches = last_matches.copy()
                    last_matches['__ts'] = pd.to_datetime(
                        last_matches['Date'], errors='coerce'
                    ).dt.normalize()
                    last_matches = last_matches.sort_values(
                        '__ts', ascending=False
                    ).drop(columns=['__ts'])
                except Exception:
                    pass

                for idx in range(len(last_matches)):
                    match = last_matches.iloc[idx]
                    tournament = str(match.get('Tournament', 'N/A') or 'N/A')
                    date = str(match.get('Date', 'N/A') or 'N/A')
                    surface = str(match.get('Surface', 'N/A') or 'N/A')
                    opponent = str(match.get('Opponent', 'N/A') or 'N/A')
                    result = str(match.get('Result', '') or '')
                    score = str(match.get('Score', 'N/A') or 'N/A')
                    status = str(match.get('Status', '') or '')

                    is_win = ("WIN" in result)
                    
                    # Build status badge HTML
                    status_badge = ""
                    if status and status.strip():
                        badge_class = {"RET": "ret", "ABD": "abd", "W/O": "wo", "DEF": "def"}.get(status, "")
                        status_badge = f'<span class="badge {badge_class}">{status}</span>'
                    
                    # Build status description
                    status_line = ""
                    if status and status.strip():
                        desc_map = {"RET": "Injury/Retirement", "ABD": "Match Abandoned", "W/O": "Walkover", "DEF": "Disqualification"}
                        desc = desc_map.get(status, status)
                        status_line = f'<p class="meta" style="margin-top:0.25rem;color:#f59e0b;">⚠️ {desc}</p>'
                    
                    card_class = "win" if is_win else "loss"
                    
                    # Build complete card HTML - using single quotes inside, double outside
                    html = f'''<div class="match-card {card_class}">
<div class="top">
<div style="flex:1;">
<h4>{tournament}</h4>
<p class="meta">{date} • {surface}</p>
</div>
<div class="right">
<span class="pill {card_class}">{result}</span>
{status_badge}
</div>
</div>
<div class="bottom">
<p class="opponent">vs {opponent}</p>
<p class="score">{score}</p>
{status_line}
</div>
</div>'''
                    
                    st.markdown(html, unsafe_allow_html=True)
            else:
                st.warning("No match data available for this player")
        else:
            st.error("Failed to retrieve player data")
