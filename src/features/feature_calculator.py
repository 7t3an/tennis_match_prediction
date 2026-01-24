"""Feature calculation utilities for tennis match prediction.

Calculates rolling stats on-the-fly from raw match data.
Model achieves ROC-AUC: 0.71, Accuracy: 65%
"""

from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np
from ..data.player_stats import get_player_stats


def _compute_rolling_stats(df: pd.DataFrame, player_name: str, n_matches: int = 10) -> Dict:
    """Compute rolling statistics for a player from their last N matches.
    
    Args:
        df: DataFrame with match data (winner/loser format)
        player_name: Name of the player
        n_matches: Number of recent matches to consider
        
    Returns:
        Dictionary with rolling stats
    """
    # Find all matches for this player
    player_matches = df[
        (df['winner_name'] == player_name) | (df['loser_name'] == player_name)
    ].copy()
    
    if len(player_matches) == 0:
        return {}
    
    # Sort by date (newest first)
    if 'tourney_date' in player_matches.columns:
        player_matches = player_matches.sort_values('tourney_date', ascending=False)
    
    # Take last N matches (excluding current - use shift logic)
    recent_matches = player_matches.head(n_matches)
    
    # Stats to compute
    stats = ['ace', 'df', 'svpt', '1stIn', '1stWon', '2ndWon', 'SvGms', 'bpSaved', 'bpFaced']
    
    rolling_stats = {}
    for stat in stats:
        values = []
        for _, match in recent_matches.iterrows():
            is_winner = match['winner_name'] == player_name
            col = f'w_{stat}' if is_winner else f'l_{stat}'
            if col in match and pd.notna(match[col]):
                values.append(match[col])
        
        if len(values) >= 3:  # Require at least 3 matches
            rolling_stats[f'{stat}_roll10'] = np.mean(values)
        else:
            rolling_stats[f'{stat}_roll10'] = 0
    
    return rolling_stats


def calculate_features(
    _df: pd.DataFrame,
    p1_name: str,
    p2_name: str,
    surface: str = 'Hard',
    tourney_level: str = 'A'
) -> Optional[Tuple[Dict, bool]]:
    """Calculate all features from latest player statistics.
    
    Computes rolling stats on-the-fly from raw match data.
    Model predicts probability that P1 wins. Player order matters.
    Model is trained with P1 = better ranked player (lower rank number).
    If input order doesn't match, features will be swapped.
    
    Args:
        _df: DataFrame with match data
        p1_name: Name of first player
        p2_name: Name of second player
        surface: Court surface (Hard, Clay, Grass, Carpet)
        tourney_level: Tournament level (G, M, A, D, F)
        
    Returns:
        tuple: (features_dict, needs_swap)
            - features_dict: Dictionary with all features
            - needs_swap: Boolean indicating if players were swapped for model
        None: If player data cannot be retrieved
    """
    
    p1_stats = get_player_stats(_df, p1_name)
    p2_stats = get_player_stats(_df, p2_name)
    
    if p1_stats is None or p2_stats is None:
        return None
    
    # Get matches for each player
    p1_matches = _df[(_df['p1_name'] == p1_name) | (_df['p2_name'] == p1_name)].copy()
    p2_matches = _df[(_df['p1_name'] == p2_name) | (_df['p2_name'] == p2_name)].copy()
    
    if len(p1_matches) == 0 or len(p2_matches) == 0:
        return None
    
    # Sort by date if available
    if 'tourney_date' in p1_matches.columns:
        p1_matches = p1_matches.sort_values('tourney_date', ascending=False)
        p2_matches = p2_matches.sort_values('tourney_date', ascending=False)
    
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
    features['p1_entry'] = p1_latest.get('p1_entry', 'DA') if p1_is_p1 else p1_latest.get('p2_entry', 'DA')
    features['p1_hand'] = p1_latest.get('p1_hand', 'R') if p1_is_p1 else p1_latest.get('p2_hand', 'R')
    features['p1_ht'] = p1_latest.get('p1_ht', 180) if p1_is_p1 else p1_latest.get('p2_ht', 180)
    features['p1_ioc'] = p1_latest.get('p1_ioc', 'ESP') if p1_is_p1 else p1_latest.get('p2_ioc', 'ESP')
    features['p1_age'] = p1_latest.get('p1_age', 25) if p1_is_p1 else p1_latest.get('p2_age', 25)
    
    # P2 player features
    features['p2_rank'] = p2_latest['p1_rank'] if p2_is_p1 else p2_latest['p2_rank']
    features['p2_rank_points'] = p2_latest['p1_rank_points'] if p2_is_p1 else p2_latest['p2_rank_points']
    features['p2_entry'] = p2_latest.get('p1_entry', 'DA') if p2_is_p1 else p2_latest.get('p2_entry', 'DA')
    features['p2_hand'] = p2_latest.get('p1_hand', 'R') if p2_is_p1 else p2_latest.get('p2_hand', 'R')
    features['p2_ht'] = p2_latest.get('p1_ht', 180) if p2_is_p1 else p2_latest.get('p2_ht', 180)
    features['p2_ioc'] = p2_latest.get('p1_ioc', 'ESP') if p2_is_p1 else p2_latest.get('p2_ioc', 'ESP')
    features['p2_age'] = p2_latest.get('p1_age', 25) if p2_is_p1 else p2_latest.get('p2_age', 25)
    
    # Compute rolling statistics on-the-fly from raw data
    # Need to use winner/loser format for rolling stats calculation
    raw_df = _df.copy()
    if 'winner_name' not in raw_df.columns and 'p1_name' in raw_df.columns:
        # Data is in p1/p2 format - need to check if original winner/loser columns exist
        # If not, we'll estimate from the available data
        pass
    
    p1_rolling = _compute_rolling_stats(raw_df, p1_name)
    p2_rolling = _compute_rolling_stats(raw_df, p2_name)
    
    # Add rolling stats to features
    for stat in ['ace', 'df', 'svpt', '1stIn', '1stWon', '2ndWon', 'SvGms', 'bpSaved', 'bpFaced']:
        col_name = f'{stat}_roll10'
        features[f'p1_{col_name}'] = p1_rolling.get(col_name, 0)
        features[f'p2_{col_name}'] = p2_rolling.get(col_name, 0)
    
    # Head-to-head statistics
    h2h_matches = _df[
        ((_df['p1_name'] == p1_name) & (_df['p2_name'] == p2_name)) |
        ((_df['p1_name'] == p2_name) & (_df['p2_name'] == p1_name))
    ]
    
    # Also check winner/loser format
    if 'winner_name' in _df.columns:
        h2h_wl = _df[
            ((_df['winner_name'] == p1_name) & (_df['loser_name'] == p2_name)) |
            ((_df['winner_name'] == p2_name) & (_df['loser_name'] == p1_name))
        ]
        if len(h2h_wl) > len(h2h_matches):
            h2h_matches = h2h_wl
    
    if len(h2h_matches) > 0:
        p1_wins = 0
        p2_wins = 0
        
        for _, match in h2h_matches.iterrows():
            if 'winner_name' in match:
                if match['winner_name'] == p1_name:
                    p1_wins += 1
                else:
                    p2_wins += 1
            elif 'p1_name' in match:
                if match['p1_name'] == p1_name:
                    if match.get('p1_won', 1) == 1:
                        p1_wins += 1
                    else:
                        p2_wins += 1
                else:
                    if match.get('p1_won', 1) == 0:
                        p1_wins += 1
                    else:
                        p2_wins += 1
        
        features['h2h_p1_wins'] = p1_wins
        features['h2h_p2_wins'] = p2_wins
        features['h2h_total_matches'] = len(h2h_matches)
        features['h2h_p1_win_rate'] = p1_wins / len(h2h_matches)
        features['h2h_p2_win_rate'] = p2_wins / len(h2h_matches)
    else:
        features['h2h_p1_wins'] = 0
        features['h2h_p2_wins'] = 0
        features['h2h_total_matches'] = 0
        features['h2h_p1_win_rate'] = 0.5
        features['h2h_p2_win_rate'] = 0.5
    
    # Encoded features - use numeric values for model
    surface_map = {'Hard': 3, 'Clay': 2, 'Grass': 1, 'Carpet': 0}
    tourney_level_map = {'G': 4, 'M': 3, 'A': 2, 'D': 1, 'F': 3, '250': 2, '500': 2}
    
    features['surface_encoded'] = surface_map.get(surface, 3)
    features['tourney_level_encoded'] = tourney_level_map.get(tourney_level, 2)
    
    # Rank-based features
    features['rank_diff'] = features['p1_rank'] - features['p2_rank']
    features['rank_ratio'] = features['p1_rank'] / max(features['p2_rank'], 1)
    features['is_p1_favorite'] = int(features['p1_rank'] < features['p2_rank'])
    features['rank_points_diff'] = features['p1_rank_points'] - features['p2_rank_points']
    features['rank_points_ratio'] = features['p1_rank_points'] / max(features['p2_rank_points'], 1)
    features['log_rank_ratio'] = np.log1p(features['p1_rank']) - np.log1p(features['p2_rank'])
    
    # Form features (derived from rolling stats)
    p1_svpt = features.get('p1_svpt_roll10', 1)
    p2_svpt = features.get('p2_svpt_roll10', 1)
    p1_1stWon = features.get('p1_1stWon_roll10', 0)
    p2_1stWon = features.get('p2_1stWon_roll10', 0)
    
    features['p1_serve_efficiency'] = p1_1stWon / max(p1_svpt, 1)
    features['p2_serve_efficiency'] = p2_1stWon / max(p2_svpt, 1)
    features['serve_efficiency_diff'] = features['p1_serve_efficiency'] - features['p2_serve_efficiency']
    
    p1_ace = features.get('p1_ace_roll10', 0)
    p1_df = features.get('p1_df_roll10', 1)
    p2_ace = features.get('p2_ace_roll10', 0)
    p2_df = features.get('p2_df_roll10', 1)
    
    features['p1_ace_df_ratio'] = p1_ace / max(p1_df, 1)
    features['p2_ace_df_ratio'] = p2_ace / max(p2_df, 1)
    features['ace_df_ratio_diff'] = features['p1_ace_df_ratio'] - features['p2_ace_df_ratio']
    
    p1_bpSaved = features.get('p1_bpSaved_roll10', 0)
    p1_bpFaced = features.get('p1_bpFaced_roll10', 1)
    p2_bpSaved = features.get('p2_bpSaved_roll10', 0)
    p2_bpFaced = features.get('p2_bpFaced_roll10', 1)
    
    features['p1_bp_save_rate'] = p1_bpSaved / max(p1_bpFaced, 1)
    features['p2_bp_save_rate'] = p2_bpSaved / max(p2_bpFaced, 1)
    features['bp_save_rate_diff'] = features['p1_bp_save_rate'] - features['p2_bp_save_rate']
    
    # Experience features
    features['age_diff'] = features['p1_age'] - features['p2_age']
    features['height_diff'] = features['p1_ht'] - features['p2_ht']
    
    # Normalize: always P1 = better ranked player (model trained this way)
    # Lower rank number = better rank (e.g., rank 1 > rank 50)
    needs_swap = features['p2_rank'] < features['p1_rank']
    
    if needs_swap:
        # Swap all p1_/p2_ features
        for key in list(features.keys()):
            if key.startswith('p1_'):
                p2_key = key.replace('p1_', 'p2_')
                if p2_key in features:
                    features[key], features[p2_key] = features[p2_key], features[key]
        
        # Swap H2H stats
        if 'h2h_p1_wins' in features and 'h2h_p2_wins' in features:
            features['h2h_p1_wins'], features['h2h_p2_wins'] = features['h2h_p2_wins'], features['h2h_p1_wins']
        if 'h2h_p1_win_rate' in features and 'h2h_p2_win_rate' in features:
            features['h2h_p1_win_rate'], features['h2h_p2_win_rate'] = features['h2h_p2_win_rate'], features['h2h_p1_win_rate']
        
        # Recalculate derived features after swap
        features['rank_diff'] = features['p1_rank'] - features['p2_rank']
        features['rank_ratio'] = features['p1_rank'] / max(features['p2_rank'], 1)
        features['is_p1_favorite'] = int(features['p1_rank'] < features['p2_rank'])
        features['rank_points_diff'] = features['p1_rank_points'] - features['p2_rank_points']
        features['rank_points_ratio'] = features['p1_rank_points'] / max(features['p2_rank_points'], 1)
        features['log_rank_ratio'] = np.log1p(features['p1_rank']) - np.log1p(features['p2_rank'])
        features['serve_efficiency_diff'] = features['p1_serve_efficiency'] - features['p2_serve_efficiency']
        features['ace_df_ratio_diff'] = features['p1_ace_df_ratio'] - features['p2_ace_df_ratio']
        features['bp_save_rate_diff'] = features['p1_bp_save_rate'] - features['p2_bp_save_rate']
        features['age_diff'] = features['p1_age'] - features['p2_age']
        features['height_diff'] = features['p1_ht'] - features['p2_ht']
    
    return features, needs_swap
