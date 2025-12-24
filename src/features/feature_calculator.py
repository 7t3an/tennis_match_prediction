"""Feature calculation utilities for tennis match prediction."""

from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np
from ..data.player_stats import get_player_stats


def calculate_features(
    _df: pd.DataFrame,
    p1_name: str,
    p2_name: str,
    surface: str = 'Hard',
    tourney_level: str = 'A'
) -> Optional[Tuple[Dict, bool]]:
    """Calculate all 48 features from latest player statistics.
    
    Note: Model predicts probability that P1 wins. Player order matters.
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
            - features_dict: Dictionary with 48 features
            - needs_swap: Boolean indicating if players were swapped for model
        None: If player data cannot be retrieved
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
        """Categorize seed into tiers."""
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
    
    return features, needs_swap
