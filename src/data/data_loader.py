"""Database loading and player extraction utilities."""

from typing import Optional, Tuple
import pandas as pd
import streamlit as st


@st.cache_data
def load_database() -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Load combined train and test datasets.
    
    Returns:
        tuple: (full_df, test_df) - Combined dataset and test dataset
    """
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
    """Extract unique players with latest statistics.
    
    Args:
        _df: DataFrame with player data
        
    Returns:
        DataFrame: Unique players sorted by rank
    """
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
