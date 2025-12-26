"""Database loading and player extraction utilities."""

from typing import Optional, Tuple
import pandas as pd
import streamlit as st
from pathlib import Path


@st.cache_data
def load_database() -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Load raw match data for Streamlit app.
    
    Uses original data from data/tml/ which has player names.
    
    Returns:
        tuple: (full_df, test_df) - Combined dataset and test dataset (2025)
    """
    try:
        data_dir = Path('data/tml')
        
        # Load years 2012-2025 for full history
        all_dfs = []
        for year in range(2012, 2026):
            file_path = data_dir / f'{year}.csv'
            if file_path.exists():
                df = pd.read_csv(file_path)
                df['data_year'] = year
                all_dfs.append(df)
        
        if not all_dfs:
            st.error("No data files found in data/tml/")
            return None, None
        
        full_df = pd.concat(all_dfs, ignore_index=True)
        
        # Convert to P1/P2 format with names
        full_df = _convert_to_p1_p2_with_names(full_df)
        
        # Split test (2025)
        test_df = full_df[full_df['data_year'] == 2025].copy()
        
        return full_df, test_df
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None


def _convert_to_p1_p2_with_names(df: pd.DataFrame) -> pd.DataFrame:
    """Convert winner/loser format to P1/P2 format preserving names."""
    import numpy as np
    
    result_rows = []
    np.random.seed(42)
    
    for idx, row in df.iterrows():
        new_row = {
            'tourney_id': row.get('tourney_id', ''),
            'tourney_name': row.get('tourney_name', ''),
            'tourney_date': row.get('tourney_date', ''),
            'surface': row.get('surface', 'Hard'),
            'tourney_level': row.get('tourney_level', 'A'),
            'data_year': row.get('data_year', 2025),
        }
        
        # Randomly assign P1/P2
        if np.random.random() < 0.5:
            # P1 = Winner
            new_row['p1_name'] = row.get('winner_name', '')
            new_row['p1_rank'] = row.get('winner_rank', 999)
            new_row['p1_rank_points'] = row.get('winner_rank_points', 0)
            new_row['p1_hand'] = row.get('winner_hand', 'R')
            new_row['p1_ht'] = row.get('winner_ht', 180)
            new_row['p1_ioc'] = row.get('winner_ioc', 'UNK')
            new_row['p1_age'] = row.get('winner_age', 25)
            new_row['p1_entry'] = row.get('winner_entry', '')
            new_row['p1_seed'] = row.get('winner_seed')
            
            new_row['p2_name'] = row.get('loser_name', '')
            new_row['p2_rank'] = row.get('loser_rank', 999)
            new_row['p2_rank_points'] = row.get('loser_rank_points', 0)
            new_row['p2_hand'] = row.get('loser_hand', 'R')
            new_row['p2_ht'] = row.get('loser_ht', 180)
            new_row['p2_ioc'] = row.get('loser_ioc', 'UNK')
            new_row['p2_age'] = row.get('loser_age', 25)
            new_row['p2_entry'] = row.get('loser_entry', '')
            new_row['p2_seed'] = row.get('loser_seed')
            
            new_row['p1_won'] = 1
        else:
            # P1 = Loser
            new_row['p1_name'] = row.get('loser_name', '')
            new_row['p1_rank'] = row.get('loser_rank', 999)
            new_row['p1_rank_points'] = row.get('loser_rank_points', 0)
            new_row['p1_hand'] = row.get('loser_hand', 'R')
            new_row['p1_ht'] = row.get('loser_ht', 180)
            new_row['p1_ioc'] = row.get('loser_ioc', 'UNK')
            new_row['p1_age'] = row.get('loser_age', 25)
            new_row['p1_entry'] = row.get('loser_entry', '')
            new_row['p1_seed'] = row.get('loser_seed')
            
            new_row['p2_name'] = row.get('winner_name', '')
            new_row['p2_rank'] = row.get('winner_rank', 999)
            new_row['p2_rank_points'] = row.get('winner_rank_points', 0)
            new_row['p2_hand'] = row.get('winner_hand', 'R')
            new_row['p2_ht'] = row.get('winner_ht', 180)
            new_row['p2_ioc'] = row.get('winner_ioc', 'UNK')
            new_row['p2_age'] = row.get('winner_age', 25)
            new_row['p2_entry'] = row.get('winner_entry', '')
            new_row['p2_seed'] = row.get('winner_seed')
            
            new_row['p1_won'] = 0
        
        result_rows.append(new_row)
    
    return pd.DataFrame(result_rows)


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
