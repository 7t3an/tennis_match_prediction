"""
Data Processor for Tennis Match Prediction Pipeline.

Handles loading, cleaning, and preprocessing of tennis match data.
NO data leakage - all operations preserve temporal causality.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Process raw tennis match data for ML pipeline.
    
    Key principles:
    1. NO DATA LEAKAGE - no future information used
    2. Temporal split - train on past, test on future
    3. Consistent handling of missing values
    """
    
    def __init__(self, data_path: str = 'data/tml'):
        self.data_path = Path(data_path)
        self.df_raw = None
        self.df_processed = None
        
    def load_data(self, start_year: int = 2012, end_year: int = 2025) -> pd.DataFrame:
        """
        Load tennis match data from yearly CSV files.
        
        Args:
            start_year: First year to load
            end_year: Last year to load
            
        Returns:
            DataFrame with all matches
        """
        logger.info(f"Loading data from {start_year} to {end_year}")
        
        all_matches = []
        
        for year in range(start_year, end_year + 1):
            file_path = self.data_path / f'{year}.csv'
            
            if file_path.exists():
                df_year = pd.read_csv(file_path, low_memory=False)
                df_year['data_year'] = year
                all_matches.append(df_year)
                logger.info(f"  {year}: {len(df_year):,} matches")
            else:
                logger.warning(f"  {year}: file not found")
                
        if not all_matches:
            raise ValueError("No data files found!")
            
        self.df_raw = pd.concat(all_matches, ignore_index=True)
        logger.info(f"Total: {len(self.df_raw):,} matches loaded")
        
        return self.df_raw
    
    def clean_data(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Clean and preprocess match data.
        
        Handles:
        - Date parsing and sorting
        - Missing value imputation
        - Data type corrections
        
        Args:
            df: DataFrame to clean (uses self.df_raw if None)
            
        Returns:
            Cleaned DataFrame
        """
        if df is None:
            df = self.df_raw.copy()
        else:
            df = df.copy()
            
        logger.info("Cleaning data...")
        initial_size = len(df)
        
        # 1. Parse and sort by date
        df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d', errors='coerce')
        df = df.sort_values('tourney_date').reset_index(drop=True)
        
        # 2. Handle player metadata
        logger.info("  Imputing player metadata...")
        
        # Hand - default to Right
        for col in ['winner_hand', 'loser_hand']:
            if col in df.columns:
                df[col] = df[col].fillna('R')
        
        # Height - median imputation
        for col in ['winner_ht', 'loser_ht']:
            if col in df.columns:
                median_ht = df[col].median()
                df[col] = df[col].fillna(median_ht)
        
        # Age - median imputation
        for col in ['winner_age', 'loser_age']:
            if col in df.columns:
                median_age = df[col].median()
                df[col] = df[col].fillna(median_age)
        
        # Country - drop rows with missing (critical for identification)
        ioc_cols = [col for col in ['winner_ioc', 'loser_ioc'] if col in df.columns]
        if ioc_cols:
            before = len(df)
            df = df.dropna(subset=ioc_cols)
            logger.info(f"  Dropped {before - len(df)} rows with missing country")
        
        # 3. Handle entry type
        for col in ['winner_entry', 'loser_entry']:
            if col in df.columns:
                df[col] = df[col].fillna('DA')
        
        # 4. Handle rank and rank points
        # Missing rank = weak player, fill with high rank (1000)
        for col in ['winner_rank', 'loser_rank']:
            if col in df.columns:
                df[col] = df[col].fillna(1000)
        
        for col in ['winner_rank_points', 'loser_rank_points']:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        # 5. Handle draw size
        if 'draw_size' in df.columns:
            mode_value = df['draw_size'].mode()
            mode_value = mode_value.iloc[0] if len(mode_value) > 0 else 32
            df['draw_size'] = df['draw_size'].fillna(mode_value)
        
        # 6. Handle indoor flag
        if 'indoor' in df.columns:
            # Convert O/I to 0/1
            df['indoor'] = df['indoor'].map({'O': 0, 'I': 1, 0: 0, 1: 1})
            df['indoor'] = df['indoor'].fillna(0).astype(int)
        
        # 7. Drop match_num if exists (not needed)
        if 'match_num' in df.columns:
            df = df.drop(columns=['match_num'])
        
        # 8. Keep seeds as NaN (unseeded)
        # Keep match statistics as NaN (will be used for rolling features)
        
        logger.info(f"  Final size: {len(df):,} ({initial_size - len(df)} dropped)")
        
        self.df_processed = df
        return df
    
    def temporal_split(
        self, 
        df: Optional[pd.DataFrame] = None,
        test_year: int = 2025
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data temporally - train on past, test on future.
        
        CRITICAL: This prevents data leakage from future to past!
        
        Args:
            df: DataFrame to split
            test_year: Year to use as test set
            
        Returns:
            Tuple of (train_df, test_df)
        """
        if df is None:
            df = self.df_processed
            
        df_train = df[df['data_year'] < test_year].copy()
        df_test = df[df['data_year'] >= test_year].copy()
        
        logger.info(f"Temporal split at {test_year}:")
        logger.info(f"  Train: {len(df_train):,} matches ({df_train['data_year'].min()}-{df_train['data_year'].max() if len(df_train) > 0 else 'N/A'})")
        logger.info(f"  Test:  {len(df_test):,} matches ({df_test['data_year'].min() if len(df_test) > 0 else 'N/A'}-{df_test['data_year'].max() if len(df_test) > 0 else 'N/A'})")
        
        return df_train, df_test
    
    def save_processed(
        self, 
        df_train: pd.DataFrame, 
        df_test: pd.DataFrame,
        output_path: str = 'data/processed'
    ):
        """
        Save processed train and test datasets.
        
        Args:
            df_train: Training DataFrame
            df_test: Test DataFrame
            output_path: Directory to save files
        """
        output_dir = Path(output_path)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        train_path = output_dir / 'train_2012_2024.csv'
        test_path = output_dir / 'test_2025.csv'
        
        df_train.to_csv(train_path, index=False)
        df_test.to_csv(test_path, index=False)
        
        logger.info(f"Saved: {train_path} ({train_path.stat().st_size / 1024**2:.1f} MB)")
        logger.info(f"Saved: {test_path} ({test_path.stat().st_size / 1024**2:.1f} MB)")


if __name__ == '__main__':
    # Test the processor
    processor = DataProcessor()
    df = processor.load_data(2012, 2025)
    df = processor.clean_data()
    train, test = processor.temporal_split()
    processor.save_processed(train, test)
