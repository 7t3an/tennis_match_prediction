"""
Feature Engineering for Tennis Match Prediction.

CRITICAL: NO DATA LEAKAGE!
- All features must be available BEFORE the match
- Rolling stats use only PAST matches (shift(1) + rolling)
- H2H uses only PAST head-to-head results
- Player IDs are NEVER used as features (would memorize strong players)
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Create ML features from tennis match data.
    
    Feature Groups (NO LEAKAGE):
    1. Rank-based: rank_diff, rank_ratio, rank_points_diff
    2. Seed-based: is_seeded, seed_tier, seed_diff (optional - disabled by default)
    3. Rolling stats: last N matches performance (SHIFTED!)
    4. H2H: historical head-to-head record
    5. Match context: surface, tournament level
    6. Player metadata: hand, height, age
    
    EXCLUDED (DATA LEAKAGE):
    - Player IDs (model would memorize strong players)
    - Current match statistics (aces, df, etc.)
    - Match result indicators
    
    NOTE: Seed features are disabled by default because they cause artificially
    inflated metrics (seeding strongly correlates with winning).
    """
    
    # Rolling window parameters
    ROLLING_WINDOW = 10
    MIN_PERIODS = 3
    
    # Match statistics columns for rolling features
    STATS_COLS = [
        'ace', 'df', 'svpt', '1stIn', '1stWon', '2ndWon',
        'SvGms', 'bpSaved', 'bpFaced'
    ]
    
    def __init__(self, use_seed_features: bool = False):
        """
        Initialize feature engineer.
        
        Args:
            use_seed_features: Whether to use seed features. Default False because 
                              seeding information strongly correlates with winning,
                              leading to artificially high AUC metrics (~0.92 vs ~0.70).
        """
        self.use_seed_features = use_seed_features
        self.feature_cols = []
        
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all features for the dataset.
        
        Args:
            df: Raw match data with winner/loser format
            
        Returns:
            DataFrame in P1/P2 format with all features
        """
        logger.info("FEATURE ENGINEERING")
        
        # Make copy and ensure sorted by date
        df = df.copy()
        df = df.sort_values('tourney_date').reset_index(drop=True)
        df['row_id'] = df.index
        
        # Step 1: Seed features
        df = self._create_seed_features(df)
        
        # Step 2: Rolling statistics (CRITICAL - no leakage!)
        df = self._create_rolling_features(df)
        
        # Step 3: Head-to-head features
        df = self._create_h2h_features(df)
        
        # Step 4: Tournament/surface features
        df = self._create_context_features(df)
        
        # Step 5: Convert to P1/P2 format
        df_features = self._convert_to_p1_p2_format(df)
        
        # Step 6: Create rank-based features (after P1/P2 conversion!)
        df_features = self._create_rank_features(df_features)
        
        # Step 7: Create form features from rolling stats
        df_features = self._create_form_features(df_features)
        
        # Step 8: Create experience features
        df_features = self._create_experience_features(df_features)
        
        # Step 9: Final cleanup
        df_features = self._cleanup_features(df_features)
        
        logger.info(f"Feature engineering complete: {len(df_features.columns)} columns")
        
        return df_features
    
    def _create_seed_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create seed-based features."""
        logger.info("Creating SEED features...")
        
        # Binary indicators
        df['is_winner_seeded'] = df['winner_seed'].notna().astype(int)
        df['is_loser_seeded'] = df['loser_seed'].notna().astype(int)
        
        # Seed difference (only when both seeded)
        df['seed_diff'] = df['winner_seed'].fillna(999) - df['loser_seed'].fillna(999)
        
        # Seed tiers
        def get_seed_tier(seed):
            if pd.isna(seed):
                return 'Not_Seeded'
            elif seed <= 4:
                return 'Top4'
            elif seed <= 8:
                return 'Top8'
            elif seed <= 16:
                return 'Top16'
            else:
                return 'Top32+'
        
        df['winner_seed_tier'] = df['winner_seed'].apply(get_seed_tier)
        df['loser_seed_tier'] = df['loser_seed'].apply(get_seed_tier)
        
        logger.info(f"  Created: is_seeded, seed_diff, seed_tier")
        return df
    
    def _create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create rolling statistics features.
        
        CRITICAL FOR NO DATA LEAKAGE:
        1. Group by player_id (not winner/loser!)
        2. Sort by date
        3. Use shift(1) - exclude current match
        4. Rolling mean on PAST matches only
        """
        logger.info("Creating ROLLING features (NO LEAKAGE)...")
        logger.info(f"   Window: {self.ROLLING_WINDOW}, Min periods: {self.MIN_PERIODS}")
        
        # Create long-format: each player's match as separate row
        winner_df = df[['tourney_date', 'row_id', 'winner_id', 'winner_name'] + 
                      [f'w_{s}' for s in self.STATS_COLS if f'w_{s}' in df.columns]].copy()
        loser_df = df[['tourney_date', 'row_id', 'loser_id', 'loser_name'] + 
                     [f'l_{s}' for s in self.STATS_COLS if f'l_{s}' in df.columns]].copy()
        
        # Rename columns
        winner_cols = ['date', 'row_id', 'player_id', 'player_name'] + self.STATS_COLS[:len([s for s in self.STATS_COLS if f'w_{s}' in df.columns])]
        loser_cols = ['date', 'row_id', 'player_id', 'player_name'] + self.STATS_COLS[:len([s for s in self.STATS_COLS if f'l_{s}' in df.columns])]
        
        winner_df.columns = winner_cols
        loser_df.columns = loser_cols
        
        # Combine and sort
        player_stats = pd.concat([winner_df, loser_df], ignore_index=True)
        player_stats = player_stats.sort_values(['player_id', 'date', 'row_id']).reset_index(drop=True)
        
        logger.info(f"   Long format: {len(player_stats):,} player-match records")
        
        # Compute rolling stats for each player
        # CRITICAL: shift(1) ensures we don't use current match stats!
        for stat in tqdm(self.STATS_COLS, desc="   Computing rolling stats"):
            if stat not in player_stats.columns:
                continue
                
            col_name = f"{stat}_roll{self.ROLLING_WINDOW}"
            
            player_stats[col_name] = player_stats.groupby('player_id')[stat].transform(
                lambda x: x.shift(1).rolling(
                    window=self.ROLLING_WINDOW, 
                    min_periods=self.MIN_PERIODS
                ).mean()
            )
        
        # Create lookup for mapping back to original data
        roll_cols = [f"{stat}_roll{self.ROLLING_WINDOW}" for stat in self.STATS_COLS 
                    if stat in player_stats.columns]
        
        player_stats['lookup_key'] = (
            player_stats['player_id'].astype(str) + '_' + 
            player_stats['row_id'].astype(str)
        )
        lookup_dict = player_stats.set_index('lookup_key')[roll_cols].to_dict('index')
        
        # Map back to winner and loser
        df['winner_lookup'] = df['winner_id'].astype(str) + '_' + df['row_id'].astype(str)
        df['loser_lookup'] = df['loser_id'].astype(str) + '_' + df['row_id'].astype(str)
        
        for stat in self.STATS_COLS:
            roll_col = f"{stat}_roll{self.ROLLING_WINDOW}"
            if roll_col not in roll_cols:
                continue
                
            # Winner rolling stats
            df[f'w_{roll_col}'] = df['winner_lookup'].map(
                lambda x: lookup_dict.get(x, {}).get(roll_col, np.nan)
            )
            
            # Loser rolling stats
            df[f'l_{roll_col}'] = df['loser_lookup'].map(
                lambda x: lookup_dict.get(x, {}).get(roll_col, np.nan)
            )
        
        # Cleanup temporary columns
        df = df.drop(['winner_lookup', 'loser_lookup'], axis=1)
        
        # Report coverage
        w_roll_filled = df['w_ace_roll10'].notna().sum() if 'w_ace_roll10' in df.columns else 0
        logger.info(f"   Rolling stats coverage: {w_roll_filled:,} / {len(df):,} ({w_roll_filled/len(df)*100:.1f}%)")
        
        return df
    
    def _create_h2h_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create head-to-head features.
        
        NO DATA LEAKAGE: Only use PAST H2H results!
        """
        logger.info("Creating H2H features (NO LEAKAGE)...")
        
        # Create canonical H2H key (alphabetically sorted)
        def create_h2h_key(row):
            players = sorted([str(row['winner_id']), str(row['loser_id'])])
            return f"{players[0]}_vs_{players[1]}"
        
        df['h2h_key'] = df.apply(create_h2h_key, axis=1)
        
        # Initialize H2H columns
        df['h2h_winner_wins'] = 0
        df['h2h_loser_wins'] = 0
        df['h2h_total_matches'] = 0
        df['h2h_winner_win_rate'] = 0.5  # Default when no H2H history
        
        # Track H2H history as we iterate through matches
        h2h_history = {}  # {h2h_key: {player_id: wins}}
        
        # Process matches chronologically (already sorted by date)
        logger.info("   Processing matches chronologically...")
        
        for idx in tqdm(df.index, desc="   Building H2H history"):
            row = df.loc[idx]
            h2h_key = row['h2h_key']
            winner_id = str(row['winner_id'])
            loser_id = str(row['loser_id'])
            
            # Get PAST H2H record (before this match)
            if h2h_key in h2h_history:
                history = h2h_history[h2h_key]
                w_wins = history.get(winner_id, 0)
                l_wins = history.get(loser_id, 0)
                total = w_wins + l_wins
                
                df.at[idx, 'h2h_winner_wins'] = w_wins
                df.at[idx, 'h2h_loser_wins'] = l_wins
                df.at[idx, 'h2h_total_matches'] = total
                df.at[idx, 'h2h_winner_win_rate'] = w_wins / total if total > 0 else 0.5
            
            # Update H2H history (for future matches)
            if h2h_key not in h2h_history:
                h2h_history[h2h_key] = {}
            if winner_id not in h2h_history[h2h_key]:
                h2h_history[h2h_key][winner_id] = 0
            if loser_id not in h2h_history[h2h_key]:
                h2h_history[h2h_key][loser_id] = 0
            
            h2h_history[h2h_key][winner_id] += 1
        
        matches_with_h2h = (df['h2h_total_matches'] > 0).sum()
        logger.info(f"   Matches with H2H history: {matches_with_h2h:,} ({matches_with_h2h/len(df)*100:.1f}%)")
        
        return df
    
    def _create_context_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create tournament and surface context features."""
        logger.info("Creating CONTEXT features...")
        
        # Tournament level encoding
        tourney_level_map = {
            'G': 4,  # Grand Slam
            'M': 3,  # Masters 1000
            'A': 2,  # ATP 500/250
            'D': 1,  # Davis Cup
            'F': 3   # Tour Finals (similar to Masters)
        }
        
        df['tourney_level_encoded'] = df['tourney_level'].map(tourney_level_map).fillna(1)
        
        # Surface encoding
        surface_map = {
            'Hard': 3,
            'Clay': 2,
            'Grass': 1,
            'Carpet': 0
        }
        
        df['surface_encoded'] = df['surface'].map(surface_map).fillna(3)
        
        logger.info(f"   Tournament level encoded: {df['tourney_level_encoded'].value_counts().to_dict()}")
        logger.info(f"   Surface encoded: {df['surface_encoded'].value_counts().to_dict()}")
        
        return df
    
    def _convert_to_p1_p2_format(self, df: pd.DataFrame, duplicate: bool = False) -> pd.DataFrame:
        """
        Convert from winner/loser format to P1/P2 format.
        
        For training: RANDOMLY assign P1 and P2 (NO duplication!)
        - This avoids data leakage from having both versions of same match
        - Model learns to predict from any player perspective
        
        Args:
            df: DataFrame in winner/loser format
            duplicate: If True, create both versions (ONLY for data augmentation, NOT for CV!)
        """
        logger.info("Converting to P1/P2 format...")
        logger.info(f"   Duplicate mode: {duplicate}")
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Define column mappings
        winner_cols = [col for col in df.columns if col.startswith('winner_') 
                      and not col.endswith('_tier') and col != 'winner_lookup']
        loser_cols = [col for col in df.columns if col.startswith('loser_') 
                     and not col.endswith('_tier') and col != 'loser_lookup']
        w_stats = [col for col in df.columns if col.startswith('w_')]
        l_stats = [col for col in df.columns if col.startswith('l_')]
        
        # Generate random swap mask
        swap_mask = np.random.random(len(df)) > 0.5
        
        result_rows = []
        
        for idx, (i, row) in enumerate(df.iterrows()):
            should_swap = swap_mask[idx]
            new_row = row.to_dict()
            
            if not should_swap:
                # P1 = Winner, P2 = Loser, target = 1
                for col in winner_cols:
                    new_row[col.replace('winner_', 'p1_')] = row[col]
                for col in loser_cols:
                    new_row[col.replace('loser_', 'p2_')] = row[col]
                for col in w_stats:
                    new_row[col.replace('w_', 'p1_')] = row[col]
                for col in l_stats:
                    new_row[col.replace('l_', 'p2_')] = row[col]
                
                # Seed tier
                if 'winner_seed_tier' in row:
                    new_row['p1_seed_tier'] = row['winner_seed_tier']
                if 'loser_seed_tier' in row:
                    new_row['p2_seed_tier'] = row['loser_seed_tier']
                
                # H2H
                if 'h2h_winner_wins' in row:
                    new_row['h2h_p1_wins'] = row['h2h_winner_wins']
                if 'h2h_loser_wins' in row:
                    new_row['h2h_p2_wins'] = row['h2h_loser_wins']
                if 'h2h_winner_win_rate' in row:
                    new_row['h2h_p1_win_rate'] = row['h2h_winner_win_rate']
                
                # Seeded flags
                if 'is_winner_seeded' in row:
                    new_row['is_p1_seeded'] = row['is_winner_seeded']
                if 'is_loser_seeded' in row:
                    new_row['is_p2_seeded'] = row['is_loser_seeded']
                
                new_row['p1_won'] = 1
            else:
                # P1 = Loser, P2 = Winner, target = 0 (SWAPPED)
                for col in winner_cols:
                    new_row[col.replace('winner_', 'p2_')] = row[col]
                for col in loser_cols:
                    new_row[col.replace('loser_', 'p1_')] = row[col]
                for col in w_stats:
                    new_row[col.replace('w_', 'p2_')] = row[col]
                for col in l_stats:
                    new_row[col.replace('l_', 'p1_')] = row[col]
                
                # Seed tier
                if 'winner_seed_tier' in row:
                    new_row['p2_seed_tier'] = row['winner_seed_tier']
                if 'loser_seed_tier' in row:
                    new_row['p1_seed_tier'] = row['loser_seed_tier']
                
                # H2H (swap!)
                if 'h2h_winner_wins' in row:
                    new_row['h2h_p2_wins'] = row['h2h_winner_wins']
                if 'h2h_loser_wins' in row:
                    new_row['h2h_p1_wins'] = row['h2h_loser_wins']
                if 'h2h_winner_win_rate' in row:
                    # Invert win rate
                    total = row.get('h2h_total_matches', 0)
                    if total > 0:
                        new_row['h2h_p1_win_rate'] = 1 - row['h2h_winner_win_rate']
                    else:
                        new_row['h2h_p1_win_rate'] = 0.5
                
                # Seeded flags (swap!)
                if 'is_winner_seeded' in row:
                    new_row['is_p2_seeded'] = row['is_winner_seeded']
                if 'is_loser_seeded' in row:
                    new_row['is_p1_seeded'] = row['is_loser_seeded']
                
                new_row['p1_won'] = 0
            
            result_rows.append(new_row)
        
        df_combined = pd.DataFrame(result_rows)
        
        # Clean up old columns
        cols_to_drop = (
            winner_cols + loser_cols + w_stats + l_stats +
            ['winner_seed_tier', 'loser_seed_tier', 'h2h_winner_wins', 'h2h_loser_wins',
             'h2h_winner_win_rate', 'is_winner_seeded', 'is_loser_seeded']
        )
        cols_to_drop = [c for c in cols_to_drop if c in df_combined.columns]
        df_combined = df_combined.drop(columns=cols_to_drop, errors='ignore')
        
        # Sort by date to maintain temporal order
        if 'tourney_date' in df_combined.columns:
            df_combined = df_combined.sort_values('tourney_date').reset_index(drop=True)
        
        logger.info(f"   Created {len(df_combined):,} rows (same as input: {len(df):,})")
        logger.info(f"   Target balance: P1 won {df_combined['p1_won'].mean()*100:.1f}%")
        
        return df_combined
    
    def _create_rank_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create rank-based features.
        
        These are created AFTER P1/P2 conversion to avoid leakage!
        """
        logger.info("Creating RANK features...")
        
        # Rank difference (negative = P1 is better ranked)
        df['rank_diff'] = df['p1_rank'] - df['p2_rank']
        
        # Rank ratio
        df['rank_ratio'] = df['p1_rank'] / df['p2_rank'].replace(0, 1)
        
        # Who is favorite based on rank
        df['is_p1_favorite'] = (df['p1_rank'] < df['p2_rank']).astype(int)
        
        # Rank points difference
        df['rank_points_diff'] = df['p1_rank_points'] - df['p2_rank_points']
        
        # Rank points ratio
        df['rank_points_ratio'] = df['p1_rank_points'] / df['p2_rank_points'].replace(0, 1)
        
        # Log rank ratio (more stable for large differences)
        df['log_rank_ratio'] = np.log1p(df['p1_rank']) - np.log1p(df['p2_rank'])
        
        # H2H P2 win rate (complement)
        if 'h2h_p1_win_rate' in df.columns:
            df['h2h_p2_win_rate'] = 1 - df['h2h_p1_win_rate']
        
        logger.info(f"   Created: rank_diff, rank_ratio, is_p1_favorite, rank_points_diff, rank_points_ratio, log_rank_ratio")
        
        return df
    
    def _create_form_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create recent form features from rolling stats.
        
        These capture player's current form and confidence.
        """
        logger.info("Creating FORM features...")
        
        # Service dominance: ratio of 1st serve wins
        for prefix in ['p1', 'p2']:
            svpt_col = f'{prefix}_svpt_roll10'
            first_won_col = f'{prefix}_1stWon_roll10'
            second_won_col = f'{prefix}_2ndWon_roll10'
            ace_col = f'{prefix}_ace_roll10'
            df_col = f'{prefix}_df_roll10'
            bp_saved_col = f'{prefix}_bpSaved_roll10'
            bp_faced_col = f'{prefix}_bpFaced_roll10'
            
            # Service efficiency (1st serve win % from available stats)
            if first_won_col in df.columns and svpt_col in df.columns:
                df[f'{prefix}_serve_efficiency'] = df[first_won_col] / df[svpt_col].replace(0, 1)
            
            # Ace to double fault ratio
            if ace_col in df.columns and df_col in df.columns:
                df[f'{prefix}_ace_df_ratio'] = df[ace_col] / (df[df_col] + 1)
            
            # Break point save rate
            if bp_saved_col in df.columns and bp_faced_col in df.columns:
                df[f'{prefix}_bp_save_rate'] = df[bp_saved_col] / df[bp_faced_col].replace(0, 1)
        
        # Differential features (P1 - P2)
        if 'p1_serve_efficiency' in df.columns and 'p2_serve_efficiency' in df.columns:
            df['serve_efficiency_diff'] = df['p1_serve_efficiency'] - df['p2_serve_efficiency']
        
        if 'p1_ace_df_ratio' in df.columns and 'p2_ace_df_ratio' in df.columns:
            df['ace_df_ratio_diff'] = df['p1_ace_df_ratio'] - df['p2_ace_df_ratio']
            
        if 'p1_bp_save_rate' in df.columns and 'p2_bp_save_rate' in df.columns:
            df['bp_save_rate_diff'] = df['p1_bp_save_rate'] - df['p2_bp_save_rate']
        
        logger.info(f"   Created form features")
        
        return df
    
    def _create_experience_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create experience and physical features.
        """
        logger.info("Creating EXPERIENCE features...")
        
        # Age difference
        if 'p1_age' in df.columns and 'p2_age' in df.columns:
            df['age_diff'] = df['p1_age'] - df['p2_age']
        
        # Height difference (potential serve advantage)
        if 'p1_ht' in df.columns and 'p2_ht' in df.columns:
            df['height_diff'] = df['p1_ht'] - df['p2_ht']
        
        logger.info(f"   Created: age_diff, height_diff")
        
        return df
    
    def _cleanup_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Final cleanup - remove columns that should not be used as features.
        
        REMOVE (DATA LEAKAGE or not useful):
        - Player IDs (would memorize strong players!)
        - Player names
        - Match identifiers
        - Current match statistics (not rolling)
        - Result columns
        
        OPTIONALLY REMOVE (if use_seed_features=False):
        - All seed-related features (they cause artificially high AUC)
        """
        logger.info("Final cleanup (removing potential leakage columns)...")
        
        # Columns to drop
        drop_cols = [
            # IDs - CRITICAL! Model would memorize strong players
            'p1_id', 'p2_id',
            
            # Names
            'p1_name', 'p2_name',
            
            # Technical
            'tourney_id', 'tourney_name', 'tourney_date',
            'row_id', 'h2h_key', 'data_year',
            
            # Match results (LEAKAGE!)
            'score', 'best_of', 'round', 'minutes',
            
            # Raw seed values always dropped
            'p1_seed', 'p2_seed',
        ]
        
        # If not using seed features, remove ALL seed-related columns
        if not self.use_seed_features:
            seed_cols_to_drop = [
                'seed_diff', 'p1_seed_tier', 'p2_seed_tier',
                'is_p1_seeded', 'is_p2_seeded'
            ]
            drop_cols.extend(seed_cols_to_drop)
            logger.info("   Removing seed features (use_seed_features=False)")
        
        # Find current match statistics to drop
        for prefix in ['p1_', 'p2_']:
            for stat in self.STATS_COLS:
                col = f'{prefix}{stat}'
                if col in df.columns and col not in drop_cols:
                    drop_cols.append(col)
        
        # Drop columns that exist
        cols_to_drop = [col for col in drop_cols if col in df.columns]
        df = df.drop(columns=cols_to_drop, errors='ignore')
        
        logger.info(f"   Dropped {len(cols_to_drop)} columns")
        logger.info(f"   Remaining: {len(df.columns)} columns")
        
        # List final features
        self.feature_cols = [col for col in df.columns if col != 'p1_won']
        
        return df
    
    def get_feature_list(self) -> List[str]:
        """Return list of feature column names."""
        return self.feature_cols
    
    def validate_no_leakage(self, df: pd.DataFrame) -> bool:
        """
        Validate that there is no data leakage in features.
        
        Checks for:
        1. No player IDs in features
        2. No current match statistics
        3. No result columns
        """
        logger.info("Validating NO DATA LEAKAGE...")
        
        issues = []
        
        # Check for player IDs
        id_cols = [col for col in df.columns if '_id' in col.lower()]
        if id_cols:
            issues.append(f"Player IDs found: {id_cols}")
        
        # Check for result columns
        result_patterns = ['score', 'winner', 'loser', 'result', 'outcome']
        for pattern in result_patterns:
            found = [col for col in df.columns if pattern in col.lower() and col != 'p1_won']
            if found:
                issues.append(f"Result columns found ({pattern}): {found}")
        
        # Check for non-rolling match stats
        for stat in self.STATS_COLS:
            for prefix in ['p1_', 'p2_', 'w_', 'l_']:
                col = f'{prefix}{stat}'
                if col in df.columns and '_roll' not in col:
                    issues.append(f"Non-rolling stat found: {col}")
        
        if issues:
            logger.error("DATA LEAKAGE DETECTED!")
            for issue in issues:
                logger.error(f"   - {issue}")
            return False
        else:
            logger.info("   No data leakage detected!")
            return True


if __name__ == '__main__':
    # Test feature engineering
    from data_processor import DataProcessor
    
    processor = DataProcessor()
    df = processor.load_data(2012, 2025)
    df = processor.clean_data()
    
    engineer = FeatureEngineer()
    df_features = engineer.create_all_features(df)
    
    # Validate
    engineer.validate_no_leakage(df_features)
    
    print("Final feature list:")
    for i, feat in enumerate(engineer.get_feature_list(), 1):
        print(f"  {i}. {feat}")
