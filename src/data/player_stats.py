"""Player statistics and match history utilities."""

from typing import Optional, Tuple
import pandas as pd
import streamlit as st
import re
import unicodedata


@st.cache_data
def _load_tml_matches():
    """Load processed matches with scores for lookup (train + latest test).
    Uses: data/processed/train_2012_2024.csv and data/processed/test_2025.csv
    Returns a DataFrame with columns: tourney_id, tourney_name, surface, tourney_date, winner_name, loser_name, score.
    """
    frames = []
    paths = [
        'data/processed/train_2012_2024.csv',
        'data/processed/test_2025.csv'
    ]
    for path in paths:
        try:
            df = pd.read_csv(
                path,
                usecols=[
                    'tourney_id', 'tourney_name', 'surface', 'tourney_date',
                    'winner_name', 'loser_name', 'score'
                ]
            )
            frames.append(df)
        except Exception as e:
            # Fallback: try different encodings if needed
            for enc in ('utf-8-sig', 'latin1', 'cp1252'):
                try:
                    df = pd.read_csv(path, encoding=enc)
                    frames.append(df[[
                        'tourney_id', 'tourney_name', 'surface', 'tourney_date',
                        'winner_name', 'loser_name', 'score'
                    ]])
                    break
                except Exception:
                    continue
            else:
                st.warning(f"Unable to load processed matches from {path}: {e}")
                continue
    if frames:
        df = pd.concat(frames, ignore_index=True)
        # Precompute normalized fields for robust lookup
        def _norm(x):
            if not isinstance(x, str):
                return ""
            s = unicodedata.normalize('NFKD', x)
            s = ''.join(ch for ch in s if not unicodedata.combining(ch))
            s = s.lower().strip()
            s = re.sub(r"\s+", " ", s)
            return s
        df['__w'] = df['winner_name'].apply(_norm)
        df['__l'] = df['loser_name'].apply(_norm)
        df['__tn'] = df['tourney_name'].apply(_norm)
        df['__ts'] = df['tourney_date'].apply(_normalize_date_to_ts)
        return df
    return pd.DataFrame()


def _normalize_date_to_ts(value):
    """Convert various date formats to pandas Timestamp (date precision)."""
    try:
        if pd.isna(value):
            return None
        # Features set often has ISO date like '2024-12-29'
        if isinstance(value, str):
            return pd.to_datetime(value).normalize()
        # TML has integer like 20240101
        if isinstance(value, (int, float)):
            return pd.to_datetime(str(int(value))).normalize()
        return pd.to_datetime(value).normalize()
    except Exception:
        return None


def _lookup_score(tml_df, tourney_id, tourney_name, feat_date, winner_name, loser_name):
    """Find a score in raw TML by best available keys.
    Strategy:
    1) Exact match by tourney_id + winner/loser.
    2) Fallback: tourney_name + winner/loser and nearest date.
    """
    if tml_df is None or len(tml_df) == 0:
        return None

    # Normalize inputs
    def _norm(x):
        if not isinstance(x, str):
            return ""
        s = unicodedata.normalize('NFKD', x)
        s = ''.join(ch for ch in s if not unicodedata.combining(ch))
        s = s.lower().strip()
        s = re.sub(r"\s+", " ", s)
        return s
    wn = _norm(winner_name)
    ln = _norm(loser_name)
    tn = _norm(tourney_name)
    ts = _normalize_date_to_ts(feat_date)

    # Try exact tourney_id match first (check both name orders)
    try:
        mask_id = tml_df['tourney_id'].astype(str) == str(tourney_id)
        mask_names = (tml_df['__w'] == wn) & (tml_df['__l'] == ln)
        mask_names_rev = (tml_df['__w'] == ln) & (tml_df['__l'] == wn)
        candidates = tml_df[mask_id & (mask_names | mask_names_rev)].copy()
        if len(candidates) > 0:
            if ts is not None and candidates['__ts'].notna().any():
                candidates['__diff'] = (candidates['__ts'] - ts).abs()
                candidates = candidates.sort_values('__diff')
            return candidates.iloc[0]['score']
    except Exception:
        pass

    # Fallback: match by tourney_name and names, then closest date
    try:
        candidates = tml_df[
            (tml_df['__tn'] == tn) &
            (
                ((tml_df['__w'] == wn) & (tml_df['__l'] == ln)) |
                ((tml_df['__w'] == ln) & (tml_df['__l'] == wn))
            )
        ].copy()
        if len(candidates) > 0:
            if ts is not None and candidates['__ts'].notna().any():
                candidates['__diff'] = (candidates['__ts'] - ts).abs()
                candidates = candidates.sort_values('__diff')
            return candidates.iloc[0]['score']
    except Exception:
        pass

    return None


def _normalize_score_for_player(score_str: str, player_won: bool) -> str:
    """Return score string in player's perspective.
    Example: original (winner perspective) "7-6(4) 7-5" ->
      - if player_won=True: "7-6(4) 7-5"
      - if player_won=False: "6-7(4) 5-7"
    Non-standard tokens (RET, W/O) remain unchanged.
    """
    if not isinstance(score_str, str) or not score_str.strip():
        return score_str or 'N/A'

    if player_won:
        return score_str

    tokens = score_str.split()
    out = []
    for tok in tokens:
        m = re.match(r"^(\d+)-(\d+)(\(\d+\))?$", tok)
        if m:
            a = m.group(1)
            b = m.group(2)
            tb = m.group(3) or ''
            out.append(f"{b}-{a}{tb}")
        else:
            out.append(tok)
    return ' '.join(out)
def _extract_match_status(score_str: str):
    """Extract special match status (RET, W/O, DEF, ABD) and cleaned score.
    Returns tuple: (status_or_None, cleaned_score_str)
    """
    if not isinstance(score_str, str):
        return None, 'N/A'

    s = score_str.strip()
    if not s:
        return None, 'N/A'

    low = s.lower()
    # Walkover
    if 'w/o' in low or 'walkover' in low:
        return 'W/O', 'W/O'
    # Retired
    if 'ret' in low or 'retired' in low:
        # remove literal tokens RET/retired from string
        cleaned = re.sub(r"\b(ret|retired)\b", "", s, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        return 'RET', cleaned if cleaned else 'RET'
    # Default
    if 'def' in low or 'default' in low:
        cleaned = re.sub(r"\b(def|default)\b", "", s, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        return 'DEF', cleaned if cleaned else 'DEF'
    # Abandoned / Suspended
    if 'abd' in low or 'abandoned' in low or 'susp' in low or 'suspended' in low:
        cleaned = re.sub(r"\b(abd|abandoned|susp|suspended)\b", "", s, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        return 'ABD', cleaned if cleaned else 'ABD'

    return None, s


def get_player_stats(_df, player_name):
    """Get player's latest statistics from most recent match.
    
    Args:
        _df: DataFrame with match data
        player_name: Name of the player
        
    Returns:
        dict: Player statistics including rank, points, and rolling stats
        None: If player not found
    """
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
    """Get player's last 10 matches for display.
    
    Args:
        _df: DataFrame with match data
        player_name: Name of the player
        
    Returns:
        DataFrame: Last 10 matches with formatted information
    """
    player_matches = _df[
        (_df['p1_name'] == player_name) | (_df['p2_name'] == player_name)
    ].copy()
    
    if len(player_matches) == 0:
        return pd.DataFrame()
    
    if 'tourney_date' in player_matches.columns:
        player_matches = player_matches.sort_values('tourney_date', ascending=False)
    
    recent_matches = player_matches.head(10).copy()
    
    formatted_matches = []
    tml_df = _load_tml_matches()
    for _, match in recent_matches.iterrows():
        is_p1 = match['p1_name'] == player_name
        opponent = match['p2_name'] if is_p1 else match['p1_name']
        
        if is_p1:
            won = match['p1_won'] == 1
        else:
            won = match['p1_won'] == 0
        
        # Try to lookup score from raw TML
        tourney_id = match.get('tourney_id', None)
        tourney_name = match.get('tourney_name', None)
        feat_date = match.get('tourney_date', None)
        score = _lookup_score(
            tml_df,
            tourney_id=tourney_id,
            tourney_name=tourney_name,
            feat_date=feat_date,
            winner_name=player_name if won else opponent,
            loser_name=opponent if won else player_name,
        )

        status, raw_score = _extract_match_status(score if score else '')
        score_display = _normalize_score_for_player(raw_score if raw_score else 'N/A', player_won=won)

        # Format date from 20251110 to 2025.11.10
        raw_date = str(match.get('tourney_date', 'N/A'))
        if raw_date and len(raw_date) == 8 and raw_date.isdigit():
            formatted_date = f"{raw_date[:4]}.{raw_date[4:6]}.{raw_date[6:]}"
        else:
            formatted_date = raw_date

        formatted_matches.append({
            'Date': formatted_date,
            'Tournament': match.get('tourney_name', 'N/A'),
            'Surface': match.get('surface', 'N/A'),
            'Opponent': opponent,
            'Result': 'WIN' if won else 'LOSS',
            'Score': score_display,
            'Status': status if status else '',
        })
    
    return pd.DataFrame(formatted_matches)
