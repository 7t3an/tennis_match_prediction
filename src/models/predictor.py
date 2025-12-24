"""Model loading and prediction utilities."""

from typing import Dict, List, Optional, Tuple
import pickle
from pathlib import Path
import pandas as pd
import streamlit as st


@st.cache_resource
def load_model() -> Tuple[Optional[object], Optional[List[str]], Optional[Dict]]:
    """Load XGBoost model, feature columns and label encoders.
    
    Returns:
        tuple: (model, feature_cols, label_encoders)
            - model: Trained XGBoost calibrated classifier
            - feature_cols: List of feature column names
            - label_encoders: Dictionary of LabelEncoders for categorical features
    """
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
    except FileNotFoundError:
        # Model files not found - app will work in limited mode
        return None, None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None


def prepare_features_for_prediction(
    features_dict: Dict,
    feature_cols: List[str],
    label_encoders: Optional[Dict]
) -> pd.DataFrame:
    """Prepare features dictionary for model prediction.
    
    Converts features dictionary to DataFrame, handles missing features,
    and encodes categorical variables using label encoders.
    
    Args:
        features_dict: Dictionary of calculated features
        feature_cols: List of expected feature column names
        label_encoders: Dictionary of LabelEncoders for categorical features
        
    Returns:
        DataFrame: Prepared features ready for model.predict_proba()
    """
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
    
    return input_df


def predict_match(
    model,
    features_df: pd.DataFrame,
    was_swapped: bool = False
) -> Tuple[float, float]:
    """Predict match outcome using trained model.
    
    Args:
        model: Trained XGBoost calibrated classifier
        features_df: Prepared features DataFrame
        was_swapped: Boolean indicating if player order was swapped
        
    Returns:
        tuple: (prob_p1_wins, prob_p2_wins)
            Probabilities for original P1 and P2 players
    """
    prob_p1_wins = model.predict_proba(features_df)[0, 1]
    prob_p2_wins = 1 - prob_p1_wins
    
    # Invert probabilities if players were swapped for model normalization
    if was_swapped:
        prob_p1_wins, prob_p2_wins = prob_p2_wins, prob_p1_wins
    
    return prob_p1_wins, prob_p2_wins
