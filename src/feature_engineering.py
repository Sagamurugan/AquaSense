"""
Feature engineering module
Creates and transforms features for model input
"""

import pandas as pd
import numpy as np


def create_sequences(data, lookback=12):
    """Create sequences for LSTM/time series models"""
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback])
    return np.array(X), np.array(y)


def add_temporal_features(df):
    """Add temporal features (hour, day, month, season)"""
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['season'] = df['month'].apply(categorize_season)
    return df


def categorize_season(month):
    """Categorize months into seasons"""
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'


def engineer_features(df):
    """Main feature engineering pipeline"""
    df = add_temporal_features(df)
    return df
