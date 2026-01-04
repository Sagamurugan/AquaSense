"""
Data preprocessing module for AquaSense
Handles data loading, cleaning, and transformation
"""

import pandas as pd
import numpy as np


def load_data(filepath):
    """Load data from CSV file"""
    return pd.read_csv(filepath)


def clean_data(df):
    """Clean dataset - remove duplicates and handle missing values"""
    df = df.drop_duplicates()
    return df


def merge_datasets(data_list):
    """Merge multiple datasets"""
    return pd.concat(data_list, axis=1)


def scale_features(df, scaler=None):
    """Scale features using StandardScaler"""
    from sklearn.preprocessing import StandardScaler
    
    if scaler is None:
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df)
    else:
        df_scaled = scaler.transform(df)
    
    return df_scaled, scaler
