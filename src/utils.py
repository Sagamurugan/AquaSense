"""
Utility functions for AquaSense project
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def save_scaler(scaler, filepath):
    """Save scaler object to file"""
    with open(filepath, 'wb') as f:
        pickle.dump(scaler, f)


def load_scaler(filepath):
    """Load scaler object from file"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def calculate_metrics(y_true, y_pred):
    """Calculate evaluation metrics"""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2
    }
    return metrics


def save_metrics(metrics, filepath):
    """Save metrics to text file"""
    with open(filepath, 'w') as f:
        for metric_name, metric_value in metrics.items():
            f.write(f"{metric_name}: {metric_value:.6f}\n")


def save_predictions(predictions, timestamps, filepath):
    """Save predictions to CSV"""
    df = pd.DataFrame({
        'timestamp': timestamps,
        'prediction': predictions
    })
    df.to_csv(filepath, index=False)


def create_directories(paths):
    """Create multiple directories if they don't exist"""
    for path in paths:
        os.makedirs(path, exist_ok=True)
