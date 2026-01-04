"""
Anomaly detection module
Detects unusual patterns in water quality data
"""

import numpy as np
from sklearn.ensemble import IsolationForest


class AnomalyDetector:
    """Detects anomalies in water quality measurements"""
    
    def __init__(self, contamination=0.1):
        self.model = IsolationForest(contamination=contamination, random_state=42)
        self.is_fitted = False
    
    def fit(self, X):
        """Fit anomaly detection model"""
        self.model.fit(X)
        self.is_fitted = True
    
    def predict(self, X):
        """Predict anomalies (-1 for anomaly, 1 for normal)"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def get_anomaly_score(self, X):
        """Get anomaly scores for samples"""
        return self.model.score_samples(X)
    
    def detect_threshold_anomalies(self, data, lower_bound, upper_bound):
        """Detect anomalies based on threshold"""
        return (data < lower_bound) | (data > upper_bound)
