"""
Anomaly detection module
Detects unusual patterns in water quality data
"""

import numpy as np
import joblib
import os
from sklearn.ensemble import IsolationForest

# Define path to the specific master model
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "anomaly_isolation_forest.pkl")

def load_anomaly_model():
    """Load the pre-trained Isolation Forest model"""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Anomaly model not found at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

def predict_anomaly(input_features):
    """
    Predict if the input is an anomaly.
    
    Args:
        input_features (list or np.array): Array of 6 features [TN, TP, NH3, NO23, OP, SSC]
        
    Returns:
        int: -1 for anomaly, 1 for normal
    """
    input_array = np.array(input_features)
    
    # Ensure 2D array
    if input_array.ndim == 1:
        input_array = input_array.reshape(1, -1)
        
    # Validation for feature count
    if input_array.shape[1] != 6:
        raise ValueError(f"Expected 6 features (TN, TP, NH3, NO23, OP, SSC), got {input_array.shape[1]}")
    
    try:
        model = load_anomaly_model()
        prediction = model.predict(input_array)
        return int(prediction[0])
    except Exception as e:
        print(f"Error during anomaly prediction: {e}")
        # Default to Normal (1) in case of error to avoid blocking the app, or re-raise
        # Prompt says "Return -1 for anomaly, 1 for normal"
        return 1

class AnomalyDetector:
    """Detects anomalies in water quality measurements"""
    
    def __init__(self, contamination=0.05):
        self.model = IsolationForest(contamination=contamination, random_state=42)
        self.is_fitted = False
    
    def fit(self, X):
        """Fit anomaly detection model"""
        self.model.fit(X)
        self.is_fitted = True
    
    def predict(self, X):
        """Predict anomalies (-1 for anomaly, 1 for normal)"""
        if not self.is_fitted:
            # Try loading pre-trained if not fitted manually
            try:
                self.model = load_anomaly_model()
                self.is_fitted = True
            except:
                raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def get_anomaly_score(self, X):
        """Get anomaly scores for samples"""
        return self.model.score_samples(X)
