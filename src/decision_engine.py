import numpy as np

def decide_and_predict(is_anomaly, prediction_value):
    if is_anomaly == -1:
        return {
            "status": "ANOMALY DETECTED",
            "message": "Prediction may be unreliable due to abnormal conditions",
            "prediction": None
        }
    else:
        return {
            "status": "NORMAL",
            "message": "Prediction is reliable",
            "prediction": float(prediction_value)
        }
