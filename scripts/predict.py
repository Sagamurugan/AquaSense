import numpy as np
import joblib
import os


BASE_DIR = os.path.dirname(os.path.dirname(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "tn_random_forest_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "tn_scaler.pkl")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

def predict_tn(tp, nh3, no23, op):
    data = np.array([[tp, nh3, no23, op]])
    
    scaled = scaler.transform(data)

    prediction = model.predict(scaled)

    return float(prediction[0][0])


