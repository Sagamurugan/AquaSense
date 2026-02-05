import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

def load_and_merge_data():
    base_path = "data/processed"
    files = {
        'TN': 'TN_processed.csv',
        'TP': 'TP_processed.csv',
        'NH3': 'NH3_processed.csv',
        'NO23': 'N023_processed.csv',
        'OP': 'OP_processed.csv',
        'SSC': 'SSC_processed.csv'
    }
    
    dfs = []
    for name, filename in files.items():
        path = os.path.join(base_path, filename)
        if not os.path.exists(path):
            print(f"Error: {path} not found.")
            return None
        
        df = pd.read_csv(path)
        if 'code' in df.columns:
            df = df.drop(columns=['code'])
        df['dateTime'] = pd.to_datetime(df['dateTime'])
        df = df.set_index('dateTime')
        dfs.append(df)
    
    merged_df = pd.concat(dfs, axis=1, join='inner')
    merged_df = merged_df.reset_index()
    return merged_df

def train_tp_model():
    print("Loading data...")
    df = load_and_merge_data()
    
    if df is None:
        return

    # Features and Target
    # Input features: TN, NH3, NO23, OP, SSC
    feature_cols = ['TN', 'NH3', 'NO23', 'OP', 'SSC']
    target_col = 'TP'
    
    X = df[feature_cols]
    y = df[target_col]
    
    print(f"Data Shape: {X.shape}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train
    print("Training Random Forest Regressor for TP...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    
    # Predict & Evaluate
    y_pred = rf_model.predict(X_test_scaled)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print("-" * 30)
    print(f"Model Evaluation (TP Prediction)")
    print(f"RMSE: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    print("-" * 30)
    
    # Save
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    joblib.dump(rf_model, os.path.join(models_dir, "tp_random_forest_model.pkl"))
    joblib.dump(scaler, os.path.join(models_dir, "tp_scaler.pkl"))
    
    print("Model and Scaler saved.")
    
    # Verify strict prohibition: Do NOT use TP as input to predict TP
    print(f"Features used: {feature_cols}")

if __name__ == "__main__":
    train_tp_model()
