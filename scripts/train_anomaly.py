import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import os

def load_and_merge_data():
    base_path = "data/processed"
    files = {
        'TN': 'TN_processed.csv',
        'TP': 'TP_processed.csv',
        'NH3': 'NH3_processed.csv',
        'NO23': 'N023_processed.csv', # Note the filename N023 vs NO23
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
        # Rename value column to parameter name if it's not already
        # The files typically have 'dateTime', 'code', 'value' or 'parameterName'
        # Let's check columns. Based on previous view of TN_processed.csv, cols are: dateTime, code, TN
        # So we can just drop 'code' and merge on 'dateTime'
        
        if 'code' in df.columns:
            df = df.drop(columns=['code'])
            
        # Ensure dateTime is datetime object
        df['dateTime'] = pd.to_datetime(df['dateTime'])
        df = df.set_index('dateTime')
        dfs.append(df)
    
    # Merge all dataframes
    merged_df = pd.concat(dfs, axis=1, join='inner')
    merged_df = merged_df.reset_index()
    return merged_df

def train_anomaly_model():
    print("Loading and merging data...")
    df = load_and_merge_data()
    
    if df is None or df.empty:
        print("Failed to load data.")
        return

    features = ['TN', 'TP', 'NH3', 'NO23', 'OP', 'SSC']
    
    # Check if all features exist
    missing_cols = [col for col in features if col not in df.columns]
    if missing_cols:
        print(f"Missing columns: {missing_cols}")
        # Identify mapping issues.
        # N023 file might contain column 'NO23' or 'N023'. 
        print(f"Available columns: {df.columns}")
        return

    X = df[features]
    print(f"Training data shape: {X.shape}")
    
    # Train Isolation Forest
    print("Training Isolation Forest (contamination=0.05)...")
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    iso_forest.fit(X)
    
    # Save model
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "anomaly_isolation_forest.pkl")
    joblib.dump(iso_forest, model_path)
    print(f"Model saved to {model_path}")
    
    # Verification
    print("Verifying model...")
    test_prediction = iso_forest.predict(X.iloc[0:5])
    print(f"Test predictions (1=Normal, -1=Anomaly): {test_prediction}")

if __name__ == "__main__":
    train_anomaly_model()
