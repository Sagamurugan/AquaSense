"""
Main script for AquaSense water quality prediction system
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from preprocess import load_data, clean_data, scale_features
from feature_engineering import create_sequences, engineer_features
from model_lstm import LSTMModel
from model_attention import AttentionModel
from anomaly_detection import AnomalyDetector
from utils import calculate_metrics, save_predictions, save_metrics, save_scaler


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='AquaSense Water Quality Prediction')
    parser.add_argument('--model', choices=['lstm', 'attention'], default='lstm',
                       help='Model type to use')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--predict', action='store_true', help='Make predictions')
    parser.add_argument('--data', type=str, default='data/processed/merged_dataset.csv',
                       help='Path to data file')
    
    args = parser.parse_args()
    
    print("🌊 AquaSense - Water Quality Prediction System")
    print(f"Using model: {args.model}")
    
    # Load and preprocess data
    print("Loading data...")
    # df = load_data(args.data)
    # df = clean_data(df)
    # df = engineer_features(df)
    
    print("Data preprocessing complete")
    
    if args.train:
        print("Training model...")
        # Training logic here
        pass
    
    if args.predict:
        print("Making predictions...")
        # Prediction logic here
        pass
    
    print("Done!")


if __name__ == "__main__":
    main()
