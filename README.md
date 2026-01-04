# AquaSense - Water Quality Prediction System

## Project Overview

AquaSense is a comprehensive water quality monitoring and prediction system that uses advanced machine learning techniques including LSTM and Attention-based neural networks to predict water quality parameters.

## Features

- **Data Processing**: Automated data cleaning and preprocessing pipeline
- **Feature Engineering**: Temporal feature creation and sequence generation for time series prediction
- **LSTM Model**: Traditional LSTM architecture for time series forecasting
- **Attention Model**: Advanced attention-based model for improved predictions
- **Anomaly Detection**: Isolation Forest-based anomaly detection in water quality data
- **Dashboard**: Interactive Streamlit dashboard for visualization and monitoring
- **Evaluation Metrics**: Comprehensive model evaluation with multiple metrics (MSE, MAE, RMSE, R²)

## Project Structure

```
AquaSense/
├── data/                          # Data directory
│   ├── raw/                       # Raw data files
│   │   ├── TN.csv                # Total Nitrogen measurements
│   │   ├── TP.csv                # Total Phosphorus measurements
│   │   ├── TN_STRF.csv           # Total Nitrogen stratified
│   │   ├── OP_STRF.csv           # Orthophosphate stratified
│   │   ├── NH3_STRF.csv          # Ammonia stratified
│   │   └── SSC_STRF.csv          # Suspended Sediment Concentration
│   ├── processed/                # Processed data
│   │   └── merged_dataset.csv    # Merged and cleaned dataset
│   └── README.md
│
├── notebooks/                     # Jupyter notebooks
│   ├── 01_data_cleaning.ipynb
│   ├── 02_exploratory_analysis.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training_LSTM.ipynb
│   ├── 05_attention_model_training.ipynb
│   └── 06_evaluation.ipynb
│
├── src/                           # Source code
│   ├── __init__.py
│   ├── preprocess.py              # Data preprocessing functions
│   ├── feature_engineering.py     # Feature creation and transformation
│   ├── model_lstm.py              # LSTM model implementation
│   ├── model_attention.py         # Attention-based model
│   ├── anomaly_detection.py       # Anomaly detection module
│   └── utils.py                   # Utility functions
│
├── models/                        # Trained models
│   ├── lstm_model.h5
│   ├── attention_model.h5
│   └── scaler.pkl
│
├── dashboard/                     # Streamlit dashboard
│   ├── app.py                     # Main dashboard application
│   ├── charts.py                  # Chart generation functions
│   └── assets/                    # Static assets
│
├── results/                       # Results and outputs
│   ├── predictions.csv            # Model predictions
│   ├── metrics.txt                # Evaluation metrics
│   └── plots/                     # Visualization plots
│
├── requirements.txt               # Python dependencies
├── main.py                        # Main entry point
└── README.md                      # This file
```

## Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Data Processing

```python
from src.preprocess import load_data, clean_data, scale_features

df = load_data('data/raw/TN.csv')
df = clean_data(df)
X_scaled, scaler = scale_features(df)
```

### Model Training

```bash
python main.py --train --model lstm --data data/processed/merged_dataset.csv
```

### Making Predictions

```bash
python main.py --predict --model lstm
```

### Dashboard

```bash
streamlit run dashboard/app.py
```

## Models

### LSTM Model

- 2-layer LSTM architecture
- Dropout regularization
- Optimized with Adam optimizer
- Input shape: (lookback, features)

### Attention Model

- LSTM with Multi-Head Attention mechanism
- Layer normalization for stability
- Advanced sequence modeling
- Better capture of long-term dependencies

## Evaluation Metrics

- **MSE** (Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **R²** (Coefficient of Determination)

## Requirements

- Python 3.8+
- TensorFlow 2.14
- Scikit-learn 1.3
- Streamlit 1.28
- Pandas 2.1
- NumPy 1.24

## Contributing

For contributions, please follow the project structure and add appropriate documentation.

## License

This project is licensed under the MIT License.

## Contact

For questions or support, please contact the project maintainer.
