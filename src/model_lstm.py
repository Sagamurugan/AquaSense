"""
LSTM Model for time series prediction

EDUCATIONAL NOTE:
-----------------
1. Time Series Data: 
   Data points collected or recorded at specific time intervals (e.g., daily water quality measurements).
   The order of data points matters significantly.

2. Temporal Dependency:
   The concept that current values are influenced by past values (e.g., Nitrogen levels today might depend on levels from the past week).
   LSTMs (Long Short-Term Memory networks) are designed to capture these dependencies over long sequences.

3. Window Size (Lookback):
   The number of past time steps the model looks at to predict the next step.
   If lookback=12, the model uses the past 12 days of data to predict day 13.

4. Epoch:
   One complete pass through the entire training dataset. 
   Too few epochs = underfitting (model doesn't learn).
   Too many epochs = overfitting (model memorizes noise).

PERFORMANCE CONTEXT:
--------------------
Why LSTM might underperform Random Forest on this dataset:
- Data Size: Deep Learning models like LSTM require massive amounts of data to generalize well. If the dataset is small (< 10,000 samples), simpler models like Random Forest often perform better.
- Feature Independence: If the water quality parameters are more correlated with each other *at the same time* rather than dependent on *past values*, Random Forest (which excels at feature interaction) will beat LSTM.
"""

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam


class LSTMModel:
    """LSTM model for water quality prediction"""
    
    def __init__(self, lookback=12, features=6):
        """
        Initialize LSTM Model
        :param lookback: Window size (number of previous time steps to use)
        :param features: Number of input features (TN, TP, etc.)
        """
        self.lookback = lookback
        self.features = features
        self.model = None
    
    def build_model(self):
        """
        Build LSTM architecture
        
        Structure:
        1. LSTM Layer: Extracts temporal features/patterns from the sequence.
        2. Dropout: Prevents overfitting by randomly turning off neurons.
        3. Dense Layers: Fully connected layers to map features to the final prediction.
        """
        self.model = Sequential([
            LSTM(50, activation='relu', input_shape=(self.lookback, self.features)),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])
        
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return self.model
    
    def train(self, X_train, y_train, epochs=50, batch_size=32, validation_split=0.2):
        """Train the LSTM model"""
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        return history
    
    def predict(self, X_test):
        """Make predictions"""
        return self.model.predict(X_test)
    
    def save(self, filepath):
        """Save model to file"""
        self.model.save(filepath)
    
    def load(self, filepath):
        """Load model from file"""
        from tensorflow.keras.models import load_model
        self.model = load_model(filepath)
