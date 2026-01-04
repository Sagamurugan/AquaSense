"""
LSTM Model for time series prediction
"""

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam


class LSTMModel:
    """LSTM model for water quality prediction"""
    
    def __init__(self, lookback=12, features=6):
        self.lookback = lookback
        self.features = features
        self.model = None
    
    def build_model(self):
        """Build LSTM architecture"""
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
