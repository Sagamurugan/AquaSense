"""
Attention-based Model for time series prediction
"""

import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, MultiHeadAttention, LayerNormalization
from tensorflow.keras.optimizers import Adam


class AttentionModel:
    """Attention-based model for water quality prediction"""
    
    def __init__(self, lookback=12, features=6):
        self.lookback = lookback
        self.features = features
        self.model = None
    
    def build_model(self):
        """Build Attention-based architecture"""
        inputs = Input(shape=(self.lookback, self.features))
        
        lstm_out = LSTM(50, activation='relu', return_sequences=True)(inputs)
        lstm_out = Dropout(0.2)(lstm_out)
        
        attention_out = MultiHeadAttention(num_heads=4, key_dim=50)(lstm_out, lstm_out)
        attention_out = LayerNormalization()(attention_out + lstm_out)
        
        lstm_out2 = LSTM(25, activation='relu')(attention_out)
        lstm_out2 = Dropout(0.2)(lstm_out2)
        
        outputs = Dense(1)(lstm_out2)
        
        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return self.model
    
    def train(self, X_train, y_train, epochs=50, batch_size=32, validation_split=0.2):
        """Train the attention model"""
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
