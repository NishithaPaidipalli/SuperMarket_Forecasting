import numpy as np
import pandas as pd
from preprocess import load_and_clean, create_sequences, MinMaxScaler
from model import build_lstm_model
import os

def train():
    # 1. Load and Preprocess
    print("Loading data...")
    df = load_and_clean('data/supermarket_sales.csv')
    values = df['Total'].values.reshape(-1, 1)
    
    # Scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(values)
    
    # Create Sequences (Using 7 days to predict the 8th)
    SEQ_LENGTH = 7
    X, y = create_sequences(scaled_data, SEQ_LENGTH)
    
    # Reshape X for LSTM: [samples, time steps, features]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    # 2. Build Model
    model = build_lstm_model((SEQ_LENGTH, 1))
    
    # 3. Train
    print("Starting training...")
    # epochs: how many times the model sees the whole dataset
    # batch_size: how many days it looks at before updating itself
    model.fit(X, y, epochs=20, batch_size=16, verbose=1)
    
    # 4. Save the model and the scaler
    if not os.path.exists('models'):
        os.makedirs('models')
        
    model.save('models/sales_model.keras')
    print("Model saved to models/sales_model.keras")

if __name__ == "__main__":
    train()
