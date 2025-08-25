import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from model_builder import build_lstm_model

def create_sequences(data, seq_length):
    """Convert array into overlapping sequences for LSTM."""
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length, 0]  # predict next close price
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def train_model(df, features, seq_length=60, epochs=10, batch_size=32):
    """
    Train an LSTM model for stock price prediction.
    
    Args:
        df (DataFrame): Stock dataframe with features
        features (list): Feature columns to use
        seq_length (int): Length of sequence (timesteps)
        epochs (int): Training epochs
        batch_size (int): Batch size
    Returns:
        model, scaler
    """
    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[features].values)

    # Build sequences
    X, y = create_sequences(scaled_data, seq_length)

    # Build model
    model = build_lstm_model((X.shape[1], X.shape[2]))

    # Train with early stopping
    es = EarlyStopping(monitor="loss", patience=3, restore_best_weights=True)
    model.fit(X, y, batch_size=batch_size, epochs=epochs, callbacks=[es], verbose=1)

    return model, scaler
