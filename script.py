import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

# Download historical stock data
def get_stock_data(stock_symbols, start_date, end_date):
    data = {}
    for symbol in stock_symbols:
        stock_data = yf.download(symbol, start=start_date, end=end_date)
        data[symbol] = stock_data[['Open']]
    return data

# Prepare the data for the LSTM model
def prepare_data(data, time_steps):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(time_steps, len(data_scaled)):
        X.append(data_scaled[i - time_steps:i, 0])
        y.append(data_scaled[i, 0])

    X, y = np.array(X), np.array(y)
    return X, y, scaler

# Build and train the LSTM model
def build_and_train_model(X_train, y_train):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    return model

# Predict and calculate accuracy
def predict_and_evaluate(model, X_test, y_test, scaler):
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
    mse = mean_squared_error(y_test, predictions)
    return predictions, y_test, mse

# Visualize results
def visualize_predictions(actual, predicted, stock_symbol):
    plt.figure(figsize=(14, 5))

    # Main plot: full range
    plt.subplot(1, 2, 1)
    plt.plot(actual, color='red', label='Actual Opening Price')
    plt.plot(predicted, color='blue', label='Predicted Opening Price')
    plt.title(f'Stock Price Prediction for {stock_symbol} (Full Range)')
    plt.xlabel('Time')
    plt.ylabel('Opening Price')
    plt.legend()

    # Zoomed-in plot: focus on differences
    plt.subplot(1, 2, 2)
    plt.plot(actual, color='red', label='Actual Opening Price')
    plt.plot(predicted, color='blue', label='Predicted Opening Price')
    plt.ylim([min(min(actual), min(predicted)) * 0.95, max(max(actual), max(predicted)) * 1.05])  # Tight scale
    plt.title(f'Stock Price Prediction for {stock_symbol} (Zoomed-In)')
    plt.xlabel('Time')
    plt.ylabel('Opening Price')
    plt.legend()

    plt.tight_layout()
    plt.show()