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
