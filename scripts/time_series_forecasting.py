# Import Libraries
import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
from pmdarima import auto_arima
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import joblib
import os

# Directory to save models
os.makedirs("../models", exist_ok=True)

def load_data(ticker, start_date, end_date):
    """Load historical stock data for the given ticker from Yahoo Finance."""
    data = yf.download(ticker, start=start_date, end=end_date)
    return data['Close'].dropna()

def split_data(data, train_size_ratio=0.8):
    """Split the data into training and testing sets."""
    train_size = int(len(data) * train_size_ratio)
    train, test = data[:train_size], data[train_size:]
    return train, test

def evaluate_forecast(test, predictions):
    """Calculate and print evaluation metrics for the forecast."""
    mae = mean_absolute_error(test, predictions)
    rmse = sqrt(mean_squared_error(test, predictions))
    mape = np.mean(np.abs((test - predictions) / test)) * 100
    print(f'MAE: {mae}, RMSE: {rmse}, MAPE: {mape}')
    return mae, rmse, mape

def arima_model(train):
    """Fit ARIMA model and make predictions."""
    arima_model = auto_arima(train, seasonal=False, trace=True, error_action='ignore', suppress_warnings=True)
    arima_fit = ARIMA(train, order=arima_model.order).fit()
    joblib.dump(arima_fit, "../models/arima_model.pkl")
    return arima_fit

def plot_predictions(train, test, predictions, title):
    """Plot train, test, and predictions."""
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train, label='Train')
    plt.plot(test.index, test, label='Test')
    plt.plot(test.index, predictions, label='Predictions')
    plt.title(title)
    plt.legend()
    plt.show()

def sarima_model(train):
    """Fit SARIMA model and make predictions."""
    sarima_model = auto_arima(train, seasonal=True, m=12, trace=True, error_action='ignore', suppress_warnings=True)
    sarima_fit = SARIMAX(train, order=sarima_model.order, seasonal_order=sarima_model.seasonal_order).fit(disp=False)
    joblib.dump(sarima_fit, "../models/sarima_model.pkl")
    return sarima_fit

def prepare_lstm_data(train, test, length=60, batch_size=32):
    """Prepare data for LSTM."""
    train_data_gen = TimeseriesGenerator(train.values, train.values, length=length, batch_size=batch_size)
    test_data_gen = TimeseriesGenerator(test.values, test.values, length=length, batch_size=batch_size)
    return train_data_gen, test_data_gen

def build_lstm_model(input_shape):
    """Define and compile LSTM model."""
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model