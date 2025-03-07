# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMAResults
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import joblib

def load_data(ticker, start_date, end_date):
    """Load historical stock data for the given ticker from Yahoo Finance."""
    data = yf.download(ticker, start=start_date, end=end_date)
    return data['Close']

def plot_forecast(data, predictions, confidence_intervals=None, title="Forecast"):
    """Plot forecast with confidence intervals."""
    plt.figure(figsize=(14, 7))
    plt.plot(data, label='Historical Data')
    plt.plot(predictions.index, predictions, label='Forecast', color='orange')
    if confidence_intervals is not None:
        plt.fill_between(predictions.index, 
                         confidence_intervals.iloc[:, 0], 
                         confidence_intervals.iloc[:, 1], color='pink', alpha=0.3)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

def arima_forecast(model, forecast_horizon):
    """Forecast using ARIMA model."""
    arima_forecast = model.get_forecast(steps=forecast_horizon)
    predictions = arima_forecast.predicted_mean
    confidence_intervals = arima_forecast.conf_int()
    return predictions, confidence_intervals

def sarima_forecast(model, forecast_horizon):
    """Forecast using SARIMA model."""
    sarima_forecast = model.get_forecast(steps=forecast_horizon)
    predictions = sarima_forecast.predicted_mean
    confidence_intervals = sarima_forecast.conf_int()
    return predictions, confidence_intervals

def lstm_forecast(model, last_data, forecast_horizon):
    """Forecast using LSTM model."""
    input_sequence = TimeseriesGenerator(last_data, last_data, length=60, batch_size=1)
    lstm_forecast = []
    current_batch = last_data[-60:]
    for _ in range(forecast_horizon):
        current_pred = model.predict(current_batch.reshape(1, 60, 1))[0]
        lstm_forecast.append(current_pred)
        current_batch = np.append(current_batch[1:], current_pred)
    return lstm_forecast