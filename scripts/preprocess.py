# Import Libraries
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

# Set plotting style
sns.set(style="whitegrid")

def load_data(tickers, start_date, end_date):
    """Load historical stock data from Yahoo Finance."""
    data = yf.download(tickers, start=start_date, end=end_date)
    return data

def clean_data(data):
    """Clean the data by forward-filling missing values."""
    print("Missing values before cleaning:\n", data.isnull().sum())
    data.fillna(method='ffill', inplace=True)
    print("Missing values after cleaning:\n", data.isnull().sum())
    return data

def plot_closing_prices(data, tickers):
    """Plot the closing prices of the specified tickers."""
    plt.figure(figsize=(14, 7))
    for ticker in tickers:
        plt.plot(data['Close'][ticker], label=ticker)
    plt.title("Closing Prices of TSLA, BND, SPY")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.show()

def plot_daily_returns(data, tickers):
    """Calculate and plot daily returns for the specified tickers."""
    daily_returns = data['Close'].pct_change().dropna()
    plt.figure(figsize=(14, 7))
    for ticker in tickers:
        plt.plot(daily_returns[ticker], label=ticker)
    plt.title("Daily Returns of TSLA, BND, SPY")
    plt.xlabel("Date")
    plt.ylabel("Daily Return")
    plt.legend()
    plt.show()
    return daily_returns

def summary_statistics(daily_returns):
    """Print summary statistics of daily returns."""
    print("Summary Statistics of Daily Returns:\n", daily_returns.describe())

def plot_rolling_volatility(daily_returns, tickers):
    """Plot rolling volatility (30-day standard deviation) for the specified tickers."""
    plt.figure(figsize=(14, 7))
    for ticker in tickers:
        rolling_std = daily_returns[ticker].rolling(window=30).std()
        plt.plot(rolling_std, label=f'{ticker} Rolling Std Dev (30 days)')
    plt.title("30-Day Rolling Volatility (Standard Deviation)")
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.legend()
    plt.show()

def decompose_seasonality(data, ticker):
    """Decompose the time series to analyze trend and seasonality for a specific ticker."""
    decomposition = seasonal_decompose(data['Close'][ticker].dropna(), model='multiplicative', period=365)
    fig = decomposition.plot()
    fig.set_size_inches(14, 10)
    plt.show()

def plot_outlier_detection(daily_returns):
    """Plot a boxplot to analyze outliers in daily returns."""
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=daily_returns)
    plt.title("Outlier Analysis in Daily Returns for TSLA, BND, SPY")
    plt.ylabel("Daily Return")
    plt.show()

def plot_daily_percentage_change(data, tickers):
    """Plot daily percentage change for the specified tickers."""
    daily_percent_change = data['Close'].pct_change() * 100  # Convert to percentage
    plt.figure(figsize=(14, 7))
    for ticker in tickers:
        plt.plot(daily_percent_change[ticker], label=f'{ticker} % Change')
    plt.title("Daily Percentage Change (Volatility)")
    plt.xlabel("Date")
    plt.ylabel("Daily Percentage Change (%)")
    plt.legend()
    plt.show()

def plot_high_low_range(data, tickers):
    """Plot the high-low percentage range for the specified tickers."""
    high_low_range = (data['High'] - data['Low']) / data['Low'] * 100
    plt.figure(figsize=(14, 7))
    for ticker in tickers:
        plt.plot(high_low_range[ticker], label=f'{ticker} High-Low Range (%)')
    plt.title("High-Low Percentage Range Over Time")
    plt.xlabel("Date")
    plt.ylabel("High-Low Range (%)")
    plt.legend()
    plt.show()

def print_key_insights():
    """Print insights into stock performance and volatility."""
    print("Key Insights from EDA:")
    print("- Tesla (TSLA) shows higher volatility compared to BND and SPY.")
    print("- BND displays stability with lower fluctuations, providing a cushion during volatile periods.")
    print("- Seasonal patterns in Tesla indicate periodic upward and downward movements.")
    print("- Outliers were detected in daily returns, especially in TSLA, signaling high volatility days.")

