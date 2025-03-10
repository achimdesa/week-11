# Import Libraries
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def load_data(assets, start_date, end_date):
    """Load historical closing prices for the specified assets."""
    data = yf.download(assets, start=start_date, end=end_date)['Close']
    return data.dropna()

def calculate_daily_returns(data):
    """Calculate daily returns from the asset price data."""
    return data.pct_change().dropna()

def convert_forecasted_returns(forecasted_returns):
    """Convert forecasted annual returns to daily returns."""
    return {asset: (1 + forecasted_returns[asset])**(1/252) - 1 for asset in forecasted_returns}

def portfolio_performance(weights, mean_returns, cov_matrix):
    """Calculate the expected return and volatility of the portfolio."""
    returns = np.sum(mean_returns * weights) * 252
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    return returns, volatility

def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.02):
    """Calculate the negative Sharpe Ratio for optimization."""
    p_return, p_volatility = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(p_return - risk_free_rate) / p_volatility

def optimize_portfolio(assets, forecasted_daily_returns, cov_matrix):
    """Optimize the portfolio to maximize the Sharpe Ratio."""
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(len(assets)))
    initial_weights = [1 / len(assets) for _ in range(len(assets))]
    
    optimal = minimize(neg_sharpe_ratio, initial_weights, args=(list(forecasted_daily_returns.values()), cov_matrix), 
                       method='SLSQP', bounds=bounds, constraints=constraints)
    
    return optimal.x

def plot_portfolio_allocation(weights, assets):
    """Plot the optimized portfolio allocation."""
    plt.figure(figsize=(10, 6))
    plt.pie(weights, labels=assets, autopct='%1.1f%%', startangle=140)
    plt.title('Optimized Portfolio Allocation')
    plt.show()

def backtest_portfolio(returns, optimal_weights):
    """Analyze portfolio performance over time through historical backtesting."""
    cumulative_returns = (returns + 1).cumprod()
    portfolio_returns = cumulative_returns.dot(optimal_weights)

    plt.figure(figsize=(14, 7))
    plt.plot(cumulative_returns, label=returns.columns)
    plt.plot(portfolio_returns, label='Optimized Portfolio', color='black', linewidth=2)
    plt.legend(loc='upper left')
    plt.title("Cumulative Returns of Optimized Portfolio vs. Individual Assets")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.show()