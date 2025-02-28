# week-11
10 Academy Kifiya AI mastery training program week 11 challenge

# Task 1: Data Preprocessing and Exploratory Data Analysis (EDA)

## Project Overview
This notebook covers Task 1 of the Week 11 challenge, focusing on data preprocessing and exploratory data analysis (EDA) for the three selected assets:
- **Tesla (TSLA)**: High-risk, high-growth stock.
- **Vanguard Total Bond Market ETF (BND)**: Stable, low-risk bond ETF.
- **S&P 500 ETF (SPY)**: Provides broad U.S. market exposure.

The goal is to clean, analyze, and visualize the data to uncover trends, volatility, and seasonality, which will set the foundation for future time series forecasting and portfolio optimization tasks.

## Data Description
Historical data from **January 1, 2015, to October 31, 2024** for TSLA, BND, and SPY includes:
- **Open, High, Low, Close**: Daily prices.
- **Adjusted Close**: Close price adjusted for dividends and splits.
- **Volume**: Number of shares traded daily.

Data was sourced using the YFinance Python library.

## Analysis Steps

### 1. Data Loading and Cleaning
- Download data for TSLA, BND, and SPY.
- Handle missing values by forward-filling.
- Check data types to ensure consistency.

### 2. Exploratory Data Analysis (EDA)
- **Closing Price Trends**: Plot daily closing prices to observe long-term trends.
- **Daily Returns**: Compute and plot daily returns to assess volatility.
- **Volatility Analysis**: Calculate 30-day rolling standard deviation for each asset to measure short-term volatility.
- **Seasonal Decomposition**: Decompose TSLA’s time series into trend, seasonality, and residual components to analyze periodic behavior.
- **Outlier Detection**: Identify significant anomalies in daily returns for each asset using a boxplot.

### 3. Key Insights
- **Tesla (TSLA)** exhibits high volatility, ideal for growth but with increased risk.
- **BND** offers stability with minimal fluctuations, suitable for balancing portfolio risk.
- **SPY** demonstrates moderate volatility, providing diversified exposure to the U.S. market.
- Seasonal patterns are evident in Tesla's price, indicating periodic highs and lows.
- Detected outliers in TSLA highlight days with extreme market movement.

## Getting Started

### Prerequisites
Ensure Python 3.8+ and the required packages are installed:
```bash
pip install yfinance pandas numpy matplotlib seaborn statsmodels

## Running the Notebook
To run Task 1, open `task-1.ipynb` in a Jupyter notebook environment and execute each cell sequentially. This notebook will output visualizations and insights based on the above steps.

## Results
- **Data Preprocessing**: Cleaned data ready for modeling.
- **EDA Findings**: Visual insights into asset trends, volatility, and seasonality.

## Next Steps
This analysis serves as the foundation for building time series forecasting models in Task 2. These models will help predict future values and enhance portfolio management.

## References
- [YFinance Documentation](https://pypi.org/project/yfinance/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [Statsmodels Seasonal Decomposition](https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.seasonal_decompose.html)
