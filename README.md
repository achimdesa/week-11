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



# Task 2: Time Series Forecasting for Tesla's Stock Price

## Project Overview
This notebook covers Task 2 of the Week 11 challenge, focusing on developing time series forecasting models to predict Tesla's future stock prices. Models such as **ARIMA**, **SARIMA**, and **LSTM** are used to create and evaluate forecasts.

## Data Description
Data includes historical closing prices for **Tesla (TSLA)** from **January 1, 2015, to October 31, 2024**, obtained using the YFinance Python library.

## Analysis Steps

### 1. Model Selection and Data Splitting
- **Model Choices**:
  - **ARIMA**: Suitable for non-seasonal data.
  - **SARIMA**: Extends ARIMA to handle seasonality.
  - **LSTM**: A recurrent neural network capable of capturing long-term dependencies.
- **Data Splitting**: Divide data into training (80%) and testing (20%) sets.

### 2. Model Training and Optimization
- **ARIMA** and **SARIMA**: Use `auto_arima` to determine optimal parameters.
- **LSTM**: Configure an LSTM network with dropout layers to prevent overfitting.

### 3. Model Evaluation
- Calculate evaluation metrics: **Mean Absolute Error (MAE)**, **Root Mean Squared Error (RMSE)**, and **Mean Absolute Percentage Error (MAPE)**.

### 4. Model Saving
- Save the trained models to the `../models` directory for later use in Task 3.

## Getting Started

### Prerequisites
Ensure Python 3.8+ and the following packages are installed:
```bash
pip install yfinance pandas numpy matplotlib statsmodels pmdarima tensorflow joblib

Running the Notebook
To run Task 2, open task-2.ipynb in a Jupyter notebook environment and execute each cell sequentially.

Results
Model Accuracy: Evaluate the models based on MAE, RMSE, and MAPE.
Saved Models: Trained models (ARIMA, SARIMA, and LSTM) are saved under ../models.
Next Steps
Proceed to Task 3 to use the saved models for forecasting Tesla's future stock prices.


```markdown
# Task 3: Forecast Future Market Trends for Tesla Stock

## Project Overview
This notebook covers Task 3, where the saved models from Task 2 are used to generate forecasts for Tesla's stock price over the next 6-12 months. This helps in analyzing future trends, potential risks, and volatility.

## Data Description
Data is loaded dynamically for **Tesla (TSLA)** from **January 1, 2015, to October 31, 2024** via YFinance, and forecasts are generated for an additional 6-12 months.

## Forecasting Steps

### 1. Load Models and Forecast
- **Select Model**: Use the saved **ARIMA**, **SARIMA**, or **LSTM** model from Task 2.
- **Generate Forecast**: Predict Tesla's stock prices for 6-12 months.

### 2. Forecast Analysis
- **Visualize Forecast**: Plot forecasted prices alongside historical data.
- **Confidence Intervals**: Show forecast uncertainty to assess risk.

### 3. Interpretation of Results
- **Trend Analysis**: Identify any upward, downward, or stable trends in the forecast.
- **Volatility and Risk**: Use confidence intervals to analyze risk and volatility.
- **Opportunities and Risks**: Highlight potential market opportunities and risks based on the forecast.

## Getting Started

### Prerequisites
Ensure Python 3.8+ and the required packages are installed:
```bash
pip install yfinance pandas numpy matplotlib statsmodels tensorflow joblib

Running the Notebook
To run Task 3, open task-3.ipynb in a Jupyter notebook environment and execute each cell.

Results
Forecasted Trends: Predicted trends and volatility insights for Tesla stock prices.
Risk Analysis: Confidence intervals highlight potential areas of high volatility.
Next Steps
Use the forecasted data in Task 4 to optimize the portfolio based on predicted trends.


```markdown
# Task 4: Portfolio Optimization Based on Forecast

## Project Overview
This notebook covers Task 4, which utilizes the forecasted data from Task 3 to optimize a portfolio containing **Tesla (TSLA)**, **Vanguard Total Bond Market ETF (BND)**, and **S&P 500 ETF (SPY)**. The objective is to adjust asset allocations to maximize returns and minimize risks based on forecasted trends.

## Data Description
Historical data for **TSLA**, **BND**, and **SPY** is fetched via YFinance, covering **January 1, 2015, to October 31, 2024**. Forecasted returns from Task 3 are used as expected returns.

## Portfolio Optimization Steps

### 1. Define Forecasted Returns and Calculate Covariance Matrix
- **Forecasted Returns**: Use forecasted returns for TSLA from Task 3 and approximate returns for BND and SPY.
- **Covariance Matrix**: Calculate to understand asset return correlations.

### 2. Portfolio Optimization
- **Objective**: Maximize the Sharpe Ratio.
- **Optimization Method**: Use `scipy.optimize.minimize` with constraints and bounds to find optimal portfolio weights.

### 3. Portfolio Analysis
- **Expected Return and Volatility**: Calculate based on optimized weights.
- **Sharpe Ratio**: Assess risk-adjusted return.
- **Backtesting**: Analyze cumulative returns of the optimized portfolio vs. individual assets.

## Getting Started

### Prerequisites
Ensure Python 3.8+ and the following packages are installed:
```bash
pip install yfinance pandas numpy matplotlib scipy


Running the Notebook
To run Task 4, open task-4.ipynb in a Jupyter notebook environment and execute each cell.

Results
Optimal Portfolio Weights: Allocation that maximizes the Sharpe Ratio.
Portfolio Performance: Expected return, volatility, and Sharpe Ratio based on optimized weights.
Backtest Analysis: Cumulative returns compared with individual assets.