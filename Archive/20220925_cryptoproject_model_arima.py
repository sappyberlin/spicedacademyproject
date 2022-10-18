# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 12:02:28 2022

@author: ssn50
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly
import seaborn as sns
# Modules for running the inferential statistics
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from pmdarima.arima import auto_arima
from prophet import Prophet

# Data from https://coincodex.com/crypto/ethereum/historical-data/
bitcoin_data = pd.read_csv("C:\\Users\\ssn50\\spiced_academy\\ginger-pipeline-student-code\\Crypto project\\coincodex_bitcoin_2010-07-23_2022-09-22.csv", parse_dates = ["Date"] )

# Checking data types and accuracy visually
bitcoin_data.describe()
bitcoin_data.info()

plt.plot(bitcoin_data["Date"], bitcoin_data["Close"])
plt.show()
plt.plot(bitcoin_data["Date"], bitcoin_data["Volume"]/bitcoin_data["Market Cap"])
plt.show()

# Adding an extra column to better capture the amount of coin traded on a given day
bitcoin_data = bitcoin_data.set_index("Date").sort_index(ascending = True)
bitcoin_data["volume_percent"] = bitcoin_data["Volume"]/bitcoin_data["Market Cap"]

# Seasonal Decompose
decompose_data_3year = seasonal_decompose(bitcoin_data["Close"], model="additive", period = (365*3))
decompose_data_3year.plot()
decompose_data_1year = seasonal_decompose(bitcoin_data["Close"], model="additive", period = 365*1)
decompose_data_1year.plot()

# Confirming the data is non-stationary H0: There is a unit root, hence non-stationary
df_test = adfuller(bitcoin_data.Close, autolag = "AIC")
print(pd.Series(df_test[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used']))

# Lag difference makes it appear that data is stationary
df_test_lag = adfuller(bitcoin_data.Close.diff().dropna(), autolag = "AIC")
print(pd.Series(df_test_lag[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used']))

# looking at acf and pacf
plot_acf(bitcoin_data.Close, lags=50)
plot_pacf(bitcoin_data.Close, lags=50)

# renaming columns needed for prophet
prophet_data = bitcoin_data.reset_index().loc[:, ["Date", "Close"]].rename({"Date" : "ds", "Close" : "y"}, axis = 1)
base_model = Prophet()
base_model.fit(prophet_data)

future_dates = base_model.make_future_dataframe(periods = 365*3)
base_forecast = base_model.predict(future_dates)
base_model.plot(base_forecast)

# Can see that Visually that the base model isn't very good