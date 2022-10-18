# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 13:37:56 2022

@author: ssn50
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly
import seaborn as sns
# Modules for running the inferential statistics

from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import sktime
from sktime.utils.plotting import plot_series
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.model_selection import temporal_train_test_split

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
bitcoin_data["volume_percent"] = bitcoin_data["Volume"]/bitcoin_data["Market Cap"]
bitcoin_data = bitcoin_data.set_index("Date").sort_index(ascending = True)
# Key to change index to period for the forecaster to work later on!
bitcoin_data.index = bitcoin_data.index.to_period("D")

# Confirming the data is non-stationary H0: There is a unit root, hence non-stationary
df_test = adfuller(bitcoin_data.Close, autolag = "AIC")
print(pd.Series(df_test[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used']))
# We fail to reject the null hypothesis, thus we consider the data to be non-stationary


# Using sktime
# Need to plot the data in ascending order for plot_series to work!
plot_series(bitcoin_data["Close"].sort_index(ascending = True))

# Creating the forecast horzion that I need
# Check tutorial https://www.sktime.org/en/stable/examples/01_forecasting.html 
# 1.2.5 Probabilistic forecasting: prediction intervals, quantile, variance, and distributional forecasts is very useful
# USE to split data up for use split data Step 1 - Splitting a historical data set in to a temporal train and test batch 


forecast_horizon = ForecastingHorizon(pd.PeriodIndex(pd.date_range(start = str(bitcoin_data.index[-1] + pd.Timedelta(days = 1)),
                                                                   periods = 1000,
                                                                   freq = "D")),
                                      is_relative=False)

"""Code below if you don't want dates"""
# forecast_horizon = ForecastingHorizon(np.arange(1,1000), is_relative = True,)

cutoff = pd.Period("2022-09-19", freq = "D")

forecast_horizon.to_absolute(cutoff)
# forecast_horizon.to_relative(cutoff = cutoff)

# Base Naive Forecasting
naive_forecast = NaiveForecaster(strategy = "last", sp = 1)

# Something funky going on when there is a df with an index
naive_forecast.fit(bitcoin_data["Close"])#.reset_index().loc[:, "Close"] )

y_pred = naive_forecast.predict(fh = forecast_horizon)

plot_series(bitcoin_data["Close"], y_pred, labels=["y", "y_pred"])

type(bitcoin_data.index)



train_data, test_data = temporal_train_test_split(y = bitcoin_data["Close"], test_size = 0.25)







"""
# LSTM model
# Baseline Model, using and 14 day lag (cumulative) for price only
bitcoin_data = bitcoin_data.sort_index(ascending = True)
for i in range(1,15):
    lag_name = f"close_lag_{i}"
    bitcoin_data[lag_name] = bitcoin_data["Close"].shift(i)
    


bitcoin_data.columns
# I want to split the data, training on pre 2020, because I want the model to be able to predict the next wave
base_model_columns = ['Close', 'close_lag_1', 'close_lag_2', 'close_lag_3',
                       'close_lag_4', 'close_lag_5', 'close_lag_6', 'close_lag_7',
                       'close_lag_8', 'close_lag_9', 'close_lag_10', 'close_lag_11',
                       'close_lag_12', 'close_lag_13', 'close_lag_14']
train_data = bitcoin_data.copy().loc[bitcoin_data.index < "2020-01-01", base_model_columns].dropna()
test_data = bitcoin_data.copy().loc[bitcoin_data.index >= "2020-01-01", base_model_columns]

x_train = train_data.drop("Close", axis = 1)
y_train = train_data["Close"]

x_test = test_data.drop("Close", axis = 1)
y_test = test_data["Close"]

# Baseline Linear Model using 14 past observations
linear_model = LinearRegression()
linear_model.fit(x_train, y_train)
y_pred = linear_model.predict(x_test)

print('Prediction Score : ' , linear_model.score(x_test, y_test))
error = mean_squared_error(y_test, y_pred)
print('Mean Squared Error : ',error)


plt.plot(x_test.index,y_test, alpha = 0.5)
plt.plot(x_test.index, y_pred, alpha = 0.5)
plt.show()

"""

