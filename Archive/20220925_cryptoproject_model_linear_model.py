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
bitcoin_data = bitcoin_data.set_index("Date").sort_index(ascending = True)
bitcoin_data["volume_percent"] = bitcoin_data["Volume"]/bitcoin_data["Market Cap"]

bitcoin_data = bitcoin_data.sort_index(ascending = True)
for i in range(1,15):
    lag_name = f"close_lag_{i}"
    bitcoin_data[lag_name] = bitcoin_data["Close"].shift(i)

# I want to split the data, training on pre 2020, because I want the model to be able to predict the next wave
base_model_columns = ['Close', 'close_lag_1', 'close_lag_2', 'close_lag_3',
                       'close_lag_4', 'close_lag_5', 'close_lag_6', 'close_lag_7',
                       'close_lag_8', 'close_lag_9', 'close_lag_10', 'close_lag_11',
                       'close_lag_12', 'close_lag_13', 'close_lag_14']
train_data = bitcoin_data.copy().loc[bitcoin_data.index < "2020-01-01", base_model_columns].dropna()
test_data = bitcoin_data.copy().loc[bitcoin_data.index >= "2020-01-01", base_model_columns]