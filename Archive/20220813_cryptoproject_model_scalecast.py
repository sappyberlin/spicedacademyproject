# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 15:55:08 2022

@author: ssn50
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly
import seaborn as sns
# Modules for running the inferential statistics
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from sklearn import preprocessing
from statsmodels.tsa.stattools import pacf
import math
import keras
from scalecast.Forecaster import Forecaster

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
bitcoin_data = bitcoin_data.set_index("Date")

# Confirming the data is non-stationary H0: There is a unit root, hence non-stationary
df_test = adfuller(bitcoin_data.Close, autolag = "AIC")
print(pd.Series(df_test[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used']))
# We fail to reject the null hypothesis, thus we consider the data to be non-stationary

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


forecaster = Forecaster(y = bitcoin_data["Close"], current_dates = bitcoin_data.index)
forecaster.plot()
forecaster.plot_acf()
forecaster.plot_pacf()


forecaster.plot_acf(diffy = True)
forecaster.plot_pacf(diffy = True)

forecaster.seasonal_decompose(diffy=True).plot()





