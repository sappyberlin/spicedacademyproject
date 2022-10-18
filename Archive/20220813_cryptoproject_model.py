# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 14:27:15 2022

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
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from keras.callbacks import EarlyStopping

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

# x_train2 = np.expand_dims(x_train, 0)

base_model = Sequential()
base_model.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[0], x_train[1], 1 ) ))
base_model.add(Dense(1))
base_model.compile(loss = "mean_squared_error", optimizer = "adam")
base_model.summary()


base_model_fitted = base_model.fit(x_train2, y_train, epochs = 50) #, validation_data = (x_test, y_test))

y_pred = base_model.predict(x_test)



plt.plot(y_test.index, y_pred)

base_model_fitted.history.keys()
# plt.subplot(1, 2, 1)
plt.plot(base_model_fitted.history["loss"], label="train data")
plt.plot(base_model_fitted.history["val_loss"], label="test data")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.legend()

rmsle = RMSLE()
