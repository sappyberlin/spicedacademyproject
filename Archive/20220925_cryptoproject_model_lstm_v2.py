# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 15:24:14 2022

@author: ssn50
"""
import pandas as pd
import numpy as np
import math
import datetime as dt
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score 
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import backend as K
#import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, RepeatVector, TimeDistributed, LSTM

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

bitcoin_data["close_lag1"] = bitcoin_data["Close"].shift(1)
bitcoin_data["pct_change"] = ((bitcoin_data["Close"] - bitcoin_data["close_lag1"])/bitcoin_data["close_lag1"])*100
bitcoin_data["pct_change_lag1"] = bitcoin_data["pct_change"].shift(1)
bitcoin_data["pct_diff_high_low"] = ((bitcoin_data["High"] - bitcoin_data["Low"])/bitcoin_data["Low"])*100
bitcoin_data["pct_diff_high_low_lag1"] = bitcoin_data["pct_diff_high_low"].shift(1)
bitcoin_data["running_max"] = bitcoin_data["Close"].cummax()
bitcoin_data["ma_pct_change"] = bitcoin_data["pct_change_lag1"].rolling(30).mean()
# Doesn't seem to help
bitcoin_data["pct_new_max_diff"] = ((bitcoin_data["running_max"].diff())/bitcoin_data["Close"])*100
bitcoin_data["pct_new_max_diff_lag1"] = bitcoin_data["pct_new_max_diff"].shift(1)
bitcoin_data["pct_diff_close_vs_max"] = ((bitcoin_data["running_max"] - bitcoin_data["Close"])/ bitcoin_data["Close"])*100
# Moving Average

# Good overview: https://colah.github.io/posts/2015-08-Understanding-LSTMs/





# convert an array of values into a dataset matrix, because of how the model needs the data to be inputted
def create_lstm_dataset(dataframe, y_col:list, x_col:list ,n_steps_in, n_steps_out):
    columns_to_use = x_col + y_col
    sequences = np.array(dataframe.loc[:, columns_to_use])
    X_out = []
    y_out = []
    
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out-1
  		# check if we are beyond the dataset
        if out_end_ix > len(sequences):
            break
  		# gather input and output parts of the pattern
        seq_x = sequences[i:end_ix, :-1]
        seq_y = sequences[end_ix-1:out_end_ix, -1]
        X_out.append(seq_x)
        y_out.append(seq_y)
        
    return np.array(X_out), np.array(y_out)



def current_lstm_model(n_steps_in, n_steps_out, x_col):
    # Current Model
    K.clear_session()
    base_model = Sequential()
    base_model.add(LSTM(units = 100, return_sequences = True, input_shape = (n_steps_in, len(x_col)), dropout = 0.4 ))
    # Check line with Activation is correct
    base_model.add(LSTM(50, activation='relu', dropout = 0.4))
    base_model.add(Dense(n_steps_out))
    # Check line with Activation is correct
    #base_model.add(Activation('linear'))
    base_model.compile(loss = "mse", optimizer = "adam")
    base_model.summary()
    
    history = base_model.fit(x, y, epochs = 25, batch_size = 32, verbose = 1)
    return history

# Model from https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/; "Multiple Parallel Input and Multi-Step Output"
def new_lstm_model(n_steps_in, n_steps_out, x_col):
    # Current Model
    K.clear_session()
    model = Sequential()
    model.add(LSTM(200, activation='relu', input_shape=(n_steps_in, len(x_col))))
    model.add(RepeatVector(n_steps_out))
    model.add(LSTM(200, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(len(x_col))))
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    
    history = model.fit(x, y, epochs = 25, batch_size = 32, verbose = 1)
    return history

recent_bitcoin_data = bitcoin_data.reset_index().loc[bitcoin_data.reset_index().loc[:,"Date"] > "2015-01-01", :]
# plt.plot(bitcoin_data.index, bitcoin_data["pct_change"])


n_steps_in= 30 
n_steps_out= 7
x_col = ["pct_change_lag1", "Volume", "pct_diff_high_low_lag1", "pct_diff_close_vs_max", "ma_pct_change"]
y_col = ["pct_change"]

x, y = create_lstm_dataset(recent_bitcoin_data.dropna(), y_col = y_col, x_col = x_col, n_steps_in = n_steps_in, n_steps_out = n_steps_out)
model_history = new_lstm_model(n_steps_in = n_steps_in, n_steps_out = n_steps_out, x_col = x_col)

recent_bitcoin_data["pct_change"].std()
14.8**0.5

plt.plot(model_history.history["loss"], label="train data")
# plt.plot(history.history["val_loss"], label="test data")
plt.title("Error")
plt.xlabel("Epoch")
plt.legend()


