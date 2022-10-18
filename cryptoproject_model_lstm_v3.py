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
from tensorflow.keras.layers import Dense, Dropout, Activation, RepeatVector, TimeDistributed, LSTM, GRU


# Good overview: https://colah.github.io/posts/2015-08-Understanding-LSTMs/


def dataframe_prep(crypto_df):
    
    crypto_df = crypto_df.sort_values("Date", ascending = True)
    # Normalazing the Volume
    crypto_df["volume_percent"] = crypto_df["Volume"]/crypto_df["Market Cap"]
    # Creating a lag value, in order to see previous value vs current value
    crypto_df["close_lag1"] = crypto_df["Close"].shift(1)
    # Looking at pct_change per day rather than Absolute value
    crypto_df["pct_change"] = ((crypto_df["Close"] - crypto_df["close_lag1"])/crypto_df["close_lag1"])*100
    crypto_df["pct_change_lag1"] = crypto_df["pct_change"].shift(1)
    # Creating a variable to get a sense of the volatility
    crypto_df["pct_diff_high_low"] = ((crypto_df["High"] - crypto_df["Low"])/crypto_df["Low"])*100
    crypto_df["pct_diff_high_low_lag1"] = crypto_df["pct_diff_high_low"].shift(1)
    # Moving Average
    crypto_df["ma_pct_change_lag1"] = crypto_df["pct_change_lag1"].rolling(14).mean()
    
    return crypto_df.dropna()
    


# convert an array of values into a dataset matrix, because of how the model needs the data to be inputted
def create_lstm_dataset(dataframe, n_steps_in, n_steps_out):    
    # columns_to_use = x_col + y_col
    sequences_x = np.array(dataframe.loc[:, ["ma_pct_change_lag1", "Volume", "pct_diff_high_low_lag1", "pct_change_lag1"]])
    sequences_y = np.array(dataframe.loc[:, ["pct_change"]])
    
    # Due to nature of our forecasting problem, we are always fit_transforming the past data
    scaler = MinMaxScaler(feature_range=(0,1))
    sequences_x = scaler.fit_transform(sequences_x)
        
    X_out = []
    y_out = []
    
    for i in range(len(sequences_x)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out-1
  		# check if we are beyond the dataset
        if out_end_ix > len(sequences_x):
            break
  		# gather input and output parts of the pattern
        seq_x = sequences_x[i:end_ix, :]
        seq_y = sequences_y[end_ix-1:out_end_ix, :]
        X_out.append(seq_x)
        y_out.append(seq_y)
        
    return np.array(X_out), np.array(y_out)


def lstm_model(n_steps_in, n_steps_out, train_x, train_y, epochs_to_use):
    # Need to clear backend before setting up any model
    K.clear_session()
    model = Sequential()
    #model.add(LSTM(units = 128, input_shape = (n_steps_in, train_x.shape[2]), dropout = 0.0, return_sequences = True)) # return_sequences = True
    #model.add(LSTM(units = 64, activation='relu', dropout = 0.0, return_sequences = True))
    #model.add(LSTM(units = 64, activation='relu', dropout = 0.0, return_sequences = True))
    model.add(LSTM(units = 64, activation='relu', dropout = 0.0))
    # model.add(Dense(50))
    model.add(Dense(n_steps_out))
    model.compile(loss = "mse", optimizer = "adam")
    # model.summary()
    
    history = model.fit(train_x, train_y, epochs = epochs_to_use, batch_size = 20, verbose = 1) #,validation_data=(test_x, test_y))
    
    return history, model


# Add stuff
def return_last_pred(crypto_df, x_test, model, n_steps_out):
    yhat_test = model.predict(x_test, verbose=0)
    # COde for very last date only
    last_pred = yhat_test[-1,:].reshape(1,-1)
    last_row = crypto_df.iloc[-1, [0, 4]].to_frame().T
    
    for i in range(0, n_steps_out):
        #running total = 0
        col_name = "percent_pred_" + str(i+1)
        # col_close_name = "Close_price "+ str(i+1)
        print(col_name)
        # print(yhat_test[:, i].shape)
        last_row[col_name] = 1 + last_pred[:, i]/100
    
    for i in range(0, n_steps_out):
        col_price_name = last_row.iloc[0,0] + pd.Timedelta(days = i+1)
        print(col_price_name)
        last_row[col_price_name] = last_row.drop("Date", axis = 1).iloc[:, 0:i+2].prod(axis = 1)
        
        
    return pd.DataFrame({"Date" :last_row.iloc[:, -14:].columns, "Close": last_row.iloc[0, -14:] }).reset_index(drop = True)