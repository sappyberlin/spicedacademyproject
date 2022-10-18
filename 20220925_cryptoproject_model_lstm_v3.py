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

bitcoin_data = pd.read_csv("C:\\Users\\ssn50\\spiced_academy\\ginger-pipeline-student-code\\Crypto project\\data\\Bitcoin.csv", parse_dates = ["Date"] )

# Good walkthrough: https://medium.com/geekculture/lstm-for-bitcoin-prediction-in-python-6e2ea7b1e4e4

# Checking data types and accuracy visually
bitcoin_data.describe()
bitcoin_data.info()

plt.plot(bitcoin_data["Date"], bitcoin_data["Close"])
plt.show()
plt.plot(bitcoin_data["Date"], bitcoin_data["Volume"]/bitcoin_data["Market Cap"])
plt.show()

# Feature Engineering
bitcoin_data = bitcoin_data.sort_values("Date", ascending = True)
# Normalazing the Volume
bitcoin_data["volume_percent"] = bitcoin_data["Volume"]/bitcoin_data["Market Cap"]
# Creating a lag value, in order to see previous value vs current value
bitcoin_data["close_lag1"] = bitcoin_data["Close"].shift(1)
# Looking at pct_change per day rather than Absolute value
bitcoin_data["pct_change"] = ((bitcoin_data["Close"] - bitcoin_data["close_lag1"])/bitcoin_data["close_lag1"])*100
bitcoin_data["pct_change_lag1"] = bitcoin_data["pct_change"].shift(1)
# Creating a variable to get a sense of the volatility
bitcoin_data["pct_diff_high_low"] = ((bitcoin_data["High"] - bitcoin_data["Low"])/bitcoin_data["Low"])*100
bitcoin_data["pct_diff_high_low_lag1"] = bitcoin_data["pct_diff_high_low"].shift(1)
# Moving Average
bitcoin_data["ma_pct_change_lag1"] = bitcoin_data["pct_change_lag1"].rolling(30).mean()

# Doesn't seem to help
bitcoin_data["running_max"] = bitcoin_data["Close"].cummax()
bitcoin_data["pct_new_max_diff"] = ((bitcoin_data["running_max"].diff())/bitcoin_data["Close"])*100
bitcoin_data["pct_new_max_diff_lag1"] = bitcoin_data["pct_new_max_diff"].shift(1)
bitcoin_data["pct_diff_close_vs_max"] = ((bitcoin_data["running_max"] - bitcoin_data["Close"])/ bitcoin_data["Close"])*100

# Good overview: https://colah.github.io/posts/2015-08-Understanding-LSTMs/


def dataframe_prep(crypto_df):
    
    crypto_df = crypto_df.sort_values("Date", ascending = True)
    # Normalazing the Volume
    crypto_df["volume_percent"] = crypto_df["Volume"]/bitcoin_data["Market Cap"]
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

    

    
    return crypto_df
    
test = dataframe_prep(bitcoin_data)

"""
scaler = MinMaxScaler(feature_range=(0,1))
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
"""

# convert an array of values into a dataset matrix, because of how the model needs the data to be inputted
def create_lstm_dataset(dataframe,
                        y_col:list, x_col:list ,n_steps_in, n_steps_out):    
    # columns_to_use = x_col + y_col
    sequences_x = np.array(dataframe.loc[:, x_col])
    sequences_y = np.array(dataframe.loc[:, y_col])
    
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


def lstm_model(n_steps_in, n_steps_out, train_x, train_y, test_x, test_y, units_1, units_2, units_3, epochs_to_use = 100):
    # Need to clear backend before setting up any model
    K.clear_session()
    model = Sequential()
    model.add(LSTM(units = units_1, input_shape = (n_steps_in, train_x.shape[2]), dropout = 0.0, return_sequences = True)) # return_sequences = True
    model.add(LSTM(units = units_2, activation='relu', dropout = 0.0, return_sequences = True))
    model.add(LSTM(units = units_3, activation='relu', dropout = 0.0))
    # model.add(Dense(50))
    model.add(Dense(n_steps_out))
    model.compile(loss = "mse", optimizer = "adam")
    #model.summary()
    
    history = model.fit(train_x, train_y, epochs = epochs_to_use, batch_size = 64, verbose = 1,
                        validation_data=(test_x, test_y))
    return history, model


n_steps_in= 30
n_steps_out= 14
units_layer_1 = 128
units_layer_2 = 64
units_layer_3 = 64
x_col = ["ma_pct_change_lag1", "Volume", "pct_diff_high_low_lag1", "pct_change_lag1"]
y_col = ["pct_change"]
epochs_to_use = 1000
train_cutoff_data = "2020-09-01"

train_data = bitcoin_data.copy().loc[(bitcoin_data.Date < train_cutoff_data) & (bitcoin_data.Date >= "2015-01-01") , :]
test_data = bitcoin_data.copy().loc[bitcoin_data.Date >= train_cutoff_data, :]

x_train, y_train = create_lstm_dataset(train_data.dropna(), 
                                       y_col = y_col, 
                                       x_col = x_col, 
                                       n_steps_in = n_steps_in, 
                                       n_steps_out = n_steps_out)
x_test, y_test = create_lstm_dataset(test_data.dropna(), y_col = y_col, x_col = x_col, 
                                     n_steps_in = n_steps_in, n_steps_out = n_steps_out)

model_history, model = lstm_model(n_steps_in = n_steps_in, n_steps_out = n_steps_out, 
                                   train_x = x_train, train_y = y_train, 
                                   test_x = x_test, test_y = y_test, 
                                   units_1 = units_layer_1, 
                                   units_2 = units_layer_2,
                                   units_3 = units_layer_3,
                                   epochs_to_use = epochs_to_use)

plt.plot(model_history.history["loss"], label="train data")
plt.plot(model_history.history["val_loss"], label="test data")
plt.title("Optimizing Process")
plt.xlabel("Epoch")
plt.xlabel("MSE")
plt.legend()

model.summary()

yhat_test = model.predict(x_test, verbose=0)

prediction_test = test_data[["Date", "Close"]].iloc[(n_steps_in - 1): (len(test_data) - n_steps_out + 1)]

test_data["pct_change_lag1"].var()


"""1 day pred
prediction_test["y_hat"] = yhat_test[:,0]
prediction_test["next_close_pred"] = prediction_test["Close"] + (prediction_test["Close"] * prediction_test["y_hat"]/100)
prediction_test["pred"] = prediction_test["next_close_pred"].shift(1)

#plt.plot(train_data.Date, train_data.Close, label="train data")
plt.plot(test_data.Date, test_data.Close, label="test data")
plt.plot(prediction_test.Date, prediction_test.pred, label="pred data")
plt.legend()
"""

"""
# 14 Day prediction
for i in range(0, n_steps_out):
    #running total = 0
    col_name = "percent_pred_" + str(i+1)
    col_close_name = "Close_price "+ str(i+1)
    print(col_name)
    # print(yhat_test[:, i].shape)
    prediction_test[col_name] = 1 + yhat_test[:, i]/100

for i in range(0, n_steps_out):
    col_price_name = "est. price on day " + str(i+1)
    prediction_test[col_price_name] = prediction_test.drop("Date", axis = 1).iloc[:, 0:i+2].prod(axis = 1)
    

    
prediction_test[test] = prediction_test.drop("Date", axis = 1).iloc[0:i+1].prod(axis = 1)

prediction_test["14_day_pred"] = prediction_test.drop("Date", axis = 1).prod(axis = 1)
prediction_test["14_day_pred_right_date"] = prediction_test["14_day_pred"].shift(n_steps_out)"""

plt.plot(test_data.Date, test_data.Close, label="test data")
plt.plot(prediction_test.Date, prediction_test["14_day_pred_right_date"], label="pred data")
plt.title(f"Prediction with {n_steps_in} steps in and {n_steps_out} steps_out and {units_layer_1} layer-1 units and {units_layer_2} layer-2 units ")
plt.legend()

# Can't seem to avoid overfitting, and only small movements


def return_last_pred(x_test):
    yhat_test = model.predict(x_test, verbose=0)
    # COde for very last date only
    last_pred = yhat_test[-1,:].reshape(1,-1)
    last_row = test_data.iloc[-1, [0, 4]].to_frame().T
    
    for i in range(0, n_steps_out):
        #running total = 0
        col_name = "percent_pred_" + str(i+1)
        col_close_name = "Close_price "+ str(i+1)
        print(col_name)
        # print(yhat_test[:, i].shape)
        last_row[col_name] = 1 + last_pred[:, i]/100
    
    for i in range(0, n_steps_out):
        col_price_name = last_row.iloc[0,0] + pd.Timedelta(days = i+1)
        print(col_price_name)
        last_row[col_price_name] = last_row.drop("Date", axis = 1).iloc[:, 0:i+2].prod(axis = 1)
        
        
    final_df = pd.DataFrame({"Date" :last_row.iloc[:, -14:].columns, "Pred": last_row.iloc[0, -14:] })