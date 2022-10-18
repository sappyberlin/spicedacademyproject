# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 11:05:33 2022

@author: ssn50
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly
import seaborn as sns
# Modules for running the inferential statistics

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_squared_log_error, r2_score



# Data from https://coincodex.com/crypto/ethereum/historical-data/
bitcoin_data = pd.read_csv("C:\\Users\\ssn50\\spiced_academy\\ginger-pipeline-student-code\\Crypto project\\coincodex_bitcoin_2010-07-23_2022-09-22.csv", parse_dates = ["Date"] )

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
"""bitcoin_data["pct_change_lag1"] = bitcoin_data["pct_change"].shift(1)"""
# Creating a variable to get a sense of the volatility
bitcoin_data["pct_diff_high_low"] = ((bitcoin_data["High"] - bitcoin_data["Low"])/bitcoin_data["Low"])*100
bitcoin_data["pct_diff_high_low_lag1"] = bitcoin_data["pct_diff_high_low"].shift(1)
# Moving Average
"""bitcoin_data["ma_pct_change_lag1"] = bitcoin_data["pct_change_lag1"].rolling(30).mean()"""

# Putting pct_change into bins
#bitcoin_data()

# Lag for N periods behind, pct_change
for i in range(1, 8): 
    col_name = "pct_change_lag" + str(i)
    col_name_vol = "vol_change_lag" + str(i)
    print(col_name)
    bitcoin_data[col_name] = bitcoin_data["pct_change"].shift(i)
    bitcoin_data[col_name_vol] = bitcoin_data["Volume"].shift(i)



future_value = 1
# bitcoin_data["future_value_7"] = bitcoin_data["pct_change"].shift(-7) 
bitcoin_data["future_value_7"] = ((bitcoin_data["Close"] - bitcoin_data["Close"].shift(future_value))/bitcoin_data["Close"].shift(future_value))*100

#bin_values = np.arange(start=-50, stop=50, step=1)
bitcoin_data["future_value_7"].hist()

bitcoin_data.columns
x_col = ['pct_change_lag1', 'vol_change_lag1','pct_change_lag2', 'vol_change_lag2', 'pct_change_lag3','vol_change_lag3', 'pct_change_lag4', 'vol_change_lag4',
         'pct_change_lag5', 'vol_change_lag5', 'pct_change_lag6', 'vol_change_lag6', 'pct_change_lag7', 'vol_change_lag7', "pct_diff_high_low_lag1"] 
y_col = ["future_value_7"]


train_data = bitcoin_data.copy().loc[(bitcoin_data.Date < "2020-01-01") & (bitcoin_data.Date >= "2015-01-01") , :].dropna()
test_data = bitcoin_data.copy().loc[bitcoin_data.Date >= "2020-01-01", :].dropna()


m_rf =  RandomForestClassifier()
random_forest_regressor = RandomForestRegressor(n_estimators = 50,
                                                max_depth = 50 )

random_forest_regressor.fit(X = train_data[x_col], y = train_data[y_col])

y_train_pred = random_forest_regressor.predict(train_data[x_col])
rsquare_score = random_forest_regressor.score(train_data[x_col], train_data[y_col])
rmse = np.sqrt(mean_squared_error(train_data[y_col], y_train_pred))

y_test_pred = random_forest_regressor.predict(test_data[x_col])
test_rsquare_score = random_forest_regressor.score(test_data[x_col], test_data[y_col])
test_rmse = np.sqrt(mean_squared_error(test_data[y_col], y_test_pred))

test_data["rf_pred"] = y_test_pred 
test_data["rf_price_pred"] = ((1 + test_data["rf_pred"]/100) * test_data["Close"]).shift(future_value)
test_data = test_data.dropna()

plt.plot(test_data.Date, test_data.Close, label = "test data")
plt.plot(test_data.Date, test_data.rf_price_pred, label = "pred data")
plt.legend()
plt.title(f"estimating {future_value} in the future")