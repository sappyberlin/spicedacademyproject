# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 09:01:57 2022

@author: ssn50
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.io as pio
import pydeck as pdk
# pio.renderers.default='browser'
# %reset -f
from cryptoproject_model_lstm_v3 import dataframe_prep, create_lstm_dataset, lstm_model, return_last_pred

# To run code in Github, need to use following cmd "streamlit run 20220930_webpage.py"


# st.set_page_config(layout = "wide")


# Title
st.title("Cryptocurrency Forecast Project")

# Explanatory Test
st.write("A Price Prediction using a LSTM forecast model")

# Choosing whether to look at Bitcoin or Ethereum
coin_choice = st.selectbox("Which coin would you like to look at?", ["Bitcoin", "Ethereum"])
# coin_choice = "Bitcoin"
filename_to_read = "C:\\Users\\ssn50\\spiced_academy\\ginger-pipeline-student-code\\Crypto project\\data\\" +str(coin_choice) + ".csv"

st.write(coin_choice)

orginal_data = pd.read_csv(filename_to_read, parse_dates = ["Date"]).sort_values(by = "Date", ascending = True)
data = dataframe_prep(orginal_data)



st.header(f"The current closing price and change data for {coin_choice} is:")

# Showing graphs here
fig_price = px.line(data, x = "Date", y = "Close", title = "The current price data", markers = False)
st.plotly_chart(fig_price)

fig_pct_change = px.line(data, x = "Date", y = "pct_change", title = "The current daily change", markers = False)
st.plotly_chart(fig_pct_change)

st.header("How would you like to forecast")

# User inputs for the LSMT model
past_forecast_period = st.slider("How many past periods should we use?", value = 14, min_value = 1, max_value =  28, step = 1)
future_forecast_period = st.slider("How many periods into the future should we use?", value = 14, min_value = 1, max_value =  28, step = 1)
epochs_to_use = st.slider("How many epochs should we use to train our data? (One epoch typically takes 2-4 seconds)", value = 20, min_value = 1, max_value =  250, step = 1)


# Code for running prediction
if st.button("Ready to see the prediction"):
    x_data, y_data = create_lstm_dataset(dataframe = data, 
                                         n_steps_in = past_forecast_period, n_steps_out = future_forecast_period)
    
    
    history, model = lstm_model(n_steps_in = past_forecast_period, n_steps_out = future_forecast_period, 
                                train_x = x_data, train_y = y_data, epochs_to_use = epochs_to_use)
    
    # st.write(model)
    
    predictions = return_last_pred(crypto_df = data, x_test = x_data, model = model, n_steps_out = future_forecast_period)
    predictions["Label"] = "Prediction"
    last_30_days = data.iloc[-30:, [0, 4]]
    last_30_days["Label"] = "Last 30 Days"
    pred_and_last_30 = pd.concat([last_30_days, predictions], join = "outer")
    
    
    fig_pred = px.line(pred_and_last_30, x = "Date", y = "Close", color = "Label", title = "Predictions", markers = False)
    st.plotly_chart(fig_pred)
   















