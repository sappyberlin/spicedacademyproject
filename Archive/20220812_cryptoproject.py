# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 16:37:52 2022

@author: ssn50
"""

import requests
import pandas as pd
import numpy as np

# Alpha Vantage Key and api connect
api_key = "3FRO0KW1AXML7EPD"

def alphavantage_get_crypto_price(symbol, exchange, api_key, start_date = None):
    api_key = str(api_key)
    api_url = "https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol=" + symbol + "&market="+ exchange + "&apikey=" + api_key
    #api_url = f'https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol={symbol}&market={exchange}&apikey={api_key}'
    raw_df = requests.get(api_url).json()
    # .T is a transpose method/attribute
    crypto_df = pd.DataFrame(raw_df['Time Series (Digital Currency Daily)']).T
    crypto_df = crypto_df.rename(columns = {'1a. open (USD)': 'open', '2a. high (USD)': 'high', '3a. low (USD)': 'low', '4a. close (USD)': 'close', '5. volume': 'volume'})
    
    for i in crypto_df.columns:
        crypto_df[i] = crypto_df[i].astype(float)
    
    crypto_df.index = pd.to_datetime(crypto_df.index)
    crypto_df = crypto_df.iloc[::-1].drop(['1b. open (USD)', '2b. high (USD)', '3b. low (USD)', '4b. close (USD)', '6. market cap (USD)'], axis = 1)
    if start_date:
        crypto_df= crypto_df[crypto_df.index >= start_date]
    return crypto_df




symbol = "test"
exchange = "test_exchange"
api_url = "https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol=" + symbol + "&market="+ exchange + "&apikey=" + api_key

# btc = alphavantage_get_crypto_price(symbol = "BTC", exchange = "USD", api_key = api_key)


# coingecko appears better suited to my needs
# https://www.coingecko.com/en/api/documentation

def coingecko_get_crypto_price(coin_name, date):
    # api_url = "https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol=" + symbol + "&market="+ exchange + "&apikey=" + api_key
    api_url = "https://api.coingecko.com/api/v3/coins/" + coin_name + "/history?date=" + str(date)
    results_json = requests.get(api_url).json()
    crypto_df = pd.DataFrame(results_json)
    
    """crypto_df = pd.DataFrame(raw_df['Time Series (Digital Currency Daily)']).T
    # crypto_df = crypto_df.rename(columns = {'1a. open (USD)': 'open', '2a. high (USD)': 'high', '3a. low (USD)': 'low', '4a. close (USD)': 'close', '5. volume': 'volume'})
    
    for i in crypto_df.columns:
        crypto_df[i] = crypto_df[i].astype(float)
    
    crypto_df.index = pd.to_datetime(crypto_df.index)
    crypto_df = crypto_df.iloc[::-1].drop(['1b. open (USD)', '2b. high (USD)', '3b. low (USD)', '4b. close (USD)', '6. market cap (USD)'], axis = 1)
    if start_date:
        crypto_df= crypto_df[crypto_df.index >= start_date]"""
    return crypto_df, api_url, results_json

test_date = "11-08-2022"
test_name = "bitcoin"

btc, api, json_output = coingecko_get_crypto_price(test_name, test_date)

# I think the json normalize function will help a lot here!

test = pd.json_normalize(json_output )




start = '2015-08-01' #YYY-MM-DD
end = '2015-08-15'


dates = pd.date_range(start = start, end = end, 
                               freq = "D").date

for i in dates: 
    print(i, type(i))






















