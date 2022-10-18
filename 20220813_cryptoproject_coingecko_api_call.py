# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 16:37:52 2022

@author: ssn50
"""

import requests
import pandas as pd
import numpy as np
import datetime
import time

# coingecko appears better suited to my needs
# https://www.coingecko.com/en/api/documentation
# Need to add a for loop timer, because "Our Free API* has a rate limit of 50 calls/minute. Need something more flexible and powerful? View our API plans now."

def coingecko_get_api(coin_name : list, start_date, end_date, save_file_loc):
    
    # I need to convert the date into a format that datetime can "loop" over and create a list for the date   
    start = datetime.datetime.strptime(start_date, '%Y-%m-%d').date()
    end = datetime.datetime.strptime(end_date, '%Y-%m-%d').date()
    # delta = datetime.timedelta(days=1)
    date_range = [start + datetime.timedelta(days=x) for x in range(0, (end-start).days + 1)]
    # print(date_range)
    # Initializing a list so I can concat the dfs at the end
    df_saved = pd.read_csv(save_file_loc, usecols=(range(0,13)))
    api_list = []
        
    for coin in coin_name:
        
    # Then i need to convert the date into a format that can be used by the coingecko api to pull the data!
        for date in date_range:
            api_date = date.strftime("%d") + "-" + date.strftime("%m") + "-" + date.strftime("%Y")
            # The api address I need
            api_url = "https://api.coingecko.com/api/v3/coins/" + coin + "/history?date=" + str(api_date)
            # print(api_url)
            # Simple if statment to avoid re-downloading data I already have
            if api_url in df_saved["api"].values:
                continue
            else:
                api_list.append(api_url)
            
    return api_list

# check how to repeat a call if failed, maybe with a while loop...
def coingecko_get_crypto_data(api_list : list, save_file_loc):
    df_list = []
    call_length = len(api_list)*1.35
    count = 1
    for api in api_list:
        results = requests.get(api)
        while(results.status_code != 200):
            # creating longer pause here to make sure I can get all the data out
            print("Server download not successful, pausing download for 60 seconds")
            time.sleep(60)
            results = requests.get(api)
        
        
        
        # extracting the data from json and putting into a df
        #results_json = results.json()
        df_row = pd.json_normalize(results.json())
            
        columns_to_create = ["community_data.reddit_accounts_active_48h", "community_data.reddit_average_comments_48h",
                            "community_data.reddit_average_posts_48h", "community_data.reddit_subscribers", "community_data.twitter_followers",
                            "market_data.current_price.usd", "market_data.market_cap.usd", "market_data.total_volume.usd",
                            "public_interest_stats.alexa_rank"]
        
        if set(columns_to_create).issubset(df_row.columns):
            df_row = df_row.loc[:, ["name", "symbol", "community_data.reddit_accounts_active_48h", "community_data.reddit_average_comments_48h",
                                                          "community_data.reddit_average_posts_48h", "community_data.reddit_subscribers", "community_data.twitter_followers",
                                                          "market_data.current_price.usd", "market_data.market_cap.usd", "market_data.total_volume.usd",
                                                          "public_interest_stats.alexa_rank"]]  \
                                                .rename(mapper = {"community_data.reddit_accounts_active_48h" : "reddit_accounts_active_48h", 
                                                        "community_data.reddit_average_comments_48h" : "reddit_average_comments_48h",
                                                        "community_data.reddit_average_posts_48h" : "reddit_average_posts_48h", 
                                                        "community_data.reddit_subscribers" : "reddit_subscribers", 
                                                        "community_data.twitter_followers" : "twitter_followers",
                                                        "market_data.current_price.usd" : "current_price_usd", 
                                                        "market_data.market_cap.usd" : "market_cap_usd", 
                                                        "market_data.total_volume.usd" : "total_volume_usd",
                                                        "public_interest_stats.alexa_rank" : "alexa_rank"}, 
                                                axis = "columns")
                                                
        else:
            # subset_and add
            df_row = df_row[["name", "symbol"]]
            for col_name in columns_to_create:
                df_row[col_name] = np.nan
                                        
        df_row["date"] = api[-10:]
        df_row["api"] = api
        df_list.append(df_row)
        time.sleep(1.35)
        
        print("We are ", round(((count*1.35)/call_length)*100, 2), "% through", " it should take ", 
              round((call_length - count*1.35), 2), " more seconds! (hopefully...)", sep = "" )
        count += 1
    
    if df_list == []:
        print("There is nothing to add to the file")
        return None
    else:
        df_final = pd.concat(df_list)
        df_final.to_csv(save_file_loc,
                    mode = "a", index = False, header = False) # I want to append to file
        return df_final


test_date_start = "2014-01-01"
test_date_end = str(datetime.date.today() - datetime.timedelta(days=1))
test_name = ["ethereum"]
save_file_loc = "C:\\Users\\ssn50\\spiced_academy\\ginger-pipeline-student-code\\Crypto project\\data.csv"

apis = coingecko_get_api(coin_name = test_name, start_date = test_date_start, end_date = test_date_end, save_file_loc = save_file_loc)
df = coingecko_get_crypto_data(apis, save_file_loc = save_file_loc)

# Link for sentiment analysis
#https://medium.com/@jamesthesken/building-an-altcoin-market-sentiment-monitor-99226a6f03f6
#https://www.bittsanalytics.com/cryptocurrency_api.php#query-parameters-9









