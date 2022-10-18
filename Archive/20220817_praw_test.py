# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 17:01:32 2022

@author: ssn50
"""
import pandas as pd
import praw

print(praw.Reddit)
print(praw.__version__)

reddit_login = praw.Reddit(user_agent = "kneejob",
                           client_id = "RdvI_NkfeiJSh9kn8Urzwg",
                           client_secret = "eZxKr3BOdtyDkm7diJymVErqcDPVkA",
                           check_for_async = False) # This step is to avoid warnings about ratelimits

print(reddit_login.read_only)

for i in reddit_login.subreddit("dogecoin").hot(limit = 10):
    print(i)
    
# I think i need to use https://pushshift.io/api-parameters/
# because reddit limits what I can crawl    