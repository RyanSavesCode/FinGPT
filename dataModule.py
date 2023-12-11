#%matplotlib inline
import pandas as pd
import numpy as np
import datetime
import yfinance as yf

from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl import config_tickers
from finrl.config import INDICATORS,TRAINED_MODEL_DIR, RESULTS_DIR
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3.common.logger import configure
from finrl.main import check_and_make_directories
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3

check_and_make_directories([TRAINED_MODEL_DIR])
# Function to pull, clean, and save data
def getData(dates,indicators,tickers):
    if len(dates[0]) != 10 or len(indicators) == 0 or len(tickers) == 0:
        print("Can't grab data! Make sure dates, indicators, and tickers are selected.")
    else:
        import itertools
        TRAIN_START_DATE = "2020-01-05" #This date MUST always be a monday.
        TRAIN_END_DATE = "2020-09-26"
        TRADE_START_DATE = "2023-10-05"
        TRADE_END_DATE = "2023-10-15"
        
        #Grab data
        df_raw = YahooDownloader(start_date = TRAIN_START_DATE,
                            end_date = TRADE_END_DATE,
                            ticker_list = tickers).fetch_data()
        
        #Feature settings
        fe = FeatureEngineer(use_technical_indicator=True,
                            tech_indicator_list = indicators,
                            use_vix=True,
                            use_turbulence=True,
                            user_defined_feature = False)
        
        #Process & clean Data
        processed = fe.preprocess_data(df_raw)
        
        list_ticker = processed["tic"].unique().tolist()
        list_date = list(pd.date_range(processed['date'].min(),processed['date'].max()).astype(str))
        combination = list(itertools.product(list_date,list_ticker))
        
        processed_full = pd.DataFrame(combination,columns=["date","tic"]).merge(processed,on=["date","tic"],how="left")
        processed_full = processed_full[processed_full['date'].isin(processed['date'])]
        processed_full = processed_full.sort_values(['date','tic'])
        
        processed_full = processed_full.fillna(0)
        
        train = data_split(processed_full, TRAIN_START_DATE,TRAIN_END_DATE)
        trade = data_split(processed_full, TRADE_START_DATE,TRADE_END_DATE)
        
        #Save data to CSV files
        train.to_csv('train_data.csv')
        trade.to_csv('trade_data.csv')
    