#%matplotlib inline
import pandas as pd
import numpy as np
import datetime
import yfinance as yf

from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl import config_tickers
from finrl.config import TRAINED_MODEL_DIR, RESULTS_DIR
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3.common.logger import configure
from finrl.main import check_and_make_directories
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3

check_and_make_directories([TRAINED_MODEL_DIR])

# Function to build environment and get prediction from DRL agent/s.
import itertools
def predict(dates,tickers,options,indicators):
    TRAIN_START_DATE = dates[0]
    TRAIN_END_DATE = dates[1]
    TRADE_START_DATE = dates[2]
    TRADE_END_DATE = dates[3]
    #Regrab data
    df_raw = YahooDownloader(start_date = TRAIN_START_DATE,
                        end_date = TRADE_END_DATE,
                        ticker_list = tickers).fetch_data()
    fe = FeatureEngineer(use_technical_indicator=True,
                        tech_indicator_list = indicators,
                        use_vix=True,
                        use_turbulence=True,
                        user_defined_feature = False)
    #Reprocess data
    processed = fe.preprocess_data(df_raw)
    list_ticker = processed["tic"].unique().tolist()
    list_date = list(pd.date_range(processed['date'].min(),processed['date'].max()).astype(str))
    combination = list(itertools.product(list_date,list_ticker))
    
    #Clean data
    processed_full = pd.DataFrame(combination,columns=["date","tic"]).merge(processed,on=["date","tic"],how="left")
    processed_full = processed_full[processed_full['date'].isin(processed['date'])]
    processed_full = processed_full.sort_values(['date','tic'])
    
    processed_full = processed_full.fillna(0)
    trade = data_split(processed_full, TRADE_START_DATE,TRADE_END_DATE)
    if_using_a2c = True
    trained_a2c = A2C.load("trained_models/agent_a2c") if if_using_a2c else None
    
    #Build environment
    stock_dimension = len(trade.tic.unique())
    state_space = 1 + 2*stock_dimension + len(indicators)*stock_dimension
    
    buy_cost_list = sell_cost_list = [0.001] * stock_dimension
    num_stock_shares = []
    for tick in tickers:
        num_stock_shares.append(int(tickers[tick]))

    env_kwargs = {
        "hmax": int(options[1]),
        "initial_amount": int(options[0]),
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": sell_cost_list,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": indicators,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4
    }
    
    e_trade_gym = StockTradingEnv(df = trade, turbulence_threshold = int(options[2]),risk_indicator_col='vix', **env_kwargs)
    
    #Make prediction
    df_account_value_a2c, df_actions_a2c = DRLAgent.DRL_prediction(
        model=trained_a2c, 
        environment = e_trade_gym) if if_using_a2c else (None, None)
    indicators.insert(0,'turbulence')
    indicators.insert(0,'vix')
    #df_account_value_a2c shows performance of portfolio
    averages = []
    finals = []
    date=df_actions_a2c.iloc[len(df_actions_a2c)-1:,0:0].index.values.astype(str)[0]
    
    #Format prediction into text output
    ticker_list = list(tickers.keys())
    print("\n\nFinRL suggests to ")
    result_values = []
    for item in ticker_list:
        result_values.append(df_actions_a2c[str(item)][0])
    start_values = list(tickers.values())
    for index in range(0,len(result_values)):
        if int(result_values[index]) < int(start_values[index]):
            print("sell " + str(int(start_values[index]) - int(result_values[index])) + " shares of " + ticker_list[index]+ ".")
        elif int(result_values[index]) > int(start_values[index]):
            print("buy " + str(int(result_values[index]) - int(start_values[index])) + " shares of " + ticker_list[index]+ ".")
        elif int(result_values[index]) == int(start_values[index]):
            print("hold " + str(int(start_values[index])) + "shares of " + ticker_list[index]+ ".")
    print("for the date " + date)
    
    #Format data to pass back to wrapper
    for index in range(0,len(tickers.keys())):
        averages.append(processed_full[indicators].iloc[index::len(tickers.keys())].mean())
        finals.append(processed_full[processed_full['date']==date][indicators].iloc[index::len(tickers.keys())].mean())
    #Return account value, predictions, metric averages, metrics for trade dates
    return [df_account_value_a2c,df_actions_a2c,averages,finals]