## install required packages
#pip install swig
#pip install wrds
#pip install pyportfolioopt
## install finrl library
#pip install git+https://github.com/AI4Finance-Foundation/FinRL.git
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

import itertools
# Function to build environment and get prediction from DRL agent/s.
def trainAgent(indicators,options):
    train = pd.read_csv('train_data.csv')
    
    train = train.set_index(train.columns[0])
    train.index.names = ['']
    
    stock_dimension = len(train.tic.unique())
    state_space = 1 + 2*stock_dimension + len(indicators)*stock_dimension
    
    #Define environment parameters
    buy_cost_list = sell_cost_list = [0.001] * stock_dimension
    num_stock_shares = [0] * stock_dimension
    
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
    
    e_train_gym = StockTradingEnv(df = train, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()
    
    # Create agent with environment parameters
    agent = DRLAgent(env = env_train)
    model_a2c = agent.get_model("a2c")
    
    # Set the corresponding values to 'True' for the algorithms that you want to use
    if_using_a2c = True
    if_using_ddpg = False
    if_using_ppo = False
    if_using_td3 = False
    if_using_sac = False
    
    # Use a2c agent
    if if_using_a2c:
        # set up logger
        tmp_path = RESULTS_DIR + '/a2c'
        new_logger_a2c = configure(tmp_path, ["stdout", "csv", "tensorboard"])
        # Set new logger
        model_a2c.set_logger(new_logger_a2c)
    
    #Begin training
    print("\nPlease wait while the model is trained...\n")
    trained_a2c = agent.train_model(model=model_a2c, 
                                tb_log_name='a2c',
                                total_timesteps=50000) if if_using_a2c else None
    
    #Save trained model
    trained_a2c.save(TRAINED_MODEL_DIR + "/agent_a2c") if if_using_a2c else None
