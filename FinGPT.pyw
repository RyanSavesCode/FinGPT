'''
Author: Ryan Cathey
Creation Date: 12/8/23
Description:
    Top-Level python executable for the FinRL integration with ChatGPT coined FinGPT.
    This application structuresa a graphical user interface and imports FinRL functions and ChatGPT functionality. 
Name: FinGPT.pyw
'''
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#Py Imports.
import tkinter as tk
from tkinter import filedialog
import subprocess
import threading
import os, sys
import pandas as pd
import numpy as np
import datetime
import yfinance as yf

#FinRL Imports.
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.config import TRAINED_MODEL_DIR, RESULTS_DIR
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3.common.logger import configure
from finrl.main import check_and_make_directories
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
import dataModule, trainModule, explainModule, predictModule

#Console stream to grab win cmd output and put in in GUI.
def capture_prints(text_widget):
    class CustomStream:
        def write(self, text):
            text_widget.config(state='normal')
            text_widget.insert('end', text)
            text_widget.see('end')
            text_widget.config(state='disabled')
        def flush(val):
            return -1

    sys.stdout = CustomStream()
    sys.stderr = CustomStream()

# Function to execute the selected script and capture output.
results = [[],[],[],[]]
explanation = ""
def execute_cmd(cmd_name):
    update_dates_list()
    update_opt_list()
    update_ind_list()
    if cmd_name[:7] == "predict":
        exec("results = "+cmd_name,globals())
    elif cmd_name[:7] == "explain":
        exec("explanation = "+cmd_name,globals())
    else:
        exec(cmd_name)

# Function to add or remove a stock name to the list.
def stock_add_rem(add_remove):
    new_stock = stock_entry.get()
    if add_remove == 'add':
        new_stock_val = stock_val_entry.get()
        if new_stock and new_stock_val and new_stock not in stock_dict:
            stock_dict[new_stock] = new_stock_val
        else:
            print("Enter a valid Ticker and # of stock")
    else:
        if new_stock and new_stock in stock_dict:
            stock_dict.pop(new_stock)
    update_stock_dict()
    stock_entry.delete(0, 'end')
    
# Function to add an indicator name to the list.
def update_stock_dict():
    stock_listbox.delete(0, 'end')
    for stock in stock_dict:
        stock_listbox.insert('end', str(stock) + ": " + str(stock_dict[stock]))
        
# Function to add an indicator name to the list.
def update_dates_list():
    dates_list[0] = start_train_date.get()
    dates_list[1] = end_train_date.get()
    dates_list[2] = start_trade_date.get()
    dates_list[3] = end_trade_date.get()
    if (dates_list[2] == dates_list[3]) :
        print("\nEnsure your start trade date is not the same as the end date!\n")

# Function to update indicator list.
def update_ind_list():
    for indicator in list(indicator_dict):
        if indicator_dict[indicator].get() == 0 and indicator in indicator_list:
            indicator_list.remove(indicator)
        elif indicator_dict[indicator].get() == 1 and indicator not in indicator_list:
            indicator_list.append(indicator)

# Function to update options list.
def update_opt_list():
    stock_opt[0] = cash_entry.get()
    stock_opt[1] = stock_max_entry.get()
    stock_opt[2] = risk_entry.get()
    
# Create the main GUI window.
root = tk.Tk()
root.title("FinGPT")

# Create a text widget for displaying the script output.
output_text = tk.Text(root, height=25, width=100)
output_text.pack(side='bottom',pady=15)
output_text.config(state='disabled')

capture_prints(output_text)

# Define GUI frames.
data_frame = tk.Frame(root, borderwidth=2)
stock_entry_frame = tk.Frame(root, borderwidth=2)
stock_list_frame = tk.Frame(root, borderwidth=2)
stock_opt_frame =  tk.Frame(root, borderwidth=2)
indicator_frame = tk.Frame(root, borderwidth=2)
button_frame = tk.Frame(root, borderwidth=2)

# Create date entry fields for the data to grab. Set default dates.
dates_list = ["","","",""]
date_label = tk.Label(data_frame, text="Enter start date for training (YYYY-MM-DD):")
date_label.pack(padx=10, pady=2)
start_train_date = tk.Entry(data_frame)
start_train_date.insert(0, "2020-01-01")
start_train_date.pack(padx=2, pady=2)
date_label = tk.Label(data_frame, text="Enter end date for training (YYYY-MM-DD):")
date_label.pack(padx=10, pady=2)
end_train_date = tk.Entry(data_frame)
end_train_date.insert(0, "2023-11-01")
end_train_date.pack(padx=2, pady=2)
date_label = tk.Label(data_frame, text="Enter start date for trading (YYYY-MM-DD):")
date_label.pack(padx=10, pady=2)
start_trade_date = tk.Entry(data_frame)
start_trade_date.insert(0, "2023-11-05")
start_trade_date.pack(padx=2, pady=2)
date_label = tk.Label(data_frame, text="Enter end date for trading (YYYY-MM-DD):")
date_label.pack(padx=10, pady=2)
end_trade_date = tk.Entry(data_frame)
end_trade_date.insert(0, "2023-11-09")
end_trade_date.pack(padx=2, pady=2)
data_frame.pack(side='left',padx=5,pady=10)

# Create stock entry fields. Set default stocks.
stock_dict = {}
stock_label = tk.Label(stock_entry_frame, text="Stock Tickers")
stock_label.pack(padx=1, pady=2)
stock_entry = tk.Entry(stock_entry_frame)
stock_entry.pack(padx=1, pady=5)
stock_val_label = tk.Label(stock_entry_frame, text="# of Stock")
stock_val_label.pack(padx=1, pady=2)
stock_val_entry = tk.Entry(stock_entry_frame)
stock_val_entry.pack(padx=1, pady=5)
stock_add_button = tk.Button(stock_entry_frame, text="Add Stock", command=lambda: stock_add_rem('add'))
stock_add_button.pack(padx=1,pady=5)
stock_rem_button = tk.Button(stock_entry_frame, text="Remove Stock", command=lambda: stock_add_rem('remove'))
stock_rem_button.pack(padx=1,pady=5)
stock_list_label = tk.Label(stock_list_frame, text="Stocks: # in Portfolio")
stock_list_label.pack(padx=1, pady=2)
stock_listbox = tk.Listbox(stock_list_frame)
stock_listbox.pack(padx=10, pady=5)
stock_entry.insert(0, "AMZN")
stock_val_entry.insert(0, 50)
stock_add_rem('add')
stock_entry.insert(0, "TSLA")
stock_add_rem('add')
update_stock_dict()
stock_entry_frame.pack(side='left',padx=1,pady=10)
stock_list_frame.pack(side='left',padx=1,pady=10)

# Create Stock Portfolio option fields. Set default stocks.
stock_opt = [0,0,0]
cash_label = tk.Label(stock_opt_frame, text="Portfolio Cash")
cash_label.pack(padx=1, pady=2)
cash_entry = tk.Entry(stock_opt_frame)
cash_entry.pack(padx=1, pady=5)
stock_max_label = tk.Label(stock_opt_frame, text="Stock Max")
stock_max_label.pack(padx=1, pady=2)
stock_max_entry = tk.Entry(stock_opt_frame)
stock_max_entry.pack(padx=1, pady=5)
risk_label = tk.Label(stock_opt_frame, text="Risk Factor")
risk_label.pack(padx=1, pady=2)
risk_entry = tk.Entry(stock_opt_frame)
risk_entry.pack(padx=1, pady=5)
cash_entry.insert(0,1000000)
stock_max_entry.insert(0, 100)
risk_entry.insert(0,70)
stock_opt_frame.pack(side='left',padx=1,pady=10)

# Create indicator entry fields.
indicator_label = tk.Label(indicator_frame, text="Indicators")
indicator_label.pack(padx=10, pady=10)
indicator_dict = {  "macd" : tk.IntVar(),
                    "boll_ub" : tk.IntVar(),
                    "boll_lb" : tk.IntVar(),
                    "rsi_30" : tk.IntVar(),
                    "cci_30" : tk.IntVar(),
                    "dx_30" : tk.IntVar(),
                    "close_30" : tk.IntVar(),
                    "close_60" : tk.IntVar()
                }
indicator_list = []
check_macd = tk.Checkbutton(indicator_frame, text="macd", variable=indicator_dict["macd"], command=update_ind_list())
check_boll_ub = tk.Checkbutton(indicator_frame, text="boll_ub", variable=indicator_dict["boll_ub"], command=update_ind_list())
check_boll_lb = tk.Checkbutton(indicator_frame, text="boll_lb", variable=indicator_dict["boll_lb"], command=update_ind_list())
check_rsi_30 = tk.Checkbutton(indicator_frame, text="rsi_30", variable=indicator_dict["rsi_30"], command=update_ind_list())
check_cci_30 = tk.Checkbutton(indicator_frame, text="cci_30", variable=indicator_dict["cci_30"], command=update_ind_list())
check_dx_30 = tk.Checkbutton(indicator_frame, text="dx_30", variable=indicator_dict["dx_30"], command=update_ind_list())
check_close_30_sma = tk.Checkbutton(indicator_frame, text="close_30_sma", variable=indicator_dict["close_30"], command=update_ind_list())
check_close_60_sma = tk.Checkbutton(indicator_frame, text="close_60_sma", variable=indicator_dict["close_60"], command=update_ind_list())
check_macd.pack()
check_boll_ub.pack()
check_boll_lb.pack()
check_rsi_30.pack()
check_cci_30.pack()
check_dx_30.pack()
check_close_30_sma.pack()
check_close_60_sma.pack()

indicator_frame.pack(side='left',padx=25,pady=10)

# Create and configure the function buttons.
button_label = tk.Label(button_frame, text="Functions")
button_label.pack(padx=10, pady=10)
pull_button = tk.Button(button_frame, text="Pull Data", command=lambda : execute_cmd("[update_dates_list(), update_ind_list(), dataModule.getData(dates_list, indicator_list, stock_dict)]"))
pull_button.pack(padx=10,pady='5')
train_button = tk.Button(button_frame, text="Train", command=lambda : execute_cmd("trainModule.trainAgent(indicator_list,stock_opt)"))
train_button.pack(padx=10,pady='5')
predict_button = tk.Button(button_frame, text="Predict", command=lambda : execute_cmd("predictModule.predict(dates_list,stock_dict,stock_opt,indicator_list)"))
predict_button.pack(padx=10,pady='5')
explain_button = tk.Button(button_frame, text="Explain", command=lambda : execute_cmd("explainModule.explain(indicator_list, results[2], results[3], stock_dict, results[1])"))
explain_button.pack(padx=25,pady='5')
elab_button = tk.Button(button_frame, text="Explain Indicators", command=lambda : execute_cmd("explainModule.elaborate(explanation)"))
elab_button.pack(padx=25,pady='5')

button_frame.pack(side='right',padx=15,pady=10)

# Run the GUI main loop.
root.mainloop()
