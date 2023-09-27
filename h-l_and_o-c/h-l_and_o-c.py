import sys
sys.path.append('../custom')

import pandas as pd
import talib
from utils import create_empty_directory

stocks = [{"name": "nifty", "symbol": "^NSEI"}, {"name":"axis", "symbol": "AXISBANK.NS"}, {"name":"tata", "symbol": "TATAMOTORS.NS"}]

def create(indicator_name, stock_name, stock_symbol):
    create_empty_directory(f"../data/{stock_name}/{indicator_name}")
    data = pd.read_csv(f'../data/{stock_name}/all_data/{stock_symbol}_all_data.csv')

    data["OC"] = data["Open"] - data["Close"]
    data["HL"] = data["High"] - data["Low"]

    data.to_csv(f"../data/{stock_name}/{indicator_name}/{stock_symbol}_{indicator_name}.csv")
    train_filename = f"../data/{stock_name}/{indicator_name}/{stock_symbol}_train_{indicator_name}.csv"
    test_filename = f"../data/{stock_name}/{indicator_name}/{stock_symbol}_test_{indicator_name}.csv"

    train_perc = 0.8
    total_rows = len(data)
    train_rows_len = int(train_perc * total_rows)
    test_rows_len = int((1 - train_perc) * total_rows)
    data.head(train_rows_len).to_csv(train_filename)
    data.tail(test_rows_len).to_csv(test_filename)


for stock in stocks:
    create("h-l_o-c", stock["name"], stock["symbol"])

print(f"Done")