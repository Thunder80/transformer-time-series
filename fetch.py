import yfinance as yf
import pandas as pd
from custom.utils import create_empty_directory

train_perc = 0.8
# Replace 'AAPL' with the stock symbol of your choice
stock_symbol = 'ETH-USD'
stock_name = "eth"

# Fetch all historical stock data
stock_data = yf.download(stock_symbol)
stock_data = stock_data[stock_data["Volume"] != 0]

create_empty_directory(f"data/{stock_name}/all_data")
# Save stock data as a CSV file
csv_filename = f"data/{stock_name}/all_data/all.csv"
train_filename = f"data/{stock_name}/all_data/train.csv"
test_filename = f"data/{stock_name}/all_data/test.csv"

stock_data.to_csv(csv_filename)

total_rows = len(stock_data)
train_rows_len = int(train_perc * total_rows)
test_rows_len = int((1 - train_perc) * total_rows)
stock_data.head(train_rows_len).to_csv(train_filename)
stock_data.tail(test_rows_len).to_csv(test_filename)

print(f"Done {csv_filename}")