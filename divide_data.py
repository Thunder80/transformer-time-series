import pandas as pd
import sys

filename = sys.argv[1]
feature = "volume_oscillator"
train_perc = 0.8

stock_data = pd.read_csv(f"data/{feature}/{filename}_all_{feature}.csv")

csv_filename = f"data/{feature}/{filename}_all_data.csv"
train_filename = f"data/{feature}/{filename}_train_{feature}.csv"
test_filename = f"data/{feature}/{filename}_test_{feature}.csv"

# stock_data.to_csv(csv_filename)

total_rows = len(stock_data)
train_rows_len = int(train_perc * total_rows)
test_rows_len = int((1 - train_perc) * total_rows)
stock_data.head(train_rows_len).to_csv(train_filename)
stock_data.tail(test_rows_len).to_csv(test_filename)

print(f"Done {csv_filename}")