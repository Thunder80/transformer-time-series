import pandas as pd
import talib

name = "doji"
# Load the CSV file into a pandas DataFrame
data = pd.read_csv('../data/nifty/all_data/^NSEI_all_data.csv')  # Replace 'stock_data.csv' with your file name

# Calculate the Doji values
data['Doji'] = talib.CDLDOJI(data['Open'], data['High'], data['Low'], data['Close'])

# Print the rows where Spinning Top pattern is identified
spinning_top_patterns = data[data['Doji'] == 100]
print(spinning_top_patterns)

data.to_csv(f"../data/nifty/doji/^NSEI_{name}.csv")
train_filename = f"../data/nifty/doji/^NSEI_train_{name}.csv"
test_filename = f"../data/nifty/doji/^NSEI_test_{name}.csv"

train_perc = 0.8
total_rows = len(data)
train_rows_len = int(train_perc * total_rows)
test_rows_len = int((1 - train_perc) * total_rows)
data.head(train_rows_len).to_csv(train_filename)
data.tail(test_rows_len).to_csv(test_filename)

print(f"Done")