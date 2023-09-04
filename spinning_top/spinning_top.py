import pandas as pd

name = "spinning_top"
# Load the CSV file into a pandas DataFrame
data = pd.read_csv('../data/AXISBANK.NS_all_data.csv')  # Replace 'stock_data.csv' with your file name

# Calculate the difference between open and close prices
data['PriceDifference'] = abs(data['Open'] - data['Close'])

# Calculate the difference between high and low prices
data['ShadowSize'] = data['High'] - data['Low']

# Set a threshold value for the maximum allowed body size
body_size_threshold = 0.1  # You can adjust this threshold as needed

# Set a threshold value for the maximum allowed shadow size
shadow_size_threshold = 0.5  # You can adjust this threshold as needed

# Identify Spinning Top candlestick patterns and assign binary values (0 or 1)
data['IsSpinningTop'] = ((data['PriceDifference'] <= body_size_threshold) &
                         (data['ShadowSize'] >= shadow_size_threshold)).astype(int)

# Print the rows where Spinning Top pattern is identified
spinning_top_patterns = data[data['IsSpinningTop'] == 1]
print(spinning_top_patterns)

data.to_csv(f"../data/AXISBANK.NS_{name}.csv")
train_filename = f"../data/AXISBANK.NS_train_{name}.csv"
test_filename = f"../data/AXISBANK.NS_test_{name}.csv"

train_perc = 0.8
total_rows = len(data)
train_rows_len = int(train_perc * total_rows)
test_rows_len = int((1 - train_perc) * total_rows)
data.head(train_rows_len).to_csv(train_filename)
data.tail(test_rows_len).to_csv(test_filename)

print(f"Done")