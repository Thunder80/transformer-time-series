import pandas as pd
from datetime import datetime, timedelta
import ta

df = pd.read_csv('../data/all_data/AXISBANK.NS_all_data.csv')


def is_doji(row):
    return int(abs(row['Open'] - row['Close']) <= 0.1 * (row['High'] - row['Low']))

df['IsDoji'] = df.apply(is_doji, axis=1)

df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

quarterly_rsi_values = []

for i, row in df.iterrows():
    start_date = i - pd.DateOffset(months=3)
    end_date = i
    quarterly_data = df.loc[start_date:end_date]
    quarterly_rsi = ta.momentum.RSIIndicator(quarterly_data['Close'], window=14).rsi()
    quarterly_rsi_values.append(quarterly_rsi.iloc[-1])

# Add the quarterly RSI values to the DataFrame as a new column
df['QuarterlyRSI'] = quarterly_rsi_values


df.to_csv("../data/doji/AXISBANK.NS_all_doji.csv", index=False)