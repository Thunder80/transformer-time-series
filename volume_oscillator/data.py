import pandas as pd
from datetime import datetime, timedelta
import ta

df = pd.read_csv('../data/all_data/AXISBANK.NS_all_data.csv')


short_period = 12
long_period = 26

df['Short_MA'] = df['Volume'].rolling(window=short_period).mean()
df['Long_MA'] = df['Volume'].rolling(window=long_period).mean()

# Calculate the Volume Oscillator
df['Volume_Osc'] = 100 * ((df['Short_MA'] - df['Long_MA']) / df['Long_MA'])


df.to_csv("../data/volume_oscillator/AXISBANK.NS_all_volume_oscillator.csv", index=False)