# utils.py
import shutil
import os
import pandas as pd

def delete_directory(path):
    try:
        shutil.rmtree(path)
        print(f"Deleted directory: {path}")
    except Exception as e:
        print(f"Error deleting directory: {e}")

def create_empty_directory(path):
    try:
        os.makedirs(path, exist_ok=True)
        print(f"Created empty directory: {path}")
    except Exception as e:
        print(f"Error creating directory: {e}")

def clean_directories():
    delete_directory("predictions/")
    create_empty_directory("predictions/")
    create_empty_directory("predictions/training")
    create_empty_directory("predictions/training/tf")
    create_empty_directory("predictions/test")
    create_empty_directory("predictions/training/batch")
    create_empty_directory("predictions/test/batch")
    create_empty_directory("models/")

    delete_directory("joblib")
    create_empty_directory("joblib")

def convert_daily_to_monthly():
    daily_data = pd.read_csv('../data/nifty.csv')

    daily_data['Date'] = pd.to_datetime(daily_data['Date'])

    monthly_data = daily_data.resample('M', on='Date').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).reset_index()

    monthly_data.to_csv('../data/nifty_monthly.csv', index=False)
