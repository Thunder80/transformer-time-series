# data_preparation.py
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from joblib import dump

def prepare_training_data(input_sequence_length, output_sequence_length, file_path, batch_size):
    data = pd.read_csv(file_path)
    data = data.dropna()
    month_data = pd.read_csv("../data/nifty_monthly.csv")
    month_data = month_data.dropna()

    time_series_data = data[["Open", "High", "Low", "Close", "Volume"]].values
    time_series_month_data = data[["Open", "High", "Low", "Close", "Volume"]].values

    scaler = MinMaxScaler()
    scaler.fit(time_series_data)
    time_series_data = scaler.transform(time_series_data)
    time_series_data = torch.tensor(time_series_data, dtype=torch.float32)

    dump(scaler, "./joblib/scaler.joblib")

    scaler.fit(time_series_month_data)
    time_series_month_data = scaler.transform(time_series_month_data)
    time_series_month_data = torch.tensor(time_series_month_data, dtype=torch.float32)

    total_samples = len(time_series_data)
    print("Total samples", total_samples)

    windows = []
    outputs = []

    for i in range(total_samples - input_sequence_length - output_sequence_length + 1):
        window = time_series_data[i : i + input_sequence_length]
        output = time_series_data[i + output_sequence_length : i + input_sequence_length + output_sequence_length]
        windows.append(window)
        outputs.append(output)

    for i in range()

    train_data = torch.stack(windows)
    train_targets = torch.stack(outputs)

    print("Training input data shape:", train_data.shape)
    print("Training ouput data shape:", train_targets.shape)
    print("\n")

    # Create DataLoader
    train_dataset = TensorDataset(train_data, train_targets)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, time_series_data
