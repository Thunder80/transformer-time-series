# data_preparation.py
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from joblib import dump
from utils import create_empty_directory

def prepare_data(input_sequence_length, output_sequence_length, file_path, batch_size, feature_names, device, root_folder):
    data = pd.read_csv(file_path)
    data = data.dropna()
    time_series_data = data[feature_names].values

    scaler = MinMaxScaler()
    scaler.fit(time_series_data)
    time_series_data = scaler.transform(time_series_data)
    time_series_data = torch.tensor(time_series_data, dtype=torch.float32, device=device)

    create_empty_directory(f"{root_folder}/joblib")
    dump(scaler, f"{root_folder}/joblib/scaler.joblib")

    total_samples = len(time_series_data)
    print("Total samples", total_samples)

    windows = []
    outputs = []

    for i in range(total_samples - input_sequence_length - output_sequence_length + 1):
        window = time_series_data[i : i + input_sequence_length]
        output = time_series_data[i + input_sequence_length : i + input_sequence_length + output_sequence_length]
        windows.append(window)
        outputs.append(output)

    data = torch.stack(windows)
    data_targets = torch.stack(outputs)

    print("Training input data shape:", data.shape)
    print("Training ouput data shape:", data_targets.shape)
    print("\n")

    # Create DataLoader
    dataset = TensorDataset(data, data_targets)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return data_loader, time_series_data
