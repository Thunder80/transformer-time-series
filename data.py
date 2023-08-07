import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset

class CSVDataset(Dataset):
    def __init__(self, file_path, seq_length):
        self.seq_length = seq_length
        self.data = self.load_data(file_path)

    def load_data(self, file_path):
        data = pd.read_csv(file_path)
        # Extract OHLC values as separate sequences
        ohlc_data = data[["Open", "High", "Low", "Close"]].values.astype(np.float32)
        return ohlc_data

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        # Extract OHLC sequence for the current window
        input_tensor = torch.tensor(self.data[idx:idx + self.seq_length], dtype=torch.float32)

        # Prepare the target tensor (e.g., the next Close value as prediction)
        target_tensor = torch.tensor(self.data[idx + self.seq_length][3], dtype=torch.float32)

        return input_tensor, target_tensor