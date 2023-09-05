import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from joblib import dump
import matplotlib.pyplot as plt
from model import TransformerModel
import os

def plot():
    data = pd.read_csv("../data/AXISBANK.NS_train_spinning_top.csv")
    data = data.dropna()
    time_series_data = data[["Open", "High", "Low", "Close", "Volume", "PriceDifference", "ShadowSize", "IsSpinningTop"]].values

    scaler = MinMaxScaler()
    scaler.fit(time_series_data)
    time_series_data = scaler.transform(time_series_data)
    time_series_data = torch.tensor(time_series_data, dtype=torch.float32)

    dump(scaler, "./joblib/scaler.joblib")

    total_samples = len(time_series_data)
    print("Total samples", total_samples)

    windows = []
    outputs = []

    for i in range(total_samples - 30 - 7 + 1):
        window = time_series_data[i : i + 30]
        output = time_series_data[i + 7 : i + 30 + 7]
        windows.append(window)
        outputs.append(output)

    train_data = torch.stack(windows)
    train_targets = torch.stack(outputs)

    model = TransformerModel(8, 4, 3, 3)
    if os.path.isfile("models/model_best.pt"):
        print("Found model")
        model.load_state_dict(torch.load("models/model_best.pt", map_location=torch.device("cpu")))
    
    predictions = model(train_data[300], train_targets[300])
    

    
    plt.plot(train_data[300, :, 3])
    plt.plot(list(range(7, 37)), train_targets[300, :, 3], linestyle="dotted")
    plt.plot(list(range(30, 37)), predictions.detach().numpy()[-7:, 3], linestyle="dotted")

    plt.show()

plot()