import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from joblib import dump
import matplotlib.pyplot as plt
from model import TransformerModel
import os
import numpy as np

def plot():
    data = pd.read_csv("../data/spinning_top/AXISBANK.NS_test_spinning_top.csv")
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

    # plt.plot(time_series_data[:, 3])

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
    
    # window_no = 310
    model.eval()
    with torch.no_grad():
        predictions = []
        all_predictions = []
        prediction_ind = []
        window_no = 351

        # window prediction
        tgts = train_targets[window_no].detach().clone()
        tgts[-7:] = 0
        predictions = model(train_data[window_no], tgts)
        print(predictions.shape)


        # all predictions
        for i in range(0, len(train_data)):
            new_src_data = train_data[i]
            new_tgt_data = train_targets[i]
            tgts = new_tgt_data.detach().clone()
            tgts[-7:] = 0

            preds = model(new_src_data, tgts)

            for pred in preds[-7:]:
                all_predictions.append(pred)
                prediction_ind.append(i + 31)


        time_series_data = scaler.inverse_transform(time_series_data)
        train_data = scaler.inverse_transform(train_data[window_no])
        train_targets = scaler.inverse_transform(train_targets[window_no])
        predictions = scaler.inverse_transform(predictions)
        all_predictions = scaler.inverse_transform(all_predictions)


        plt.plot(time_series_data[:, 3], color="orange", label="Original")
        plt.plot(prediction_ind, all_predictions[:, 3], color="blue", label="Predictions")

        plt.plot(list(range(window_no, window_no + 30)), train_data[:, 3], color="red")
        plt.plot(list(range(window_no + 7, window_no + 37)), train_targets[:, 3], linestyle="dotted", color="pink")
        plt.plot(list(range(window_no + 30, window_no + 37)), predictions[-7:, 3], color="green", marker="o")
        plt.legend()
        plt.show()

plot()