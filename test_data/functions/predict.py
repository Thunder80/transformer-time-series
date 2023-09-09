# prediction_plotting.py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
from joblib import load
import os

def predict(model, time_series_data, input_sequence_length, output_sequence_length, feature_size, training = True):
    model.eval()
    predictions = []
    prediction_ind = []

    skip = output_sequence_length
    with torch.no_grad():
        for i in range(0, len(time_series_data) - input_sequence_length, skip):
            new_src_data = time_series_data[i:i + input_sequence_length].unsqueeze(0)
            # new_tgt_data = time_series_data[i + output_sequence_length : i + input_sequence_length + output_sequence_length].unsqueeze(0)

            tgt = torch.zeros(1, output_sequence_length, feature_size)

            for j in range(output_sequence_length):
                prediction = model(new_src_data, tgt[:, :j+1, :])
                tgt[0, j] = prediction[0, -1]

            preds = tgt.cpu().numpy()

            for pred in preds[0, -output_sequence_length:]:
                predictions.append(pred)
                prediction_ind.append(i + input_sequence_length + 1)

            
            # preds = model(new_src_data, new_tgt_data)

            # preds = preds.cpu().detach().numpy()
            # for pred in preds[0,-output_sequence_length:]:
            #     predictions.append(pred)
            #     prediction_ind.append(i + input_sequence_length + output_sequence_length)

    predictions = np.array(predictions)

    scaler = load("./joblib/scaler.joblib")
    print(predictions.shape)
    predictions = scaler.inverse_transform(predictions)

    return prediction_ind, predictions

