import torch
import numpy as np
from joblib import load

def predict(model, time_series_data, criterion, input_sequence_length, output_sequence_length, feature_size, device, root_folder):
    model.eval()
    predictions = []
    prediction_ind = []

    skip = output_sequence_length
    total_loss = 0.0  # Initialize the total loss for the epoch
    with torch.no_grad():
        for i in range(0, len(time_series_data) - input_sequence_length, skip):
            new_src_data = time_series_data[i:i + input_sequence_length].unsqueeze(0)
            new_tgt_data = time_series_data[i + input_sequence_length : i + input_sequence_length + output_sequence_length].unsqueeze(0)

            tgt = torch.zeros(1, output_sequence_length, feature_size, device=device)

            for j in range(output_sequence_length):
                prediction = model(new_src_data, tgt[:, :j+1, :])
                tgt[0, j] = prediction[0, -1]


            # print(tgt.shape, new_tgt_data.shape)
            loss = criterion(tgt[:, :new_tgt_data.shape[1], :], new_tgt_data)
            preds = tgt.cpu().numpy()

            j = 0
            for pred in preds[0, -output_sequence_length:]:
                predictions.append(pred)
                prediction_ind.append(i + j + input_sequence_length + 1)
                j += 1


            total_loss += loss

    predictions = np.array(predictions)

    scaler = load(f"{root_folder}/joblib/scaler.joblib")
    predictions = scaler.inverse_transform(predictions)

    return prediction_ind, predictions, total_loss


def predict_multi(model, time_series_data, criterion, input_sequence_length, output_sequence_length, feature_size, device, root_folder, daily_features, weekly_features, time_series_data_weekly):
    model.eval()
    predictions = []
    prediction_ind = []

    skip = output_sequence_length
    total_loss = 0.0  # Initialize the total loss for the epoch
    with torch.no_grad():
        for i in range(0, len(time_series_data) - input_sequence_length, skip):
            print(time_series_data.shape)
            new_src_data_daily = time_series_data[i:i + input_sequence_length].unsqueeze(0)
            new_tgt_data_daily = time_series_data[i + input_sequence_length : i + input_sequence_length + output_sequence_length].unsqueeze(0)


            new_src_data_weekly = time_series_data_weekly[i:i + input_sequence_length].unsqueeze(0)
            new_tgt_data_weekly = time_series_data_weekly[i + input_sequence_length : i + input_sequence_length + output_sequence_length].unsqueeze(0)

            tgt = torch.zeros(1, output_sequence_length, feature_size, device=device)
            
            for j in range(output_sequence_length):
                prediction = model(new_src_data_daily, tgt[:, :j+1, :], new_src_data_weekly, tgt[:, :j+1, :])
                tgt[0, j] = prediction[0, -1]


            # print(tgt.shape, new_tgt_data.shape)
            loss = criterion(tgt[:, :new_tgt_data_daily.shape[1], :], new_tgt_data_daily)
            preds = tgt.cpu().numpy()

            j = 0
            for pred in preds[0, -output_sequence_length:]:
                predictions.append(pred)
                prediction_ind.append(i + j + input_sequence_length + 1)
                j += 1


            total_loss += loss

    predictions = np.array(predictions)

    scaler = load(f"{root_folder}/joblib/scaler.joblib")
    predictions = scaler.inverse_transform(predictions)

    return prediction_ind, predictions, total_loss
