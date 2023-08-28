# prediction_plotting.py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
from joblib import load
import os

def predict_and_plot(epoch, model, time_series_data, input_sequence_length, output_sequence_length):
    model.eval()
    predictions = []
    prediction_ind = []

    # Perform predictions
    skip = output_sequence_length
    with torch.no_grad():
        for i in range(0, len(time_series_data) - input_sequence_length, skip):
            new_src_data = time_series_data[i:i + input_sequence_length]
            new_tgt_data = time_series_data[i + output_sequence_length : i + input_sequence_length + output_sequence_length]
            preds = model(new_src_data, new_tgt_data)

            for pred in preds[-output_sequence_length:]:
                predictions.append(pred)
                prediction_ind.append(i + input_sequence_length + output_sequence_length)

    # print(predictions[:5])
    predictions = np.array(predictions)
    # print("Predicted values:", predictions)

    # print("Training data shape:", time_series_data.shape)
    # print("Predictions shape:", predictions.shape)

    scaler = load("./joblib/scaler.joblib")
    time_series_data = scaler.inverse_transform(time_series_data)
    predictions = scaler.inverse_transform(predictions)
    
    plt.clf()
    plt.plot(time_series_data[:, 3], color="red", label="Input")
    plt.plot(prediction_ind, predictions[:, 3], color="blue", label="Predictions")
    plt.title(f"Close price predictions epoch {epoch}")
    plt.legend()
    plt.savefig(f'predictions/training/pred_{epoch}.png') 

def plot_for_window(epoch, model, batch_data, batch_targets, batch_no, predictions, input_sequence_length, output_sequence_length):
    model.eval()

    scaler = load("./joblib/scaler.joblib")
    print(f"Ploting epoch {epoch}, batch {batch_no}")
    for i in range(0, len(batch_data)):
        os.makedirs(f"predictions/batch/epoch_{epoch}/batch_{batch_no}", exist_ok=True)
        src = batch_data[i].numpy()
        tgt = batch_targets[i].numpy()
        preds = predictions[i].detach().numpy()
        
        src = scaler.inverse_transform(src)
        tgt = scaler.inverse_transform(tgt)
        preds = scaler.inverse_transform(preds)
        
        true_tgt = np.concatenate((src[:, 3], tgt[-output_sequence_length:, 3]))
        true_preds = np.concatenate((src[:, 3], preds[-output_sequence_length:, 3]))
        matplotlib.use("Agg")
        plt.clf()
        plt.plot(src[:, 3], color="blue", label="input", linewidth=3)
        plt.plot(true_preds, color="green", label="predictions", marker="o")
        plt.plot(true_tgt, color="red", label="target", linestyle="dotted", marker="o")
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Close price")
        plt.title(f"Close price prediction epoch_{epoch} batch_{batch_no} {i}")
        if training:
            plt.savefig(f'predictions/training/batch/epoch_{epoch}/batch_{batch_no}/pred_epoch_{epoch}_batch_{batch_no}_{i}.png') 
        else:
            plt.savefig(f'predictions/test/batch/epoch_{epoch}/batch_{batch_no}/pred_epoch_{epoch}_batch_{batch_no}_{i}.png') 

def plot_for_test(epoch, model, batch_data, batch_targets, batch_no, predictions, input_sequence_length, output_sequence_length):
    model.eval()

    scaler = load("./joblib/scaler.joblib")
    print(f"Ploting epoch {epoch}, batch {batch_no}")
    for i in range(0, len(batch_data)):
        src = batch_data[i].numpy()
        tgt = batch_targets[i].numpy()
        preds = predictions[i].detach().numpy()
        
        src = scaler.inverse_transform(src)
        tgt = scaler.inverse_transform(tgt)
        preds = scaler.inverse_transform(preds)
        
        true_tgt = np.concatenate((src[:, 3], tgt[-output_sequence_length:, 3]))
        true_preds = np.concatenate((src[:, 3], preds[-output_sequence_length:, 3]))
        matplotlib.use("Agg")
        plt.clf()
        plt.plot(src[:, 3], color="blue", label="input", linewidth=3)
        plt.plot(true_preds, color="green", label="predictions", marker="o")
        plt.plot(true_tgt, color="red", label="target", linestyle="dotted", marker="o")
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Close price")
        plt.title(f"Close price prediction epoch_{epoch} batch_{batch_no} {i}")
        if training:
            plt.savefig(f'predictions/training/batch/epoch_{epoch}/batch_{batch_no}/pred_epoch_{epoch}_batch_{batch_no}_{i}.png') 
        else:
            plt.savefig(f'predictions/test/epoch_{epoch}/batch_{batch_no}/pred_epoch_{epoch}_batch_{batch_no}_{i}.png') 

