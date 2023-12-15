import matplotlib.pyplot as plt
from joblib import load
import os
import numpy as np
import matplotlib

def plot(time_series_data, prediction_ind, predictions, title, file_name_with_path,root_folder, show = False):
    scaler = load(f"{root_folder}/joblib/scaler.joblib")
    time_series_data = scaler.inverse_transform(time_series_data.cpu())
    
    plt.clf()
    plt.plot(time_series_data[:, 0], color="red", label="Input")
    plt.plot(prediction_ind, predictions[:, 0], color="blue", label="Predictions", alpha=0.4)
    plt.title(title)
    plt.legend()

    plt.savefig(file_name_with_path)
    if show:
        plt.show()


def plot_for_window(epoch, batch_data, batch_targets, batch_no, predictions, output_sequence_length, root_folder, training = True):
    scaler = load(f"{root_folder}/joblib/scaler.joblib")
    print(f"Ploting epoch {epoch}, batch {batch_no}")
    for i in range(0, len(batch_data)):
        if training:
            os.makedirs(f"predictions/training/batch/epoch_{epoch}/batch_{batch_no}", exist_ok=True)
        else:
            os.makedirs(f"predictions/test/batch/epoch_{epoch}/batch_{batch_no}", exist_ok=True)

        src = batch_data[i].cpu().numpy()
        tgt = batch_targets[i].cpu().numpy()
        preds = predictions[i].cpu().detach().numpy()
        
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


def plot_loss(losses, title, file_name_with_path):
    plt.clf()
    plt.plot(losses, color="green", label="Loss")
    plt.title(title)
    plt.legend()

    plt.savefig(file_name_with_path)