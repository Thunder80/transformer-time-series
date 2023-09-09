# test.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import os
from testing import test_model
from train_data import prepare_training_data
from model import TransformerModel
from utils import clean_directories
from torch import nn, optim
from predict import plot_for_window, predict_and_plot
import torch
import matplotlib.pyplot as plt
from joblib import load

def test30Days():
    feature_size = 8
    nhead = 4
    num_encoder_layers = 3
    num_decoder_layers = 3
    # lr = 0.001
    batch_size = 5
    num_epochs = 100
    input_sequence_length = 30
    output_sequence_length = 7

    device = torch.device("cpu")
    train_loader, time_series_data = prepare_training_data(input_sequence_length, output_sequence_length, "../data/AXISBANK.NS_test_spinning_top.csv", batch_size=batch_size, device=device)

    model = TransformerModel(feature_size, nhead, num_encoder_layers, num_decoder_layers)
    if os.path.isfile("models/model_best.pt"):
        print("Found model")
        model.load_state_dict(torch.load("models/model_best.pt", map_location=device))

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    with torch.no_grad():
        batch_data = time_series_data[366:396, :]
        batch_targets = time_series_data[403:433, :]
        predictions = model(batch_data, batch_targets)
                
        print(predictions.shape, batch_targets.shape)
        # Compute loss:,
        # loss = criterion(predictions[-output_sequence_length:, :], batch_targets[-output_sequence_length:, :])  # Predict next timestep
        
        # total_loss += loss.detach().item()
        # if batch_no % 30 == 0:
        #     plot_for_window(epoch=0, model=model, batch_data=batch_data, batch_targets=batch_targets, batch_no=batch_no, predictions=predictions, input_sequence_length=input_sequence_length, output_sequence_length=output_sequence_length, training=False)
        # batch_no += 1
        
        # print(f"Epoch 1/{num_epochs}, Loss: {total_loss:.4f}")

        scaler = load("./joblib/scaler.joblib")
        batch_data = scaler.inverse_transform(batch_data)
        batch_targets = scaler.inverse_transform(batch_targets)
        predictions = scaler.inverse_transform(predictions)
        prediction_ind = [31, 32, 33, 34, 35, 36, 37]
        print("Testing finished!")
        plt.plot(batch_data[:, 3], color="red", label="Data")
        plt.plot(list(range(7, 37)), batch_targets[:, 3], color="pink", label="Target")
        plt.plot(list(range(30, 37)), predictions[-output_sequence_length:, 3], color="blue", label="Predictions")
        plt.title(f"Close price predictions epoch")
        plt.legend()
        plt.show()

def main():
    # Hyperparameters
    feature_size = 8
    nhead = 4
    num_encoder_layers = 3
    num_decoder_layers = 3
    # lr = 0.001
    batch_size = 5
    num_epochs = 100
    input_sequence_length = 30
    output_sequence_length = 7

    device = torch.device("cpu")
    train_loader, time_series_data = prepare_training_data(input_sequence_length, output_sequence_length, "../data/spinning_top/AXISBANK.NS_test_spinning_top.csv", batch_size=batch_size, device=device)

    model = TransformerModel(feature_size, nhead, num_encoder_layers, num_decoder_layers)
    if os.path.isfile("models/model_best.pt"):
        print("Found model")
        model.load_state_dict(torch.load("models/model_best.pt", map_location=device))

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    test_model(model, train_loader, time_series_data, criterion, optimizer, num_epochs, input_sequence_length, output_sequence_length)

main()
# test30Days()