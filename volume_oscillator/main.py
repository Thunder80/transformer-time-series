# main.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import os
from train_data import prepare_training_data
from model import TransformerModel
from training import train_model
from predict import predict_and_plot
from utils import clean_directories
from torch import nn, optim
import torch

def main():
    clean_directories()

    # Hyperparameters
    feature_size = 7
    nhead = 7
    num_encoder_layers = 3
    num_decoder_layers = 3
    # lr = 0.001
    batch_size = 5
    num_epochs = 100
    input_sequence_length = 30
    output_sequence_length = 7
    device = torch.device("mps")

    train_loader, time_series_data = prepare_training_data(input_sequence_length, output_sequence_length, "../data/doji/AXISBANK.NS_train_doji.csv", batch_size=batch_size, device=device)

    model = TransformerModel(feature_size, nhead, num_encoder_layers, num_decoder_layers).to(device)
    if os.path.isfile("models/model_best.pt"):
        print("Found model")
        model.load_state_dict(torch.load("models/model_best.pt"))

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    train_model(model, train_loader, time_series_data, criterion, optimizer, num_epochs, input_sequence_length, output_sequence_length)

if __name__ == "__main__":
    main()
