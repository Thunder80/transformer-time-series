import sys
sys.path.append('../custom')

import os
import torch

from model import TransformerModel
from training import train_model
from data import prepare_data
from utils import clean_directories

from torch import nn, optim


def main():
    clean_directories()

    # Hyperparameters
    features = ["Open", "High", "Low", "Close", "Volume", "Doji"]
    feature_size = len(features)
    nhead = 3
    num_encoder_layers = 3
    num_decoder_layers = 3
    # lr = 0.001
    batch_size = 32
    num_epochs = 1000
    input_sequence_length = 40
    output_sequence_length = 7
    device = torch.device("cpu")
    file_path = "../data/tata/doji/TATAMOTORS.NS_train_doji.csv"

    train_loader, time_series_data = prepare_data(input_sequence_length=input_sequence_length, output_sequence_length=output_sequence_length, file_path=file_path, batch_size=batch_size, feature_names=features, device=device)

    model = TransformerModel(feature_size, nhead, num_encoder_layers, num_decoder_layers).to(device)
    if os.path.isfile("models/model_best.pt"):
        print("Found model")
        model.load_state_dict(torch.load("models/model_best.pt"))

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    train_model(model, train_loader, time_series_data, criterion, optimizer, num_epochs, input_sequence_length, output_sequence_length, feature_size, device)

if __name__ == "__main__":
    main()

# 300 epoch loss 1.6