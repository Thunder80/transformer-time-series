import os
import torch

from functions.model import TransformerModel
from functions.training import train_model
from functions.predict import predict
from functions.data import prepare_data
from functions.utils import clean_directories

from torch import nn, optim


def main():
    clean_directories()

    # Hyperparameters
    feature_size = 5
    nhead = 5
    num_encoder_layers = 3
    num_decoder_layers = 3
    # lr = 0.001
    batch_size = 5
    num_epochs = 100
    input_sequence_length = 30
    output_sequence_length = 7
    device = torch.device("cpu")

    train_loader, time_series_data = prepare_data(input_sequence_length=input_sequence_length, output_sequence_length=output_sequence_length, file_path="../data/nifty.csv", batch_size=batch_size, feature_names=["Open", "High", "Low", "Close", "Volume"], device=device)

    model = TransformerModel(feature_size, nhead, num_encoder_layers, num_decoder_layers).to(device)
    if os.path.isfile("models/model_best.pt"):
        print("Found model")
        model.load_state_dict(torch.load("models/model_best.pt"))

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    train_model(model, train_loader, time_series_data, criterion, optimizer, num_epochs, input_sequence_length, output_sequence_length, feature_size)

if __name__ == "__main__":
    main()
