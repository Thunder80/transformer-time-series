import os
import torch

from custom.model import TransformerModel
from custom.training import train_model
from custom.data import prepare_data
from custom.utils import clean_directories

from torch import nn, optim


def main():
    # Hyperparameters
    feature_size = 5
    nhead = 5
    num_encoder_layers = 3
    num_decoder_layers = 3
    # lr = 0.001
    batch_size = 32
    num_epochs = 1000
    input_sequence_length = 40
    output_sequence_length = 7
    device = torch.device("mps")

    train_loader, time_series_data = prepare_data(input_sequence_length=input_sequence_length, output_sequence_length=output_sequence_length, file_path="../data/nifty/all_data/train.csv", batch_size=batch_size, feature_names=["Open", "High", "Low", "Close", "Volume"], device=device, root_folder="./results")

    model = TransformerModel(feature_size, nhead, num_encoder_layers, num_decoder_layers).to(device)
    if os.path.isfile("models/model_best.pt"):
        print("Found model")
        model.load_state_dict(torch.load("models/model_best.pt"))

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    train_model(model, train_loader, time_series_data, criterion, optimizer, num_epochs, input_sequence_length, output_sequence_length, feature_size, device, "./results", 0.1, 0.1)

if __name__ == "__main__":
    main()
