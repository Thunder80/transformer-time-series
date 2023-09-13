import sys
sys.path.append('../custom')

import os
import torch
from torch import nn

from model import TransformerModel
from testing import test_model
from data import prepare_data


def test():
    # Hyperparameters
    features = ["Open", "High", "Low", "Close", "Volume", "Spinning_top"]
    feature_size = len(features)
    nhead = 3
    num_encoder_layers = 3
    num_decoder_layers = 3
    # lr = 0.001
    batch_size = 5
    input_sequence_length = 40
    output_sequence_length = 7
    device = torch.device("cpu")
    file_path = "../data/tata/doji/TATAMOTORS.NS_test_doji.csv"

    _, time_series_data = prepare_data(input_sequence_length=input_sequence_length, output_sequence_length=output_sequence_length, file_path=file_path, batch_size=batch_size, feature_names=features, device=device)

    model = TransformerModel(feature_size, nhead, num_encoder_layers, num_decoder_layers).to(device)
    if os.path.isfile("models/model_best.pt"):
        print("Found model")
        model.load_state_dict(torch.load("models/model_best 400.pt"))

    criterion = nn.MSELoss()
    test_model(model, time_series_data, criterion, input_sequence_length, output_sequence_length, feature_size, device)


test()
