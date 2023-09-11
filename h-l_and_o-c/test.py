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
    feature_size = 8
    nhead = 4
    num_encoder_layers = 3
    num_decoder_layers = 3
    # lr = 0.001
    batch_size = 5
    input_sequence_length = 30
    output_sequence_length = 7
    device = torch.device("cpu")

    _, time_series_data = prepare_data(input_sequence_length=input_sequence_length, output_sequence_length=output_sequence_length, file_path="../data/spinning_top/AXISBANK.NS_test_spinning_top.csv", batch_size=batch_size, feature_names=["Open", "High", "Low", "Close", "Volume", "PriceDifference", "ShadowSize", "IsSpinningTop"], device=device)

    model = TransformerModel(feature_size, nhead, num_encoder_layers, num_decoder_layers).to(device)
    if os.path.isfile("models/model_best.pt"):
        print("Found model")
        model.load_state_dict(torch.load("models/model_best.pt"))

    criterion = nn.MSELoss()
    test_model(model, time_series_data, criterion, input_sequence_length, output_sequence_length, feature_size, device)


test()
