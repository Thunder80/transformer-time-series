import sys
sys.path.append('../custom')

import os
import torch
from torch import nn

from model import TransformerModel, MultiTimeHorizonTransformerModel
from testing import test_model_multi
from data import prepare_data


def test():
    # Hyperparameters
    daily_features = ["Close", "RSI", "MACD"]
    weekly_features = ["wk_close", "wk_rsi", "wk_rsi"]
    feature_size = len(daily_features)
    nhead = 3
    num_encoder_layers = 3
    num_decoder_layers = 3
    # lr = 0.001
    batch_size = 32
    input_sequence_length = 40
    output_sequence_length = 7
    device = torch.device("cpu")
    file_path = "../data/mydata/test_1.csv"
    workspace = "./results"

    train_loader_daily, time_series_data = prepare_data(input_sequence_length=input_sequence_length, output_sequence_length=output_sequence_length, file_path=file_path, batch_size=batch_size, feature_names=daily_features, device=device, root_folder=workspace)

    train_loader_weekly, time_series_data_weekly = prepare_data(input_sequence_length=input_sequence_length, output_sequence_length=output_sequence_length, file_path=file_path, batch_size=batch_size, feature_names=weekly_features, device=device, root_folder=workspace)

    model = MultiTimeHorizonTransformerModel(feature_size, nhead, num_encoder_layers, num_decoder_layers).to(device)
    if os.path.isfile("results/models/model_best.pt"):
        print("Found model")
        model.load_state_dict(torch.load("results/models/model_best.pt"))

    criterion = nn.MSELoss()
    test_model_multi(model, time_series_data, criterion, input_sequence_length, output_sequence_length, feature_size, device, workspace, daily_features, weekly_features, time_series_data_weekly)


test()
