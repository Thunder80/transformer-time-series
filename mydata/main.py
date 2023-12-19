import os
import torch

from custom.model import MultiTimeHorizonTransformerModel
from custom.training import train_multi_model
from custom.data import prepare_data
from custom.utils import clear_prev_preds

from torch import nn, optim


def main():
    # Hyperparameters
    daily_features = ["Close", "RSI", "MACD"]
    weekly_features = ["wk_close", "wk_rsi", "wk_rsi"]
    feature_size = len(daily_features)
    nhead = 3
    num_encoder_layers = 3
    num_decoder_layers = 3
    # lr = 0.001
    batch_size = 32
    num_epochs = 500
    input_sequence_length = 40
    output_sequence_length = 7
    device = torch.device("mps")
    file_path = "../data/mydata/train_1.csv"
    workspace = "./results"

    clear_prev_preds(workspace)

    train_loader_daily, time_series_data = prepare_data(input_sequence_length=input_sequence_length, output_sequence_length=output_sequence_length, file_path=file_path, batch_size=batch_size, feature_names=daily_features, device=device, root_folder=workspace)

    train_loader_weekly, time_series_data_weekly = prepare_data(input_sequence_length=input_sequence_length, output_sequence_length=output_sequence_length, file_path=file_path, batch_size=batch_size, feature_names=weekly_features, device=device, root_folder=workspace)

    model = MultiTimeHorizonTransformerModel(feature_size, nhead, num_encoder_layers, num_decoder_layers).to(device)

    if os.path.isfile("models/model_best.pt"):
        print("Found model")
        model.load_state_dict(torch.load("models/model_best.pt"))

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    train_multi_model(model, train_loader_daily, train_loader_weekly, time_series_data, time_series_data_weekly, criterion, optimizer, num_epochs, input_sequence_length, output_sequence_length, feature_size, device, workspace, daily_features, weekly_features, 0.2, 0.2)

if __name__ == "__main__":
    main()
