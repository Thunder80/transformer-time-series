import pandas as pd
from itertools import product
import torch
import os
from torch import nn, optim

from model import TransformerModel
from testing import test_model
from data import prepare_data
from training import train_model
from utils import create_empty_directory

"""
conf_file.txt
dataset - string
features - string[]
num_encoder_layers - number[]
num_decoder_layers - number[]
batch_size - number[]
input_sequence_length - number[]
output_sequence_length - number[]
"""

def train(features, nhead, num_encoder_layers, num_decoder_layers, batch_size, num_epochs, input_sequence_length, output_sequence_length, dataset, root_folder, stock, device):
    file_path = f"../data/{stock}/{dataset}/train.csv"
    feature_size = len(features)

    train_loader, time_series_data = prepare_data(input_sequence_length=input_sequence_length, output_sequence_length=output_sequence_length, file_path=file_path, batch_size=batch_size, feature_names=features, device=device, root_folder=root_folder)

    model = TransformerModel(feature_size, nhead, num_encoder_layers, num_decoder_layers).to(device)
    if os.path.isfile(f"{root_folder}/models/model_best.pt"):
        print("Found model")
        model.load_state_dict(torch.load(f"{root_folder}/models/model_best.pt"))

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    train_model(model, train_loader, time_series_data, criterion, optimizer, num_epochs, input_sequence_length, output_sequence_length, feature_size, device, root_folder=root_folder)

def test(features, nhead, num_encoder_layers, num_decoder_layers, batch_size, num_epochs, input_sequence_length, output_sequence_length, dataset, root_folder, stock):
    file_path = f"../data/{stock}/{dataset}/test.csv"

    _, time_series_data = prepare_data(input_sequence_length=input_sequence_length, output_sequence_length=output_sequence_length, file_path=file_path, batch_size=batch_size, feature_names=features, device=device, root_folder=root_folder)

    model = TransformerModel(feature_size, nhead, num_encoder_layers, num_decoder_layers).to(device)
    if os.path.isfile(f"{root_folder}/models/model_best.pt"):
        print("Found model")
        model.load_state_dict(torch.load(f"{root_folder}/models/model_best.pt"))

    criterion = nn.MSELoss()
    test_model(model, time_series_data, criterion, input_sequence_length, output_sequence_length, feature_size, device, root_folder=root_folder)


def all_models(conf_file):
    data = pd.read_csv(conf_file)

    device = torch.device("cpu")
    for index, row in data.iterrows():
        num_epochs = 1000
        dataset = row['dataset']
        features = row['features'].split("|")
        nheads = row['nhead'].split("|")
        num_encoder_layers = row['num_encoder_layers'].split("|")
        num_decoder_layers = row['num_decoder_layers'].split("|")
        batch_sizes = row['batch_size'].split("|")
        input_sequence_lengths = row['input_sequence_length'].split("|")
        output_sequence_lengths= row['output_sequence_length'].split("|")
        stocks = ["nifty", "axis", "tata"]

        total_combinations = product(range(len(nheads)), range(len(num_encoder_layers)), range(len(num_decoder_layers)), range(len(batch_sizes)), range(len(input_sequence_lengths)), range(len(output_sequence_lengths)), range(len(stocks)))

        i = 0
        create_empty_directory("all")
        for combo in total_combinations:
            nhead = int(nheads[combo[0]])
            num_encoder_layer = int(num_encoder_layers[combo[1]])
            num_decoder_layer = int(num_decoder_layers[combo[2]])
            batch_size = int(batch_sizes[combo[3]])
            input_sequence_length = int(input_sequence_lengths[combo[4]])
            output_sequence_length = int(output_sequence_lengths[combo[5]])
            stock = stocks[combo[6]]
            root_folder = f"all/{stock}_{dataset}_{i}"

            create_empty_directory(root_folder)
            train(features=features, nhead=nhead, num_encoder_layers=num_encoder_layer, num_decoder_layers=num_decoder_layer, batch_size=batch_size, num_epochs=num_epochs, input_sequence_length=input_sequence_length, output_sequence_length=output_sequence_length, dataset=dataset, root_folder=root_folder, stock=stock, device=device)
            i = i + 1


all_models("config.csv")
