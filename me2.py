import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import math
import time
import sys
import os
import shutil

# Define the Transformer model
class TransformerModel(nn.Module):
    def __init__(self, feature_size, nhead, num_encoder_layers, num_decoder_layers):
        super(TransformerModel, self).__init__()
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=feature_size, nhead=nhead, batch_first=True), num_encoder_layers)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=feature_size, nhead=nhead, batch_first=True), num_decoder_layers)
        # self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers)
        # self.fc = nn.Linear(d_model, d_model)
        # self.output_layer = nn.Linear(feature_size, 1)
        # self.init_weights()


    # def init_weights(self):
    #     initrange = 0.1    
    #     self.decoder.bias.data.zero_()
    #     self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, tgt):
        src_mask = src.shape[1] if len(src.shape) == 3 else src.shape[0]
        tgt_mask = tgt.shape[1] if len(tgt.shape) == 3 else tgt.shape[0]

        encoder_mask = self._generate_square_subsequent_mask(src_mask)
        decoder_mask = self._generate_square_subsequent_mask(tgt_mask)

        # encoder_output = self.encoder(src, encoder_mask)
        # decoder_output = self.decoder(tgt, encoder_output, decoder_mask)

        # print(encoder_mask)
        encoder_output = self.encoder(src, encoder_mask)
        # output = self.output_layer(encoder_output)
        decoder_output = self.decoder(tgt, encoder_output, decoder_mask)

        # print(output.shape)
        return decoder_output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

def predict(epoch):
    model.eval()
    predictions = []
    prediction_ind = []

    # Perform predictions
    skip = output_sequence_length
    with torch.no_grad():
        for i in range(0, total_samples - input_sequence_length, skip):
            new_src_data = time_series_data[i:i + input_sequence_length]
            new_tgt_data = time_series_data[i + output_sequence_length : i + input_sequence_length + output_sequence_length]
            preds = model(new_src_data, new_tgt_data)

            for pred in preds[-output_sequence_length:]:
                predictions.append(pred)
                prediction_ind.append(i + input_sequence_length + output_sequence_length)

    # print(predictions[:5])
    predictions = np.array(predictions)
    # print("Predicted values:", predictions)

    print("Training data shape:", time_series_data.shape)
    print("Predictions shape:", predictions.shape)

    plt.clf()
    plt.plot(time_series_data[:, 3], color="red")
    plt.plot(prediction_ind, predictions[:, 3], color="blue")
    plt.savefig(f'predictions/pred_{epoch}.png') 


def delete_directory(path):
    try:
        shutil.rmtree(path)
        print(f"Deleted directory: {path}")
    except Exception as e:
        print(f"Error deleting directory: {e}")

def create_empty_directory(path):
    try:
        os.makedirs(path, exist_ok=True)
        print(f"Created empty directory: {path}")
    except Exception as e:
        print(f"Error creating directory: {e}")

def clean():
    delete_directory("predictions/")
    create_empty_directory("predictions/")


clean()
# Hyperparameters
feature_size = 5 # Number of features
nhead = 5 # Number of attention heads
num_encoder_layers = 3 # Number of encoder layers
num_decoder_layers = 3 # Number of decoder layers
lr = 0.001 # Learning rate
batch_size = 5 # Batch size
num_epochs = 100 # Number of training epochs
input_sequence_length = 30
output_sequence_length = 5

data = pd.read_csv("./data/test.csv")
data = data.dropna()
time_series_data = data[["Open", "High", "Low", "Close", "Volume"]].values
scaler = MinMaxScaler()
time_series_data = scaler.fit_transform(time_series_data)
time_series_data = torch.tensor(time_series_data, dtype=torch.float32)

total_samples = len(time_series_data)
print("Total samples", total_samples)

windows = []
outputs = []

for i in range(total_samples - input_sequence_length - output_sequence_length + 1):
    window = time_series_data[i : i + input_sequence_length]
    output = time_series_data[i + output_sequence_length : i + input_sequence_length + output_sequence_length]
    windows.append(window)
    outputs.append(output)

train_data = torch.stack(windows)
train_targets = torch.stack(outputs)

print("Training input data shape:", train_data.shape)
print("Training ouput data shape:", train_targets.shape)
print("\n")


# Create DataLoader
train_dataset = TensorDataset(train_data, train_targets)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model
model = TransformerModel(feature_size, nhead, num_encoder_layers, num_decoder_layers)
# model = TransformerRaj(feature_size=4)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

if os.path.isfile("models/model_best.pt"):
    print("Found model")
    model.load_state_dict(torch.load("models/model_best.pt"))

# Training loop
for epoch in range(num_epochs):
    total_loss = 0.0

    model.train()
    for batch_data, batch_targets in train_loader:
        optimizer.zero_grad()
        
        # Forward pass~
        predictions = model(batch_data, batch_targets)
        
        # print(predictions.shape, batch_targets.shape)
        # Compute loss
        loss = criterion(predictions[:, -output_sequence_length:, :], batch_targets[:, -output_sequence_length:, :])  # Predict next timestep
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
        total_loss += loss.detach().item()
    
    if epoch % 10 == 0:
        predict(epoch)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

print("Training finished!")
predict(epoch=num_epochs)
# version = update_version("VERSION_FILE")
model_path = "models/" + f"/model_best.pt"
torch.save(model.state_dict(), model_path)