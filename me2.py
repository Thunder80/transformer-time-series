import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Define the Transformer model
class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, nhead, num_encoder_layers, num_decoder_layers):
        super(TransformerModel, self).__init__()
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, batch_first=True), num_encoder_layers)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=input_dim, nhead=1, batch_first=True), num_decoder_layers)
        # self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers)
        self.fc = nn.Linear(d_model, d_model)
        self.output_layer = nn.Linear(input_sequence_length, 1)

        
    def forward(self, src, tgt):
        encoder_output = self.encoder(src)
        
        print(encoder_output.shape, tgt.shape)
        # output = self.output_layer(linear_output.squeeze(len(linear_output.size()) - 1))
        decoder_output = self.decoder(tgt, encoder_output)

        linear_output = self.fc(src)
        return linear_output

# Hyperparameters
input_dim = 4         # Number of input features
output_dim = 1        # Number of output features
d_model = 4           # Embedding dimension
nhead = 4             # Number of attention heads
num_encoder_layers = 1 # Number of encoder layers
num_decoder_layers = 1 # Number of decoder layers
lr = 0.01              # Learning rate
batch_size = 30        # Batch size
num_epochs = 100      # Number of training epochs
input_sequence_length = 20
output_sequence_length = 1

data = pd.read_csv("./data/nifty_data.csv")
data = data.dropna()
time_series_data = data[["Open", "High", "Low", "Close"]].values
scaler = MinMaxScaler()
time_series_data = scaler.fit_transform(time_series_data)
time_series_data = torch.tensor(time_series_data, dtype=torch.float32)

total_samples = len(time_series_data)
print("Total samples", total_samples)

windows = []
outputs = []

for i in range(total_samples - input_sequence_length - output_sequence_length + 1):
    window = time_series_data[i : i + input_sequence_length]
    output = time_series_data[i + input_sequence_length : i + input_sequence_length + output_sequence_length]  # Output is the last entry in the window
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
model = TransformerModel(input_dim, output_dim, d_model, nhead, num_encoder_layers, num_decoder_layers)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training loop
for epoch in range(num_epochs):
    total_loss = 0.0
    for batch_data, batch_targets in train_loader:
        optimizer.zero_grad()
        
        # print(batch_data.shape, batch_targets.shape)
        # Forward pass~
        predictions = model(batch_data, batch_targets)  # Exclude the last target for input to the decoder
        
        # Compute loss
        # print("Loss", predictions.shape, batch_targets.shape)
        loss = criterion(predictions, batch_targets)  # Predict next timestep
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

print("Training finished!")

model.eval()

predictions = []
predictions.append([time_series_data[0][3]])
prediction_ind = [0]
# Perform predictions
with torch.no_grad():
    for i in range(0, total_samples - input_sequence_length, input_sequence_length):
        new_src_data = time_series_data[i:i + input_sequence_length]
        new_tgt_data = torch.zeros(1)
        # print(new_src_data.shape, new_tgt_data.shape)
        pred = model(new_src_data, new_tgt_data)  # Predict based on previous outputs
        predictions.append(pred)
        prediction_ind.append(i + input_sequence_length)

# print(predictions[:5])
predictions = np.array(predictions)
print("Predicted values:", predictions)

print("Training data shape:", time_series_data.shape)
print("Predictions shape:", predictions.shape)


plt.plot(time_series_data[:, 3], color="red")
plt.plot(prediction_ind, predictions, color="blue")
plt.show()