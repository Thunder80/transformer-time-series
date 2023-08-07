import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# custom imports
from nn import TransformerStockPrediction
from data import CSVDataset
from utils import update_version
from global_data import SEQ_LENGTH, BATCH_SIZE, NUM_EPOCHS, EMBED_SIZE, NUM_HEADS, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, LEARNING_RATE, VERSION_FILE, MODELS_PATH, TRAIN_FILE_PATH

csv_dataset = CSVDataset(file_path=TRAIN_FILE_PATH, seq_length=SEQ_LENGTH)
train_size = int(0.8 * len(csv_dataset))
test_size = len(csv_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(csv_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
# i = iter(train_loader)
# data = next(i)
# features, labels = data
# print(features, labels)

# Initialize the model and optimizer
model = TransformerStockPrediction(input_dim=4, output_dim=1, embed_size=EMBED_SIZE,
                                   num_heads=NUM_HEADS, num_encoder_layers=NUM_ENCODER_LAYERS,
                                   num_decoder_layers=NUM_DECODER_LAYERS)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_function = nn.MSELoss()

# Training loop
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0

    for input_seq, target_close in train_loader:
        optimizer.zero_grad()

        # Move tensors to GPU if available
        input_seq = input_seq.transpose(0, 1)# Transpose to (seq_length, batch_size, input_dim)
        target_close = target_close.unsqueeze(dim=1)# Add a dimension for output_dim

        # Forward pass
        predicted_close = model(input_seq, input_seq)  # Using the same input for encoder and decoder
        loss = loss_function(predicted_close, target_close)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}")

version = update_version(VERSION_FILE)
model_path = MODELS_PATH + f"/model_{version}.pt"
torch.save(model.state_dict(), model_path)