import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np

# Define the Transformer-based model
class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, n_heads, n_layers, hidden_dim):
        super(TransformerModel, self).__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=n_heads, dim_feedforward=hidden_dim),
            num_layers=n_layers
        )
        self.fc = nn.Linear(input_dim, output_dim)
        self.output_layer = nn.Linear(sequence_length, 1)
        
    def forward(self, x):
        x = self.transformer(x)
        x = self.fc(x)
        x = self.output_layer(x.squeeze(2))
        return x

# Hyperparameters
input_dim = 4   # Number of input features
output_dim = 1  # Number of output features
n_heads = 2     # Number of attention heads
n_layers = 2    # Number of Transformer layers
hidden_dim = 128  # Hidden dimension in the feedforward network

# Create a sample dataset (replace this with your own dataset loading)
num_samples = 1000
sequence_length = 50
X = np.random.randn(num_samples, sequence_length, input_dim).astype(np.float32)
y = np.random.randn(num_samples, output_dim).astype(np.float32)

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train)
y_train_tensor = torch.tensor(y_train)
X_test_tensor = torch.tensor(X_test)
y_test_tensor = torch.tensor(y_test)

# Create DataLoader for training and testing
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize the model, loss function, and optimizer
model = TransformerModel(input_dim, output_dim, n_heads, n_layers, hidden_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        pred_y = model(batch_X)
        loss = criterion(pred_y, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# Testing loop
model.eval()
test_loss = 0
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        pred_y = model(batch_X)
        test_loss += criterion(pred_y, batch_y).item()

avg_test_loss = test_loss / len(test_loader)
print(f"Average Test Loss: {avg_test_loss:.4f}")

model.eval()

predictions = []
predictions.append([0])
prediction_ind = [0]
# Perform predictions
with torch.no_grad():
    for i in range(0, num_samples - sequence_length, sequence_length):
        new_src_data = X_train_tensor[i:i + sequence_length]
        new_tgt_data = torch.zeros(20, 4)
        print(new_src_data.shape, new_tgt_data.shape)
        pred = model(new_src_data, new_tgt_data)  # Predict based on previous outputs
        predictions.append(pred)
        prediction_ind.append(i + input_sequence_length)

# print(predictions[:5])
predictions = np.array(predictions)
print(predictions)

print("Training data shape:", time_series_data.shape)
print("Predictions shape:", predictions.shape)


plt.plot(time_series_data[:, 3], color="red")
plt.plot(prediction_ind, predictions, color="blue")
plt.show()