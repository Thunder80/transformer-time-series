import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("data/nifty_data.csv")
stock_prices = data[["Open", "High", "Low", "Close"]].values.astype(np.float32)
train = stock_prices[0:161]
test = stock_prices[161:]
num_days = len(train)

input_seq_len = 10
output_seq_len = 10
num_samples = num_days - input_seq_len - output_seq_len + 1
test_num_samples = len(test) - input_seq_len - output_seq_len + 1

# src = []
# for i in range(num_samples):
#     src.append(stock_prices[i: i + input_seq_len])
src_data = torch.tensor(np.array([train[i:i+input_seq_len] for i in range(num_samples)])).float()
tgt_data = torch.tensor(np.array([train[i+input_seq_len:i+input_seq_len+output_seq_len] for i in range(num_samples)])).float()

class StockPriceTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dropout):
        super(StockPriceTransformer, self).__init__()
        self.input_linear = nn.Linear(4, d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_layers, dropout=dropout)
        self.output_linear = nn.Linear(d_model, 4)

    def forward(self, src, tgt):
        src = self.input_linear(src)
        tgt = self.input_linear(tgt)
        output = self.transformer(src, tgt)
        output = self.output_linear(output)
        return output

d_model = 64
nhead = 4
num_layers = 2
dropout = 0.1

model = StockPriceTransformer(d_model, nhead, num_layers, dropout=dropout)

epochs = 1
lr = 0.001
batch_size = 16

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

train_output = []

for epoch in range(epochs):
    for i in range(0, num_samples, batch_size):
        src_batch = src_data[i:i+batch_size]
        tgt_batch = tgt_data[i:i+batch_size]

        optimizer.zero_grad()
        output = model(src_batch, tgt_batch)
        print(output.shape)
        print(output)
        break
        loss = criterion(output, tgt_batch)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

# src = torch.tensor(stock_prices[-input_seq_len:]).unsqueeze(-1).unsqueeze(1).float()
tgt = torch.zeros(output_seq_len, 1, 1)

src_data = torch.tensor(np.array([test[i:i+input_seq_len] for i in range(test_num_samples)])).float()
tgt_data = torch.tensor(np.array([test[i+input_seq_len:i+input_seq_len+output_seq_len] for i in range(test_num_samples)])).float()

with torch.no_grad():
    prediction = model(src_data, tgt_data)

print("Next 10 days of stock prices:", prediction)

train_output = np.array(train_output)
plt.figure(figsize=(10, 8))
plt.title("Stock data")
plt.plot(train[:, 3])
print(train_output.shape)
plt.plot(train_output[:, 3])
plt.show()