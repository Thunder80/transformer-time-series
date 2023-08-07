import sys
import torch

from global_data import VERSION_FILE, EMBED_SIZE, NUM_HEADS, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, SEQ_LENGTH
from nn import TransformerStockPrediction
from utils import read_number_from_file

model_version = read_number_from_file(VERSION_FILE)

if len(sys.argv) == 2:
    model_version = sys.argv[1]

model = TransformerStockPrediction(input_dim=4, output_dim=1, embed_size=EMBED_SIZE,
                                   num_heads=NUM_HEADS, num_encoder_layers=NUM_ENCODER_LAYERS,
                                   num_decoder_layers=NUM_DECODER_LAYERS)
model.load_state_dict(torch.load(f"models/model_{model_version}.pt"))
model.eval()

test_input = torch.tensor([
    [537, 542, 528, 532],
    [532, 538, 525, 529],
    [529, 535, 522, 526],
    [526, 532, 518, 530],
    [530, 536, 522, 534],
    [534, 540, 526, 536],
    [536, 542, 528, 530],
    [530, 536, 522, 528],
    [528, 534, 520, 524],
    [524, 530, 516, 520],
    [520, 526, 512, 518],
    [518, 524, 510, 516],
    [516, 522, 508, 512],
    [512, 518, 504, 508],
    [508, 514, 500, 504]
], dtype=torch.float32).transpose(0, 1).unsqueeze(0)  # Shape: (seq_length, batch_size=1, input_dim=4)

# Predict using the loaded model
with torch.no_grad():
    predicted_close = model(test_input[:-1], test_input[:-1])  # Prediction using the encoder input only
    predicted_close = predicted_close.squeeze(1)  # Remove the batch dimension
    print("Predicted Close Prices:", predicted_close.tolist())




