from predict import plot_for_window, predict_and_plot
import torch

def test_model(model, train_loader, time_series_data, criterion, optimizer, num_epochs, input_sequence_length, output_sequence_length):
    total_loss = 0.0
    model.eval()
    batch_no = 0
    with torch.no_grad():
        for batch_data, batch_targets in train_loader:
            # Forward pass~
            tgts = torch.zeros(batch_targets.shape)
            predictions = model(batch_data, tgts)
            
            # print(predictions[:, -output_sequence_length:, :].shape, batch_targets[:, -output_sequence_length:, :].shape)
            # Compute loss
            loss = criterion(predictions[:, -output_sequence_length:, :], batch_targets[:, -output_sequence_length:, :])  # Predict next timestep
            
            batch_data.shape
            total_loss += loss.detach().item()
            if batch_no % 30 == 0:
                plot_for_window(epoch=0, model=model, batch_data=batch_data, batch_targets=batch_targets, batch_no=batch_no, predictions=predictions, input_sequence_length=input_sequence_length, output_sequence_length=output_sequence_length, training=False)
            batch_no += 1
      
    print(f"Epoch 1/{num_epochs}, Loss: {total_loss:.4f}")

    print("Testing finished!")
    predict_and_plot(epoch=num_epochs, model=model, time_series_data=time_series_data, input_sequence_length=input_sequence_length, output_sequence_length=output_sequence_length, training=False)
    