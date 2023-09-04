from predict import predict_and_plot, plot_for_window
import torch

def train_model(model, train_loader, time_series_data, criterion, optimizer, num_epochs, input_sequence_length, output_sequence_length):
    min_loss = 500
    for epoch in range(num_epochs):
        total_loss = 0.0

        model.train()
        batch_no = 0
        for batch_data, batch_targets in train_loader:
            optimizer.zero_grad()
            
            # Forward pass~
            predictions = model(batch_data, batch_targets)
            
            # print(predictions[:, -output_sequence_length:, :].shape, batch_targets[:, -output_sequence_length:, :].shape)
            # Compute loss
            loss = criterion(predictions[:, -output_sequence_length:, :], batch_targets[:, -output_sequence_length:, :])  # Predict next timestep
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            batch_data.shape
            total_loss += loss.detach().item()
            if epoch % 10 == 0 and batch_no % 500 == 0:
                plot_for_window(epoch=epoch, model=model, batch_data=batch_data, batch_targets=batch_targets, batch_no=batch_no, predictions=predictions, input_sequence_length=input_sequence_length, output_sequence_length=output_sequence_length)
            batch_no += 1
        
        if epoch % 10 == 0:
            predict_and_plot(epoch=epoch, model=model, time_series_data=time_series_data, input_sequence_length=input_sequence_length, output_sequence_length=output_sequence_length)
        
        if total_loss < min_loss:
            model_path = "models/" + f"/model_best.pt"
            torch.save(model.state_dict(), model_path)

            
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

    print("Training finished!")
    predict_and_plot(epoch=num_epochs, model=model, time_series_data=time_series_data, input_sequence_length=input_sequence_length, output_sequence_length=output_sequence_length)
    
    