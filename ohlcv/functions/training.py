from functions.predict import predict
from functions.plot import plot, plot_for_window
import torch
import numpy as np
from joblib import load
import random

def train_model(model, train_loader, time_series_data, criterion, optimizer, num_epochs, input_sequence_length, output_sequence_length, feature_size):
    min_loss = 500
    scaler = load("./joblib/scaler.joblib")
    for epoch in range(num_epochs):
        total_loss = 0.0

        model.train()
        batch_no = 0
        teacher_forcing_preds = []
        initial_threshold = 0.8
        minimum_threshold = 0.0
        decreasing_factor = 0.02
        for batch_data, batch_targets in train_loader:
            optimizer.zero_grad()

            random_number = random.random()
            probability_threshold = initial_threshold


            tgt = torch.zeros(batch_targets[:, :-1, :].shape)
            if random_number < probability_threshold:
                tgt = batch_targets[:, -output_sequence_length+1:, :]
            # Forward pass~
            predictions = model(batch_data, tgt)
            # print(predictions.shape)

            # print(batch_targets[:, -output_sequence_length+1:, :], batch_targets)
            # Compute loss
            loss = criterion(predictions[:, :, 3], batch_targets[:, -output_sequence_length+1:, 3])  # Predict next timestep

            for pred in predictions:
                teacher_forcing_preds.append(pred[-1].detach())
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            batch_data.shape
            total_loss += loss.detach().item()
            # if epoch % 10 == 0 and batch_no % 500 == 0:
            #     plot_for_window(epoch=epoch, batch_data=batch_data, batch_targets=batch_targets, batch_no=batch_no, predictions=predictions,output_sequence_length=output_sequence_length)
            batch_no += 1
        
        teacher_forcing_preds = np.array(teacher_forcing_preds)
        teacher_forcing_preds = scaler.inverse_transform(teacher_forcing_preds)
        plot(time_series_data=time_series_data, prediction_ind=list(range(30, len(teacher_forcing_preds) + 30)), predictions=teacher_forcing_preds, title=f"Teacher forcing pred {epoch}", file_name_with_path=f"./predictions/training/tf/tf_{epoch}.png")

        if epoch % 10 == 0:
            prediction_ind, all_predictions, test_loss = predict(model=model, time_series_data=time_series_data, criterion=criterion, input_sequence_length=input_sequence_length, output_sequence_length=output_sequence_length, feature_size=feature_size)

            print(f"Test loss at {epoch} = {test_loss}")
            plot(time_series_data=time_series_data, prediction_ind=prediction_ind, predictions=all_predictions, title=f"Close price for {epoch}", 
            file_name_with_path=f"./predictions/training/epoch_{epoch}.png")
        
        if total_loss < min_loss:
            model_path = "models/" + f"/model_best.pt"
            torch.save(model.state_dict(), model_path)

            
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

    print("Training finished!")
    prediction_ind, all_predictions = predict(model=model, time_series_data=time_series_data, input_sequence_length=input_sequence_length, output_sequence_length=output_sequence_length, feature_size=feature_size)
    
    plot(time_series_data=time_series_data, prediction_ind=prediction_ind, predictions=all_predictions, title=f"Close price for {epoch}", file_name_with_path=f"./predictions/training/epoch_{epoch}.png")