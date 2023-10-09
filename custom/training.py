from custom.predict import predict
from custom.utils import create_empty_directory
from custom.plot import plot, plot_for_window, plot_loss
import torch
import numpy as np
from joblib import load
import random
from tqdm import tqdm 

def train_model(model, train_loader, time_series_data, criterion, optimizer, num_epochs, input_sequence_length, output_sequence_length, feature_size, device, root_folder, probability_decrease = 0.1, decrease_rate = 0.1):
    min_loss = 500
    decrease_after = num_epochs * decrease_rate
    losses = []
    scaler = load(f"{root_folder}/joblib/scaler.joblib")
    probability_thresold = 1.05
    create_empty_directory(f"{root_folder}/predictions/training/tf")
    create_empty_directory(f"{root_folder}/models")

 
    for epoch in tqdm(range(num_epochs),  desc="Training: "):
        total_loss = 0.0

        model.train()
        batch_no = 0
        teacher_forcing_preds = []
        

        sampled_count, non_sampled_count = 0, 0

        for batch_data, batch_targets in train_loader:
            optimizer.zero_grad()

            random_number = random.random()


            tgt = torch.zeros(batch_targets[:, :-1, :].shape, device=device)
            if random_number < probability_thresold:
                tgt = batch_targets[:, -output_sequence_length+1:, :]
                sampled_count += 1
            else:
                non_sampled_count += 1
                
            predictions = model(batch_data, tgt)

            # print(predictions.shape, batch_targets[:, -output_sequence_length+1:, :].shape)
            # Compute loss
            loss = criterion(predictions, batch_targets[:, 1:, :])  # Predict next timestep

            for pred in predictions:
                teacher_forcing_preds.append(pred[-1].cpu().detach().numpy())
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            total_loss += loss.detach().item()
            # if epoch % 10 == 0 and batch_no % 500 == 0:
            #     plot_for_window(epoch=epoch, batch_data=batch_data, batch_targets=batch_targets, batch_no=batch_no, predictions=predictions,output_sequence_length=output_sequence_length)
            batch_no += 1

        if epoch % decrease_after == 0:
            probability_thresold -= probability_decrease
            min_loss = min_loss + 1
        
        print(probability_thresold)
        teacher_forcing_preds = np.array(teacher_forcing_preds)
        teacher_forcing_preds = scaler.inverse_transform(teacher_forcing_preds)
        plot(time_series_data=time_series_data, prediction_ind=list(range(30, len(teacher_forcing_preds) + 30)), predictions=teacher_forcing_preds, title=f"Teacher forcing pred {epoch}", file_name_with_path=f"./{root_folder}/predictions/training/tf/tf_{epoch}.png", root_folder=root_folder)

        if epoch % 10 == 0:
            prediction_ind, all_predictions, test_loss = predict(model=model, time_series_data=time_series_data, criterion=criterion, input_sequence_length=input_sequence_length, output_sequence_length=output_sequence_length, feature_size=feature_size, device=device, root_folder=root_folder)

            print(f"Test loss at {epoch} = {test_loss}")
            plot(time_series_data=time_series_data, prediction_ind=prediction_ind, predictions=all_predictions, title=f"Close price for {epoch}", 
            file_name_with_path=f"./{root_folder}/predictions/training/epoch_{epoch}.png", root_folder=root_folder)
        
        if total_loss < min_loss:
            print("Model updated")
            model_path = f"{root_folder}/models/" + f"/model_best.pt"
            torch.save(model.state_dict(), model_path)
            min_loss = total_loss

        losses.append(total_loss)

        plot_loss(losses=losses, title=f"Loss after epoch {epoch}", file_name_with_path=f"./{root_folder}/predictions/training/loss.png")
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}, Loss per sample: {total_loss / len(time_series_data)} Sampled count: {sampled_count}, Non sampled count: {non_sampled_count}")

    print("Training finished!")
    prediction_ind, all_predictions, total_loss = predict(model=model, time_series_data=time_series_data, criterion=criterion, input_sequence_length=input_sequence_length, output_sequence_length=output_sequence_length, feature_size=feature_size, device=device, root_folder=root_folder)
    
    plot(time_series_data=time_series_data, prediction_ind=prediction_ind, predictions=all_predictions, title=f"Close price for {epoch}", file_name_with_path=f"./{root_folder}/predictions/training/epoch_{epoch}.png", root_folder=root_folder)