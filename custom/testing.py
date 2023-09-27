from predict import predict
from plot import plot
import torch

def test_model(model, time_series_data, criterion, input_sequence_length, output_sequence_length, feature_size, device, root_folder):
    with torch.no_grad():
        prediction_ind, all_predictions, total_loss = predict(model=model, time_series_data=time_series_data, criterion=criterion, input_sequence_length=input_sequence_length, output_sequence_length=output_sequence_length, feature_size=feature_size, device=device)

        print(f"Total loss {total_loss}, Loss per sample: {total_loss / len(time_series_data)}")
        plot(time_series_data=time_series_data, prediction_ind=prediction_ind, predictions=all_predictions, title=f"Close price for test data", file_name_with_path=f"./{root_folder}/predictions/test/test.png", show=True)
    print("Testing finished!")
    