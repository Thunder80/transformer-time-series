from custom.predict import predict
from custom.plot import plot
import torch
from custom.utils import create_empty_directory

def test_model(model, time_series_data, criterion, input_sequence_length, output_sequence_length, feature_size, device, root_folder):
    create_empty_directory(f"{root_folder}/predictions/test")
    with torch.no_grad():
        prediction_ind, all_predictions, total_loss = predict(model=model, time_series_data=time_series_data, criterion=criterion, input_sequence_length=input_sequence_length, output_sequence_length=output_sequence_length, feature_size=feature_size, device=device, root_folder=root_folder)

        print(f"Total loss {total_loss}, Loss per sample: {total_loss / len(time_series_data)}")
        plot(time_series_data=time_series_data, prediction_ind=prediction_ind, predictions=all_predictions, title=f"Close price for test data", file_name_with_path=f"./{root_folder}/predictions/test/test.png", show=True, root_folder=root_folder)

        with open(f"{root_folder}/test_loss.txt", 'w') as file:
                file.write(f"Total loss = {total_loss}\nLoss per sample = {total_loss / len(time_series_data)}")
    print("Testing finished!")
    