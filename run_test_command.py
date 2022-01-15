import os, argparse
import timeit
from glob import glob

import tflite_runtime.interpreter as tflite
import numpy as np
import librosa

from src.utils.helpers import load_mel_from_file
from src.utils.misc import command_classes_dict

def run(args):

    # Load test data
    test_data_paths = glob(os.path.join(args.test_data_dir, f"*.wav"))
    test_data = []
    for path in test_data_paths:
        # Load mel from filepath
        mel = load_mel_from_file(path)
        # Expand batch dimension
        mel = np.expand_dims(mel, axis = 0)

        test_data.append(mel)

    # Load the TFLite model and allocate tensors
    model = tflite.Interpreter(model_path = args.model_path)

    # Get input and output tensors
    input_details = model.get_input_details()
    output_details = model.get_output_details()

    # Test the model on test data.
    print("Filename | Predicted Class | Prob (%) | Inference Time")
    for input_data, filename in zip(test_data, test_data_paths):

        st = timeit.default_timer()
        # Set model input shape (because audio data shape are dynamic)
        model.resize_tensor_input(
            input_details[0]['index'], 
            input_data.shape,
        )
        model.allocate_tensors()
        # Set current test data as input to model
        model.set_tensor(input_details[0]['index'], input_data)
        # Run inference
        model.invoke()
        # Get prediction
        prediction = model.get_tensor(output_details[0]['index'])
        # Get hard class index and its probability
        predicted_class_idx = int(np.argmax(prediction, axis = -1))
        predicted_class = command_classes_dict[predicted_class_idx]
        prob = prediction[0][predicted_class_idx]
        # Save output
        et = timeit.default_timer() - st
        print(f"{filename} ==> {predicted_class} | {prob*100} % | {et} sec.")

    


if __name__ == "__main__":

    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type = str)
    parser.add_argument("--test_data_dir", type = str)

    args = parser.parse_args()
	
    # Run app
    run(args)