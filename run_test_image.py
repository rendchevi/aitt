import os, argparse
import timeit
from glob import glob

import tensorflow as tf
import numpy as np
from skimage.io import imread

from src.utils.misc import image_classes_dict

def run(args):

    # Load test data
    test_data_paths = glob(os.path.join(args.test_data_dir, f"*.jpg"))
    test_data = []
    for path in test_data_paths:
        # Load image in dtype: uint8
        image = imread(path, as_gray = True)
        # Normalize image to [0.0, 1.0] with dtype: float32
        image = image / 255.0
        # Expand batch and channel dimension
        image = np.expand_dims(np.expand_dims(image, axis = 0), axis = -1).astype(np.float32)
        test_data.append(image)

    # Load the TFLite model and allocate tensors
    model = tf.lite.Interpreter(model_path = args.model_path)
    model.allocate_tensors()

    # Get input and output tensors
    input_details = model.get_input_details()
    output_details = model.get_output_details()

    input_shape = input_details[0]['shape']

    # Test the model on test data.
    print("Filename | Predicted Class | Prob (%) | Inference Time")
    for input_data, filename in zip(test_data, test_data_paths):

        st = timeit.default_timer()
        # Set current test data as input to model
        model.set_tensor(input_details[0]['index'], input_data)
        # Run inference
        model.invoke()
        # Get prediction
        prediction = model.get_tensor(output_details[0]['index'])
        # Get hard class index and its probability
        predicted_class_idx = int(np.argmax(prediction, axis = -1))
        predicted_class = image_classes_dict[predicted_class_idx]
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