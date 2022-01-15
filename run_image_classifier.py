# Import utilities
import os, argparse

# Import image processing libraries
import numpy as np
import cv2
from skimage.transform import resize
from skimage.color import rgb2gray

# Import helper functions
from src.utils.helpers import bgr_to_rgb

def run_app(args):

	# Instantiate camera object
	camera = cv2.VideoCapture(0)
	
	# Create app window
	cv2.namedWindow("MyApp", cv2.WINDOW_NORMAL) 

	# Camera streaming loop
	while camera.isOpened():
		  
		# Capture camera frame
		return_value, frame = camera.read()

		### Perform your image processing here ###
		
		# 0.   Meta information fetching
		# 0.1. Get frame shape
		num_row, num_col, num_channels = frame.shape
		# 0.2. Define region-of-interest (ROI) coordinates
		center_row, center_col = num_row // 2, num_col // 2
		x0 = center_row - (args.roi_size // 2)
		xt = center_row + (args.roi_size // 2)
		y0 = center_col - (args.roi_size // 2)
		yt = center_col + (args.roi_size // 2)
		# 0.3. Annotate the ROI
		cv2.rectangle(frame, (y0, x0), (yt, xt), color = (255, 0, 0))

		# 1.   Preprocess the ROI frame
		# 1.1. Get the ROI frame
		frame_roi = frame[x0:xt, y0:yt, :].copy()
		# 1.2. Switch the channels order from BGR -> RGB
		frame_roi = bgr_to_rgb(frame_roi)
		# 1.3. Convert RGB to Grayscale
		frame_roi = np.expand_dims(rgb2gray(frame_roi), axis = -1)
		# 1.4. Convert Grayscale to binary image
		frame_roi[frame_roi < args.thresh_binary] = 0.0
		frame_roi[frame_roi >= args.thresh_binary] = 1.0
		frame[x0:xt, y0:yt, :] = frame_roi.astype(np.uint8) * 255
		# 1.4. Resize to match the classifier's input size
		frame_roi = resize(frame_roi, (args.input_size, args.input_size))
		# 1.5. Forward-pass to the classifier
		prediction = "Hotdogs"
		# 1.6. Display prediction
		cv2.putText(frame, prediction, (y0, x0), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
		
		# Display the resulting frame
		cv2.imshow("MyApp", frame)
		  
		# Define exit button as "q"
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	  
	# After the loop release the camera object
	camera.release()
	# Destroy all the windows
	cv2.destroyAllWindows()
	
if __name__ == "__main__":

	# Arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("--roi_size", type = int, default = 256)
	parser.add_argument("--input_size", type = int, default = 28)
	parser.add_argument("--thresh_binary", type = float, default = 0.5)
	
	args = parser.parse_args()
	
	# Run app
	run_app(args)