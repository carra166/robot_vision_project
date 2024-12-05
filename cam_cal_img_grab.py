from __future__ import print_function # Python 2/3 compatibility
import cv2 # Import the OpenCV library
import numpy as np # Import Numpy library
import time

# Create VideoCapture objects for each camera
cap_right_id = 1
cap_left_id = 0
cap_right = cv2.VideoCapture(cap_right_id)  # Right Camera (below Headphones dongle)
cap_left = cv2.VideoCapture(cap_left_id)  # Left Camera (to right of Headphones dongle)
cam_w = cap_left.get(cv2.CAP_PROP_FRAME_WIDTH)
cam_h = cap_left.get(cv2.CAP_PROP_FRAME_HEIGHT)

# Filenames are just an increasing number
d = 0

# Discard edges to improve calibration
crop_width = int(cam_w * 0.9)
def cropHorizontal(image):
    return image[:,
            int((cam_w - crop_width) / 2):
            int(crop_width + (cam_w - crop_width) / 2)]

#seconds = 10
#t_end = time.time() + seconds
#while time.time() < t_end:
while True:
    # Capture frame-by-frame
    if not (cap_left.grab() and cap_right.grab()):
        print("No more frames")
        break

    _, frame_l = cap_left.retrieve()
    frame_l = cropHorizontal(frame_l)
    _, frame_r = cap_right.retrieve()
    frame_r = cropHorizontal(frame_r)

    # Different directories for each camera
    LEFT_PATH = "capture/left/%d.jpg"%d
    RIGHT_PATH = "capture/right/%d.jpg"%d
    # LEFT_PATH = "capture/left/1.jpg"
    # RIGHT_PATH = "capture/right/1.jpg"

    # Save the frames
    cv2.imwrite(LEFT_PATH, frame_l)
    cv2.imwrite(RIGHT_PATH, frame_r)

    # Display the resulting frames
    cv2.imshow('frame right', frame_r)
    cv2.imshow('frame left', frame_l)

    # If "q" is pressed on the keyboard,
    # exit this loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    d += 1

# Close down the video stream
cap_right.release()
cap_left.release()
cv2.destroyAllWindows()