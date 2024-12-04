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

# Different directories for each camera
LEFT_PATH = "capture/left/{:06d}.jpg"
RIGHT_PATH = "capture/right/{:06d}.jpg"

# Filenames are just an increasing number
frameId = 0

minutes = 1
t_end = time.time() + 60 * minutes
while time.time() < t_end:
    # Capture frame-by-frame
    ret_r, frame_r = cap_right.read()
    ret_l, frame_l = cap_left.read()

    # Actually save the frames
    cv2.imwrite(LEFT_PATH.format(frameId), frame_l)
    cv2.imwrite(RIGHT_PATH.format(frameId), frame_r)
    frameId += 1