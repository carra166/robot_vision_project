from __future__ import print_function # Python 2/3 compatibility
from cam_cal_img_grab import cropHorizontal
import cv2 # Import the OpenCV library
import numpy as np # Import Numpy library
import sys

REMAP_INTERPOLATION = cv2.INTER_LINEAR

if len(sys.argv) != 2:
    print("Syntax: {0} CALIBRATION_FILE".format(sys.argv[0]))
    sys.exit(1)

calibration = np.load(sys.argv[1], allow_pickle=False)
imageSize = tuple(calibration["imageSize"])
leftMapX = calibration["leftMapX"]
leftMapY = calibration["leftMapY"]
leftROI = tuple(calibration["leftROI"])
rightMapX = calibration["rightMapX"]
rightMapY = calibration["rightMapY"]
rightROI = tuple(calibration["rightROI"])

# Create VideoCapture objects for each camera
cap_right_id = 1
cap_left_id = 0
cap_right = cv2.VideoCapture(cap_right_id)  # Right Camera (below Headphones dongle)
cap_left = cv2.VideoCapture(cap_left_id)  # Left Camera (to right of Headphones dongle)
cam_w = cap_left.get(cv2.CAP_PROP_FRAME_WIDTH)
cam_h = cap_left.get(cv2.CAP_PROP_FRAME_HEIGHT)
DEPTH_VISUALIZATION_SCALE = cam_w + cam_h
crop_width = int(cam_w * 0.9)

stereoMatcher = cv2.StereoBM_create()
stereoMatcher.setMinDisparity(4)
stereoMatcher.setNumDisparities(128)
stereoMatcher.setBlockSize(21)
stereoMatcher.setROI1(leftROI)
stereoMatcher.setROI2(rightROI)
stereoMatcher.setSpeckleRange(16)
stereoMatcher.setSpeckleWindowSize(45)

while True:
    # Capture frame-by-frame
    if not (cap_left.grab() and cap_right.grab()):
        print("No more frames")
        break

    _, frame_l = cap_left.retrieve()
    frame_l = cropHorizontal(frame_l)
    _, frame_r = cap_right.retrieve()
    frame_r = cropHorizontal(frame_r)

    right_h, right_w = frame_r.shape[:2]
    left_h, left_w = frame_l.shape[:2]

    if (right_w, right_h) != imageSize:
        print("Right camera has different size than the calibration data")
        break

    if (left_w, left_h) != imageSize:
        print("Left camera has different size than the calibration data")
        break

    fixedRight = cv2.remap(frame_r, rightMapX, rightMapY, REMAP_INTERPOLATION)
    fixedLeft = cv2.remap(frame_l, leftMapX, leftMapY, REMAP_INTERPOLATION)

    grayLeft = cv2.cvtColor(fixedLeft, cv2.COLOR_BGR2GRAY)
    grayRight = cv2.cvtColor(fixedRight, cv2.COLOR_BGR2GRAY)
    depth = stereoMatcher.compute(grayLeft, grayRight)

    cv2.imshow('left', fixedLeft)
    cv2.imshow('right', fixedRight)
    cv2.imshow('depth', depth / DEPTH_VISUALIZATION_SCALE)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close down the video stream
cap_right.release()
cap_left.release()
cv2.destroyAllWindows()