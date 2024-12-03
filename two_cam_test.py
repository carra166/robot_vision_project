from __future__ import print_function # Python 2/3 compatibility
import cv2 # Import the OpenCV library
import numpy as np # Import Numpy library


def main():
    # Create VideoCapture objects for each camera
    cap_right = cv2.VideoCapture(1) # Right Camera (below Headphones dongle)
    cap_left = cv2.VideoCapture(0)  # Left Camera (to right of Headphones dongle)

    # TODO: This might need to be changed or adjusted
    # Create the background subtractor object
    # Use the last 700 video frames to build the background
    back_sub = cv2.createBackgroundSubtractorMOG2(history=700,
                                                  varThreshold=25, detectShadows=True)

    # TODO: Adjust this as needed
    # Create kernel for morphological operation
    # You can tweak the dimensions of the kernel
    # e.g. instead of 20,20 you can try 30,30.
    kernel = np.ones((20, 20), np.uint8)

    while True:

        # Capture frame-by-frame
        ret_right, frame_right = cap_right.read()
        ret_left, frame_left = cap_left.read()

        # Use every frame to calculate the foreground mask and update the background
        fg_mask_right = back_sub.apply(frame_right)
        fg_mask_left = back_sub.apply(frame_left)

        # Close dark gaps in foreground object using closing
        fg_mask_right = cv2.morphologyEx(fg_mask_right, cv2.MORPH_CLOSE, kernel)
        fg_mask_left = cv2.morphologyEx(fg_mask_left, cv2.MORPH_CLOSE, kernel)

        # Remove salt and pepper noise with a median filter
        fg_mask_right = cv2.medianBlur(fg_mask_right, 5)
        fg_mask_left = cv2.medianBlur(fg_mask_left, 5)

        # TODO: Here is where we should adjust parameters to bound object
        # TODO: Both frames are filtered for smoothing by now, can do depth calculations here

        # Threshold the image to make it either black or white
        _right, fg_mask_right = cv2.threshold(fg_mask_right, 127, 255, cv2.THRESH_BINARY)
        _left, fg_mask_left = cv2.threshold(fg_mask_left, 127, 255, cv2.THRESH_BINARY)

        # Find the index of the largest contour and draw bounding box
        fg_mask_bb_right = fg_mask_right
        fg_mask_bb_left = fg_mask_left
        contours_right, hierarchy_right = cv2.findContours(fg_mask_bb_right, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        contours_left, hierarchy_left = cv2.findContours(fg_mask_bb_left, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        areas_right = [cv2.contourArea(c) for c in contours_right]
        areas_left = [cv2.contourArea(c) for c in contours_left]

        # If there are no contours in either image
        if (len(areas_right) < 1) or (len(areas_left) < 1):

            # Display the resulting frame
            cv2.imshow('frame right', frame_right)
            cv2.imshow('frame left', frame_left)

            # If "q" is pressed on the keyboard,
            # exit this loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Go to the top of the while loop
            continue

        else:
            # Find the largest moving object in each camera
            max_index_right = np.argmax(areas_right)
            max_index_left = np.argmax(areas_left)

        # Draw the bounding box
        cnt_right = contours_right[max_index_right]
        cnt_left = contours_left[max_index_left]
        x_right, y_right, w_right, h_right = cv2.boundingRect(cnt_right)
        x_left, y_left, w_left, h_left = cv2.boundingRect(cnt_left)
        cv2.rectangle(frame_right, (x_right, y_right), (x_right + w_right, y_right + h_right), (0, 255, 0), 3)
        cv2.rectangle(frame_left, (x_left, y_left), (x_left + w_left, y_left + h_left), (0, 255, 0), 3)

        # Draw circle in the center of the bounding box
        x2_right = x_right + int(w_right / 2)
        y2_right = y_right + int(h_right / 2)
        x2_left = x_left + int(w_left / 2)
        y2_left = y_left + int(h_left / 2)
        cv2.circle(frame_right, (x2_right, y2_right), 4, (0, 255, 0), -1)
        cv2.circle(frame_left, (x2_left, y2_left), 4, (0, 255, 0), -1)

        # Print the centroid coordinates (we'll use the center of the bounding box) on the image
        text_right = "x: " + str(x2_right) + ", y: " + str(y2_right)
        text_left = "x: " + str(x2_left) + ", y: " + str(y2_left)
        cv2.putText(frame_right, text_right, (x2_right - 10, y2_right - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame_left, text_left, (x2_left - 10, y2_left - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('frame right', frame_right)
        cv2.imshow('frame left', frame_left)

        # If "q" is pressed on the keyboard,
        # exit this loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Close down the video stream
    cap_right.release()
    cap_left.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    print(__doc__)
    main()