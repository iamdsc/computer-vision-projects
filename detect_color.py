""" Author: Dilpreet Singh Chawla """
""" Github: iamdsc """

# importing the required modules
import numpy as np
import cv2

# capturing the video
# 0: using the webcam
# video.mp4 for using a file
vid = cv2.VideoCapture(0)

while True:
    _, frame = vid.read()     # Reading the frame

    # Using HSV (Hue, Saturation, Value) color format
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # let us detect red, blue, yellow and pink colors
    # Steps :-
    # 1. defining the color boundaries for the respective colors in the HSV color space
    # 2. find colors within specified boundaries and apply the mask
    # 3. remove noise from the frame by blurring
    # 3. show color frame
    
    ## Detecting red color
    lower_red = np.array([-10,50,50])
    upper_red = np.array([10,255,255])
    red_mask = cv2.inRange(hsv, lower_red, upper_red)
    red_output = cv2.bitwise_and(frame, frame, mask = red_mask)
    red_median_blur = cv2.medianBlur(red_output, 15)
    cv2.imshow('red_frame', red_median_blur)

    ## Detecting blue color
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    blue_output = cv2.bitwise_and(frame, frame, mask = blue_mask)
    blue_median_blur = cv2.medianBlur(blue_output, 15)
    cv2.imshow('blue_frame', blue_median_blur)

    ## Detecting yellow color
    lower_yellow = np.array([20,50,50])
    upper_yellow = np.array([40,255,255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_output = cv2.bitwise_and(frame, frame, mask = yellow_mask)
    yellow_median_blur = cv2.medianBlur(yellow_output, 15)
    cv2.imshow('yellow_frame', yellow_median_blur)

    ## Detecting pink color
    lower_pink = np.array([145,50,50])
    upper_pink = np.array([155,255,255])
    pink_mask = cv2.inRange(hsv, lower_pink, upper_pink)
    pink_output = cv2.bitwise_and(frame, frame, mask = pink_mask)
    pink_median_blur = cv2.medianBlur(pink_output, 15)
    cv2.imshow('pink_frame', pink_median_blur)

    if cv2.waitKey(5) & 0xFF == 27:
        break

vid.release()   # Releasing Webcam
cv2.destroyAllWindows()     # Closing all windows
