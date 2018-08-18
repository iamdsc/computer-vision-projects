# Detecting blinks with facial landmarks and opencv
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2


# Eye aspect ratio function
def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordiantes
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A+B)/(2.0 * C)
    
    return ear

# define 2 constants, one for the EAR to indicate blink
# and then a second for the no. of consecutive
# frames the eye must be below the threshold
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 2

# initialize the frame counters and total number of blinks
COUNTER = 0
TOTAL = 0

# intialize dlib's face detetor (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor..")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# grab the indexes of the facial landmarks for the left and right eye
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream thread
print("[INFO] starting video stream thread..")
vs = VideoStream(src=0).start()

# loop over frames from the video stream
while True:
    frame = vs.read()
    #frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    rects = detector(gray, 0)

    # loop over the face detections
    for rect in rects:
        # determine the facial landmarks for the face region,
        # then convert the facial landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # extract the left and right eye coordinates, then use the
        # coordinates to compute the ear for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # avg ear
        ear = (leftEAR + rightEAR)/2.0

        # compute the convex hull for the left and right eye,
        # then visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0,255,0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0,255,0), 1)
        
        # check to see if ear is below the blink threshold
        # and if so increment the blink frame counter        
        if ear < EYE_AR_THRESH:
            COUNTER += 1
        else:
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                TOTAL += 1
            COUNTER = 0

        # draw the total number of blinks on frame
        # along with computed ear for the frame
        cv2.putText(frame, "BLINKS: {}".format(TOTAL), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cv2.destroyAllWindows()
vs.stop()
