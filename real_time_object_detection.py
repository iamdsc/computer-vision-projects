# import necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import imutils
import time
import cv2


# initialize the list of class labels MobileNet SSD was trained to detect
# generate a set of bounding box colors for each class
CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')

# initialize the video stream,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs = VideoStream(src='traffic.MTS').start()
fps = FPS().start()

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maxm width of 400 pixels
    try:
        frame = vs.read()
        frame = imutils.resize(frame, width=400)

        # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        # Resizing to a fixed 300x300 pixels and normalizing it.
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

        # pass the blob through the network and obtain the detections and predictions
        net.setInput(blob)
        detections = net.forward()

        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence associated with the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the confidence
            # is greater than minimum confidence
            if confidence > 0.2:
                # extract index of class label from the detections
                idx = int(detections[0, 0, i, 1])
                
                # compute the (x, y)-coordinates of the bounding box for the object
                box = detections[0, 0, i, 3:7]*np.array([w, h, w, h])
                # extract (x,y) coordinates of the box
                (startX, startY, endX, endY) = box.astype('int')

                # draw the prediction on the frame
                label = '{}:{:.2f}%'.format(CLASSES[idx], confidence*100)
                cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
                # we want label to be displayed above, but if there is no room
                # we will display just below the top
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
                # show the output frame
        cv2.imshow('Frame', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): # if 'q' key is pressed, break from the loop
            break
        # update FPS counter
        fps.update()
    except:
        break

# stop the timer and display FPS information
fps.stop()
print('[INFO] elapsed time: {:.2f}'.format(fps.elapsed()))
print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

cv2.destroyAllWindows()
vs.stop()
