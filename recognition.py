import cv2
import numpy as np
import os

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

recognizer = cv2.face.LBPHFaceRecognizer_create()
assure_path_exists("trainer/")

# Load the trained model
recognizer.read('trainer/trainer.yml')

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

img = cv2.imread("unknown2.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(gray, 1.3, 5)
for(x,y,w,h) in faces:
    id_, confidence = recognizer.predict(gray[y:y+h,x:x+w])
    if id_ >= 11 and id_ <= 15:  
        print("User 1")
    elif id_ >= 21 and id_ <= 25:
        print("User 2")
    elif id_ >= 31 and id_ <=35:
        print("User 3")
    elif id_ >= 41 and id_ <=45:
        print("User 4")
    elif id_ >= 51 and id_ <=55:
        print("User 5")
    elif id_ >= 61 and id_ <= 65:
        print("User 6")
    else:
        print("Unknown")

if len(faces) == 0:
    print("No faces found")
