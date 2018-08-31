# for creating training model for a new dataset
# uses LBPH Face Recognizer

import cv2
import numpy as np
from PIL import Image
import os


def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    print(imagePaths)
    faceSamples = []
    ids = []
    for imagePath in  imagePaths:
        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img,'uint8')
        id_=int(os.path.split(imagePath)[-1].split(".")[0])
        print(id_)
        faces = detector.detectMultiScale(img_numpy)
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id_)
    return faceSamples,ids

faces,ids = getImagesAndLabels('dataset')
recognizer.train(faces, np.array(ids))

assure_path_exists('trainer/')
recognizer.save('trainer/trainer.yml')
