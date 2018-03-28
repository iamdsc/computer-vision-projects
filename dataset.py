# for creating new dataset of users
import cv2
import os
import time

# checks for a directory and creates if not exists
def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

def create_update_dataset(face_id):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    vid = cv2.VideoCapture(0)
    f_id = face_id
    c=1
    assure_path_exists("dataset/")
    while(True):
        ret, frame = vid.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)
            roi_color = frame[y:y+h, x:x+w]
        cv2.imshow('frame', frame)
        if cv2.waitKey(10) & 0xff == 27: # Esc to click photos in different expressions
            cv2.imwrite("dataset/"+str(f_id)+"."+str(c)+".jpg", roi_color)
            c+=1
        elif cv2.waitKey(10) & 0xFF == ord('q'): # Press q to Exit
            break        
    vid.release()
    cv2.destroyAllWindows()   

# Enter ID of User
face_id = int(input("Enter User ID : "))
create_update_dataset(face_id)
