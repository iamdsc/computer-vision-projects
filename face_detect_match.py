# using face_recognition module 
import face_recognition


def detect_match(img1, img2):
    known_image = face_recognition.load_image_file(img1)
    unknown_image = face_recognition.load_image_file(img2)
    try:
        my_face_encoding = face_recognition.face_encodings(known_image)[0]
        unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]
    except IndexError:
        print("I wasn't able to locate any faces in at least one of the images. Check the image files. Aborting...")
        quit()
    results = face_recognition.compare_faces([my_face_encoding], unknown_face_encoding)
    return(results[0])

print(detect_match('r1.jpg', 'r2.jpg'))
