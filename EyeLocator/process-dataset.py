import os

import cv2
import dlib

if __name__ == "__main__":
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

    dataset_size = 1491
    for i in range(dataset_size):
        img = cv2.imread(f'./face-cleaning-set/ ({i+1}).jpg')
        img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(img_grey, 1.3, 5)
        try:
            (y, x, h, w) = faces[0]
            resize = cv2.resize(img_grey[x : x + w, y : y + h], (128, 128))
            detect = detector(resize, 1)
            shape = predictor(resize, detect[0])
            # left eye location
            x1, y1 = (shape.part(37).x + shape.part(40).x) // 2, (
                shape.part(37).y + shape.part(40).y
            ) // 2
            # right eye location
            x2, y2 = (shape.part(43).x + shape.part(46).x) // 2, (
                shape.part(43).y + shape.part(46).y
            ) // 2
            cv2.circle(resize, (x1, y1), 2, 255, -1)
            cv2.circle(resize, (x2, y2), 2, 255, -1)
            cv2.imwrite(f'./face-cleaning-set/processed/ ({i+1}).jpg', resize)
        except:
            os.remove(f"./face-cleaning-set/({i+1}).jpg")
