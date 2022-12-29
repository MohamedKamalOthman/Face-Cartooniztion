import cv2
import dlib

# import imutils
import numpy as np
import scipy.fft as fft
from progress.bar import Bar
from ffextractor import FeaturesLocator


def distance(x1, y1, x2, y2):
    # distance smaller thn 10 pixels
    return ((x1-x2)**2 + (y1-y2)**2)**1/2 < 30


if __name__ == "__main__":
    locator = FeaturesLocator(load=True, path="./results")
    data_size = 500
    left_eye_correct = 0
    right_eye_correct = 0
    nose_correct = 0
    mouth_correct = 0
    left_eye_false = 0
    right_eye_false = 0
    nose_false = 0
    mouth_false = 0
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(
        "./shape_predictor_68_face_landmarks.dat")
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    print("\nStarted testing...")
    for i in range(data_size):
        try:
            img = cv2.imread(f'./face-test-set/{i+100}.jpg')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(img, 1.3, 5)
            (y, x, h, w) = faces[0]
            img = cv2.resize(img[x: x + w, y: y + h], (128, 128))
            features = locator.findfeatures(img)
            detect = detector(img, 1)
            shape = predictor(img, detect[0])
            # left eye location
            x1, y1 = (shape.part(37).x + shape.part(40).x) // 2, (
                shape.part(37).y + shape.part(40).y
            ) // 2
            # right eye location
            x2, y2 = (shape.part(43).x + shape.part(46).x) // 2, (
                shape.part(43).y + shape.part(46).y
            ) // 2
            # nose tip
            x3, y3 = shape.part(30).x, shape.part(30).y
            # mouth center
            x4, y4 = (shape.part(54).x + shape.part(48).x) // 2, (
                shape.part(54).y + shape.part(48).y
            ) // 2
            if distance(features['left_eye'][0], features['left_eye'][1], x1, y1):
                left_eye_correct += 1
            else:
                left_eye_false += 1
            if distance(features['right_eye'][0], features['right_eye'][1], x2, y2):
                right_eye_correct += 1
            else:
                right_eye_false += 1
            if distance(features['nose'][0], features['nose'][1], x3, y3):
                nose_correct += 1
            else:
                nose_false += 1
            if distance(features['mouth'][0], features['mouth'][1], x4, y4):
                mouth_correct += 1
            else:
                mouth_false += 1
        except:
            continue
    print(
        f"Left Eye accuracy: {left_eye_correct*100.0/(left_eye_correct+left_eye_false)}%")
    print(f"correct: {left_eye_correct}, false: {left_eye_false}")
    print(
        f"Right Eye accuracy: {right_eye_correct*100.0/(right_eye_correct+right_eye_false)}%")
    print(f"correct: {right_eye_correct}, false: {right_eye_false}")
    print(
        f"Nose accuracy: {nose_correct*100.0/(nose_correct+nose_false)}%")
    print(
        f"correct: {nose_correct}, false: {nose_false}")
    print(
        f"Mouth accuracy: {mouth_correct*100.0/(mouth_correct+mouth_false)}%")

    print(
        f"correct: {mouth_correct}, false: {mouth_false}")
