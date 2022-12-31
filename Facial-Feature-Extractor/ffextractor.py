import cv2
import dlib

# import imutils
import numpy as np
import scipy.fft as fft
from progress.bar import Bar


class FeaturesLocator:
    """
    Facial Feature Extractor\n
    Uses ASEF Locator
    """

    def __init__(self, load=False, path="./") -> None:
        if load:
            self.load(path)
        else:
            self.left_eye_asef = fft.fft2(np.zeros([128, 128]))
            self.right_eye_asef = fft.fft2(np.zeros([128, 128]))
            self.nose_asef = fft.fft2(np.zeros([128, 128]))
            self.mouth_asef = fft.fft2(np.zeros([128, 128]))

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

    def load(self, path="./"):
        """Load trained data from filesystem"""
        self.left_eye_asef = np.load(path + "/left_eye_asef.npy")
        self.right_eye_asef = np.load(path + "/right_eye_asef.npy")
        self.nose_asef = np.load(path + "/nose_asef.npy")
        self.mouth_asef = np.load(path + "/mouth_asef.npy")

    def train(self, images):
        "Train with iterable images"
        taken, skipped = 0, 0
        bar = Bar(
            "Training Images",
            max=len(images),
            suffix="%(percent)d%% - %(elapsed_td)s - %(eta_td)s",
        )
        for img in bar.iter(images):
            try:
                img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(img_grey, 1.3, 5)
                (y, x, h, w) = faces[0]
                resize = cv2.resize(img_grey[x : x + w, y : y + h], (128, 128))
                detect = self.detector(resize, 1)
                shape = self.predictor(resize, detect[0])
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
                # Applying filters
                left_eye_gauss = self.gauss(x1, y1)
                right_eye_gauss = self.gauss(x2, y2)
                nose_gauss = self.gauss(x3, y3)
                mouth_gauss = self.gauss(x4, y4)
                # fourier transform
                left_eye_gauss_f = fft.fft2(left_eye_gauss)
                right_eye_gauss_f = fft.fft2(right_eye_gauss)
                nose_gauss_f = fft.fft2(nose_gauss)
                mouth_gauss_f = fft.fft2(mouth_gauss)
                img_f = fft.fft2(resize)
                f1 = np.divide(
                    left_eye_gauss_f, img_f, out=np.zeros_like(img_f), where=img_f != 0
                )
                f2 = np.divide(
                    right_eye_gauss_f, img_f, out=np.zeros_like(img_f), where=img_f != 0
                )
                f3 = np.divide(
                    nose_gauss_f, img_f, out=np.zeros_like(img_f), where=img_f != 0
                )
                f4 = np.divide(
                    mouth_gauss_f, img_f, out=np.zeros_like(img_f), where=img_f != 0
                )
                self.left_eye_asef += f1
                self.right_eye_asef += f2
                self.nose_asef += f3
                self.mouth_asef += f4
                taken += 1
            except:
                skipped += 1
                continue
        # Normalize fft
        self.left_eye_asef /= taken
        self.right_eye_asef /= taken
        self.nose_asef /= taken
        self.mouth_asef /= taken
        print(f"Finished training, taken = {taken}, skipped = {skipped}")

    def gauss(self, x, y, sigma=4.0):
        return np.fromfunction(
            lambda i, j: np.exp(
                -1.0 * (np.square(i - x) + np.square(j - y)) / np.square(sigma)
            ),
            (128, 128),
        )

    def findfeatures(self, image):
        # img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        scale = [image.shape[0] / 128, image.shape[1] / 128]
        img = cv2.resize(image, [128, 128])
        img_f = fft.fft2(img)
        # detection
        left_eye_img = fft.ifft2(img_f * self.left_eye_asef)
        right_eye_img = fft.ifft2(img_f * self.right_eye_asef)
        nose_img = fft.ifft2(img_f * self.nose_asef)
        mouth_img = fft.ifft2(img_f * self.mouth_asef)
        left_eye = np.multiply(
            np.unravel_index(left_eye_img.argmax(), left_eye_img.shape), scale
        ).astype(np.uint32)
        right_eye = np.multiply(
            np.unravel_index(right_eye_img.argmax(), right_eye_img.shape), scale
        ).astype(np.uint32)
        nose = np.multiply(
            np.unravel_index(nose_img.argmax(), left_eye_img.shape), scale
        ).astype(np.uint32)
        mouth = np.multiply(
            np.unravel_index(mouth_img.argmax(), mouth_img.shape), scale
        ).astype(np.uint32)
        return {
            "left_eye": left_eye,
            "right_eye": right_eye,
            "nose": nose,
            "mouth": mouth,
        }
