import cv2
import dlib

# import imutils
import numpy as np
import scipy.fft as fft
from progress.bar import Bar


class eyelocator:
    def __init__(self, load=False, path="./") -> None:
        if load:
            self.load(path)
        else:
            self.left_eye_asef = fft.fft2(np.zeros([128, 128]))
            self.right_eye_asef = fft.fft2(np.zeros([128, 128]))

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

    def load(self, path="./"):
        self.left_eye_asef = np.load(path + "/left_eye_asef.npy")
        self.right_eye_asef = np.load(path + "/right_eye_asef.npy")

    def train(self, images):
        taken, skipped = 0, 0
        bar = Bar(
            'Training Images',
            max=len(images),
            suffix='%(percent)d%% - %(elapsed_td)s - %(eta_td)s',
        )
        for img in bar.iter(images):
            try:
                img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(img_grey, 1.3, 5)
                # for _ in range(8):
                # rx, ry = np.random.randint(-4, 4, (2))
                (y, x, h, w) = faces[0]
                resize = cv2.resize(img_grey[x : x + w, y : y + h], (128, 128))
                # angle = ((np.random.rand() * np.pi) / 8) - (np.pi / 16)
                # resize = imutils.rotate(resize, angle)
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
                left_gauss = self.gauss(x1, y1)
                right_gauss = self.gauss(x2, y2)

                left_gauss_f = fft.fft2(left_gauss)
                right_gauss_f = fft.fft2(right_gauss)
                img_f = fft.fft2(resize)
                # if math.isinf(np.abs(f1)) or math.isinf(np.abs(f2)):
                #     skipped += 1
                #     continue
                f1 = np.divide(
                    left_gauss_f, img_f, out=np.zeros_like(img_f), where=img_f != 0
                )
                f2 = np.divide(
                    right_gauss_f, img_f, out=np.zeros_like(img_f), where=img_f != 0
                )
                self.left_eye_asef += f1
                self.right_eye_asef += f2
                taken += 1
            except:
                skipped += 1
                continue
        self.left_eye_asef / taken
        self.right_eye_asef / taken
        print(f'Finished training, taken = {taken}, skipped = {skipped}')

    def gauss(self, x, y, sigma=4.0):
        return np.fromfunction(
            lambda i, j: np.exp(
                -1.0 * (np.square(i - x) + np.square(j - y)) / np.square(sigma)
            ),
            (128, 128),
        )

    def findeye(self, image):
        # img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(image, [128, 128])
        img_f = fft.fft2(img)
        left_eye_img = fft.ifft2(img_f * self.left_eye_asef)
        right_eye_img = fft.ifft2(img_f * self.right_eye_asef)
        x1, y1 = np.unravel_index(left_eye_img.argmax(), left_eye_img.shape)
        x2, y2 = np.unravel_index(right_eye_img.argmax(), right_eye_img.shape)
        cv2.ellipse(img, (x1, y1), (9, 4), 0, 0, 360, color=255, thickness=1)
        cv2.ellipse(img, (x2, y2), (9, 4), 0, 0, 360, color=255, thickness=1)
        return img