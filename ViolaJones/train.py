from ViolaJones import ViolaJones
import pickle
import numpy as np
import time
import matplotlib.pyplot as plt
from util import *
import skimage.io as io
from skimage.color import rgb2gray, rgb2hsv
import cv2
import commonfunctions as cf
from haar_utils import *
from haar_feature import *


def train(num_pos, num_neg):
    with open("Datasets/faces_cropped.pkl", "rb") as f:
        data = pickle.load(f)
    with open("Datasets/non-faces.pkl", "rb") as f:
        data += pickle.load(f)
    start_pos = 0
    end_pos = start_pos + num_pos
    start_neg = 5000
    end_neg = start_neg + num_neg
    train_data = data[start_pos:end_pos] + data[start_neg:end_neg]
    images = []
    labels = []
    for tup in train_data:
        images.append(tup[0])
        labels.append(tup[1])

    featurespath = (
        "_"
        + str(start_pos)
        + "_"
        + str(end_pos)
        + "_"
        + str(start_neg)
        + "_"
        + str(end_neg)
    )
    clf = ViolaJones(
        layers=[40],
        featurespath="_"
        + str(start_pos)
        + "_"
        + str(end_pos)
        + "_"
        + str(start_neg)
        + "_"
        + str(end_neg),
    )

    clf.train(images, np.array(labels))  # X_f (optional, to speed-up training)
    print("Training finished!")

    # Save weights
    print("\nSaving weights...")
    clf.save("cvj_weights_data_set_2" + featurespath + str(len(clf.layers)))
    print("Weights saved!")


if __name__ == "__main__":
    train(100, 50)
