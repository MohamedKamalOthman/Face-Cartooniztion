import time
import glob
from rectangle_region import RectangleWindow
from haar_feature import HaarFeature
import commonfunctions as cf  # this a custom module found the commonfunctions.py
from haar_feature import *
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from progress.bar import Bar
from skimage.color import rgb2gray
import skimage.io as io
import numpy as np

"""
integral image 
input: 
    [ [5, 2, 3, 4, 1], 
      [1, 5, 4, 2, 3],
      [2, 2, 1, 3, 4],
      [3, 5, 6, 4, 5],
      [4, 1, 3, 2, 6] ]

outout of the integral_image function
    [ [0,0,  0,  0,  0,  0 ],
      [0,5,  7,  10, 14, 15],
      [0,6,  13, 20, 26, 30],
      [0,8,  17, 25, 34, 42],
      [0,11, 25, 39, 52, 65],
      [0,15, 30, 47, 62, 81] ]

"""


# Rename integeral_img to integeral_image
def integeral_image(image):

    # the length of the row
    row = len(image)

    # length of the column
    column = len(image)

    integeral_img = np.zeros((row, column))

    # iteratre over each pixel in the image
    for x in range(row):
        for y in range(column):

            # first element in the matrix so no sum
            if x == 0 and y == 0:
                integeral_img[x][y] = image[x][y]

            # first row so no need to sum all the previous rows
            elif x == 0:
                integeral_img[x][y] = integeral_img[x][y - 1] + image[x][y]

            # first column so no need to sum all the previous column
            elif y == 0:
                integeral_img[x][y] = integeral_img[x - 1][y] + image[x][y]

            # previous row + previous column - (previous column and row) + the current point
            else:
                integeral_img[x][y] = (
                    integeral_img[x - 1][y]
                    + integeral_img[x][y - 1]
                    - integeral_img[x - 1][y - 1]
                ) + image[x][y]

    return integeral_img


# Generate values from Haar features ==> White rectangles will be substracted from black ones to get each feature
# Parameters:
# - image_width, image_height: size of the original image
# - shift_amount, the amount of distance the window will be shifted
# - minimum_width, minimum_height: the starting size of the haar window
def build_features(
    image_width, image_height, shift_amount=1, minimum_width=1, minimum_height=1
):

    # [Tuple(positive(white) regions, negative(black) regions),...]
    features = []

    # scale feature window using size
    for window_width in range(minimum_width, image_width + 1):
        for window_height in range(minimum_height, image_height + 1):

            # iterarte over changing position of the feature window
            # initial x coordinate of the top left of the window
            x_top_left = 0

            while x_top_left + window_width < image_width:

                y_top_left = 0
                while y_top_left + window_height < image_height:

                    # all possible Haar regions
                    immediate = RectangleWindow(
                        x_top_left, y_top_left, window_width, window_height
                    )  # ==> |o|

                    right = RectangleWindow(
                        x_top_left + window_width,
                        y_top_left,
                        window_width,
                        window_height,
                    )  # ==> | |o|
                    right_2 = RectangleWindow(
                        x_top_left + window_width * 2,
                        y_top_left,
                        window_width,
                        window_height,
                    )  # ==> | | |o|

                    bottom = RectangleWindow(
                        x_top_left,
                        y_top_left + window_height,
                        window_width,
                        window_height,
                    )  # ==> | |/|o|
                    bottom_2 = RectangleWindow(
                        x_top_left,
                        y_top_left + window_height * 2,
                        window_width,
                        window_height,
                    )  # ==> | |/| |/|o|

                    bottom_right = RectangleWindow(
                        x_top_left + window_width,
                        y_top_left + window_height,
                        window_width,
                        window_height,
                    )  # ==> | |/| |o|

                    # Two - rectangle haar features

                    # Horizontal | 1 | -1 |  or  | white | black |
                    if x_top_left + window_width * 2 < image_width:
                        features.append(HaarFeature([immediate], [right]))

                    # Vertical |w|b|?? => not me
                    # Vertical | -1 |   or   | black |
                    #          |  1 |        | white |
                    if y_top_left + window_height * 2 < image_height:
                        features.append(HaarFeature([bottom], [immediate]))

                    # Three - rectangle haar features

                    # Horizontal |w|b|w| ?? => not me
                    # Horizontal | -1 | 1 | -1 | or | black | white | black |
                    if x_top_left + window_width * 3 < image_width:
                        features.append(HaarFeature([immediate, right_2], [right]))

                    # Vertical |w|b|w|?? => not me
                    # Vertical | -1 |   or   | black |
                    #          |  1 |        | white |
                    #          | -1 |        | black |
                    if y_top_left + window_height * 3 < image_height:
                        features.append(HaarFeature([immediate, bottom_2], [bottom]))

                    # Four - rectangle haar features
                    if (
                        x_top_left + window_width * 2 < image_width
                        and y_top_left + window_height * 2 < image_height
                    ):
                        features.append(
                            HaarFeature([immediate, bottom_right], [bottom, right])
                        )

                    # shift window position
                    y_top_left += shift_amount

                # shift window position
                x_top_left += shift_amount

    return features


# Build features of all the training data (integral images)


def apply_features(features_integral_images, features):

    # X : features
    features_values = np.zeros(
        (len(features), len(features_integral_images)), dtype=np.int32
    )

    # each row will contain a list of features , for example:
    # feature[0][i] is the first feature of the image of index i in the data set..
    # y: will be kept as it is => f0=([...], y); f1=([...], y),...
    # to display progress
    bar = Bar(
        "Processing features",
        max=len(features),
        suffix="%(percent)d%% - %(elapsed_td)s - %(eta_td)s",
    )

    for i, feature in bar.iter(enumerate(features)):
        # Compute the value of feature 'i' for each image in the training set
        # it will be Input of the classifier_i
        features_values[i] = list(
            map(
                lambda integral_img: feature.get_haar_feature_value(integral_img),
                features_integral_images,
            )
        )
    bar.finish()

    # [[ftr0 of img0, ftr0 of img1, ...][ftr1 of img0, ftr1 of img1, ...],....]
    return features_values
