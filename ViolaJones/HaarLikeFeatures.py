import enum
import numpy as np

# Understood
#start

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


# Understood
class HaarlikeType(enum.Enum):
    two_horizontal = 0
    two_vertical = 1
    three_horizontal = 2
    three_vertical = 3
    four_diagonal = 4
    types_count = 5


# This class has the functions to get the haar like features from the integral image
class HaarlikeFeatureDetector:

    haar_window = [  # width ==> cols , hight ==> rows
        # for two_horizontal type
        (2, 1),
        # for two_vertical type
        (1, 2),
        # for three_horizontal type
        (3, 1),
        # for three_vertical type
        (1, 3),
        # for four_diagonal type
        (2, 2),
    ]

    # Understood
    # constructor takes all the required paramenters required to get the haar like features
    def __init__(self, image):

        # setting the width of the Haar feature window
        self.width = image.shape[1]

        # setting the height of the feature
        self.height = image.shape[0]

        # setting the image that we want to get the haar like features for
        self.image = np.array(image)

        self.integral_img = integeral_image(image)

    # Understood
    @staticmethod
    def get_window_sum(
        integral_image, starting_row, starting_column, window_width, window_height
    ):

        """
        get sum of pixels of specific window (rectangle) inside an image
        we need to use the integral image as it simplify the
        calculation of this sum, using the integral image,
        instead of sum all the pixels inside the window of
        the orginal image we just can take to 4 corners of
        the window inside the integral images and the sum
        will be = topLeft + bottomRight - topRight - bottomLeft
        y: staring row
        x: starting col
        w: width of the window
        h: height of the window
        """

        starting_row = int(starting_row)
        starting_column = int(starting_column)
        window_width = int(window_width)
        window_height = int(window_height)

        if (
            starting_row < 0
            or starting_column < 0
            or window_height < 0
            or window_width < 0
        ):
            return 0

        if starting_column == 0 or starting_row == 0:
            top_left = 0
        else:
            top_left = integral_image[starting_row - 1, starting_column - 1]

        if starting_column == 0:
            top_right = 0
        else:
            top_right = integral_image[
                starting_row - 1 + window_width, starting_column - 1
            ]

        if starting_row == 0:
            bottom_left = 0
        else:
            bottom_left = integral_image[
                starting_row - 1, starting_column - 1 + window_height
            ]

        bottom_right = integral_image[
            starting_row - 1 + window_width, starting_column - 1 + window_height
        ]

        return top_left + bottom_right - top_right - bottom_left

    # Understood
    @staticmethod
    def determine_features(width, height):
        """
        Determine the count of features for all types of the haarWindows
        Parameters:
        width : int
            The width of the window.
        height : int
            The height of the window.
        Returns:
        features_count : int
            The features count of this window size
        descriptions : list of shape = [features_cnt, [haartype, x, y, w, h]]
            The descriptions of each feature.
        """

        features_count = 0
        # get the total features count for all types of haarWindows
        # there are 5 types of haarWindows, each type has multiple positions and multiple sizes

        for haar_type in range(HaarlikeType.types_count.value):

            # haarFeatureWindow width & height
            (
                haar_feature_window_size_x,
                haar_feature_window_size_y,
            ) = __class__.haar_window[haar_type]

            # for each position
            for column in range(0, width - haar_feature_window_size_x + 1):
                for row in range(0, height - haar_feature_window_size_y + 1):
                    # x: col
                    # y: row
                    # for each size (starting from size= haar_feature_window_sizeX, haar_feature_window_sizeY and
                    # increment the width and hight by haar_feature_window_sizeX, haar_feature_window_sizeY until
                    # we reach the end of the window width & height)
                    # the reason for the devision is to know how many
                    # HFW will fit in the remaining original window
                    features_count += int(
                        (width - column) / haar_feature_window_size_x
                    ) * int((height - row) / haar_feature_window_size_y)

        descriptions = np.zeros((features_count, 5))
        index = 0

        for haar_type in range(HaarlikeType.types_count.value):
            (
                haar_feature_window_size_x,
                haar_feature_window_size_y,
            ) = __class__.haar_window[haar_type]

            # for each size
            for w in range(
                haar_feature_window_size_x, width + 1, haar_feature_window_size_x
            ):
                for h in range(
                    haar_feature_window_size_y, height + 1, haar_feature_window_size_y
                ):

                    # for each position
                    for y in range(0, height - h + 1):
                        for x in range(0, width - w + 1):

                            # x: position(col number)
                            # y: position(row, number)
                            # w: width(#cols)
                            # h: height(#rows)
                            descriptions[index] = [haar_type, x, y, w, h]
                            index += 1

        return features_count, descriptions

    # Understood
    @staticmethod
    def extract_features(integral_image, features_descriptions):

        # extract the features from an image based on the features count and descriptions
        # Returns: list of all haar like features in the whole image

        rows, columns = integral_image.shape

        ## make array of features
        features = np.zeros(len(features_descriptions))
        index = 0
        for desc in features_descriptions:
            features[index] = HaarlikeFeatureDetector.getSingleFeature(
                integral_image,
                HaarlikeType(desc[0]),
                desc[1],
                desc[2],
                desc[3],
                desc[4],
            )
            index += 1

        return features

    # Understood
    @staticmethod
    def get_single_feature(integral_image, haar_type, x, y, w, h):
        """
        get the featuree in a specific window position
        x= starting col
        y= starting row
        notice that x,y are flipped as the haarTypeWindows
        dimensions is the opposite of the coordinate system
        w= width of the region
        h= height of the region
        """

        white_region_sum = 0
        black_region_sum = 0

        if haar_type == HaarlikeType.two_horizontal:
            white_region_sum = HaarlikeFeatureDetector.get_window_sum(
                integral_image, x, y, w / 2, h
            )  # negative regions
            black_region_sum = HaarlikeFeatureDetector.get_window_sum(
                integral_image, x + w / 2, y, w / 2, h
            )  # positive regions

        elif haar_type == HaarlikeType.two_vertical:
            white_region_sum = HaarlikeFeatureDetector.get_window_sum(
                integral_image, x, y, w, h / 2
            )  # negative regions
            black_region_sum = HaarlikeFeatureDetector.get_window_sum(
                integral_image, x, y + h / 2, w, h / 2
            )  # positive regions

        elif haar_type == HaarlikeType.three_horizontal:
            white_region_sum = HaarlikeFeatureDetector.get_window_sum(
                integral_image, x, y, w / 3, h
            ) + HaarlikeFeatureDetector.get_window_sum(
                integral_image, x + 2 * w / 3, y, w / 3, h
            )  # negative regions
            black_region_sum = HaarlikeFeatureDetector.get_window_sum(
                integral_image, x + w / 3, y, w / 3, h
            )  # positive regions

        elif haar_type == HaarlikeType.three_vertical:
            white_region_sum = HaarlikeFeatureDetector.get_window_sum(
                integral_image, x, y, w, h / 3
            ) + HaarlikeFeatureDetector.get_window_sum(
                integral_image, x, y + 2 * h / 3, w, h / 3
            )  # negative regions
            black_region_sum = HaarlikeFeatureDetector.get_window_sum(
                integral_image, x, y + h / 3, w, h / 3
            )  # positive regions

        elif haar_type == HaarlikeType.four_diagonal:
            white_region_sum = HaarlikeFeatureDetector.get_window_sum(
                integral_image, x, y, w / 2, h / 2
            ) + HaarlikeFeatureDetector.get_window_sum(
                integral_image, x + w / 2, y + h / 2, w / 2, h / 2
            )  # negative regions
            black_region_sum = HaarlikeFeatureDetector.get_window_sum(
                integral_image, x + w / 2, y, w / 2, h / 2
            ) + HaarlikeFeatureDetector.get_window_sum(
                integral_image, x, y + h / 2, w / 2, h / 2
            )  # positive regions

        return white_region_sum - black_region_sum

    # Understood
    @staticmethod
    def apply_features(features, training_data):
        """
        inputs:
        features: the o/p of determine_features[1]
        training_data: a list of tuples(IntegralImg, classification)

        Maps features onto the training dataset
        X=
        [
        [img1 ftr1, img2 ftr1, img3 ftr1],
        [img1 ftr2, img2 ftr2, img3 ftr2],
        [img1 ftr3, img2 ftr3, img3 ftr3],
        [img1 ftr4, img2 ftr4, img3 ftr4],
        [img1 ftr5, img2 ftr5, img3 ftr5],
        .
        .
        .
        ]

        """
        X = np.zeros((len(features), len(training_data)))

        # list of classifications of the image
        y = np.array(list(map(lambda data: data[1], training_data)))

        i = 0
        for feature_description in features:
            feature_extractor = lambda intImg: HaarlikeFeatureDetector.getSingleFeature(
                intImg,
                HaarlikeType(feature_description[0]),
                feature_description[1],
                feature_description[2],
                feature_description[3],
                feature_description[4],
            )
            X[i] = list(map(lambda data: feature_extractor(data[0]), training_data))
            i += 1
        return X, y
