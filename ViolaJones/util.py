import pickle
import numpy as np
import cv2
from haar_utils import *
import imutils

#start

## to draw boxes in the original image
def render_boxes(image, regions) -> Image.Image:
    canvas = np.copy(image)
    for col, row, width, height in regions:

        cv2.rectangle(
            canvas, (row - width, col - height), (row + width, col + height), 255, 2
        )
    return canvas


def non_maximum_supression_multiscale(regions, threshold=0.5):
    boxes = np.array(regions)

    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats ==> this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indicies
    pick = []

    # boxes = [[x, y, w, h]]
    # grab the coordinates of the bounding boxes
    start_x = boxes[:, 0] - boxes[:, 2] - 1
    start_y = boxes[:, 1] - boxes[:, 3] - 1

    end_x = boxes[:, 0] + boxes[:, 2]
    end_y = boxes[:, 1] + boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding boxes by the bottom-right y-coordinate of the bounding box
    area = (end_x - start_x + 1) * (end_y - start_y + 1)

    indicies = np.argsort(area)

    # keep looping while some indicies still remain in the indicies list
    while len(indicies) > 0:

        # grab the last index in the indicies list and add the index value to the list of picked indicies
        last = len(indicies) - 1
        i = indicies[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of the bounding box and the smallest (x, y) coordinates for the end of the bounding box
        x_maximum = np.maximum(start_x[i], start_x[indicies[:last]])
        y_maximum = np.maximum(start_y[i], start_y[indicies[:last]])

        x_minimum = np.minimum(end_x[i], end_x[indicies[:last]])
        y_minimum = np.minimum(end_y[i], end_y[indicies[:last]])

        # compute the width and height of the bounding box
        width = np.maximum(0, x_minimum - x_maximum + 1)
        height = np.maximum(0, y_minimum - y_maximum + 1)

        # compute the ratio of overlap
        overlap = (width * height) / area[indicies[:last]]

        # delete all indexes from the index list that have
        indicies = np.delete(
            indicies, np.concatenate(([last], np.where(overlap > threshold)[0]))
        )

    # return only the bounding boxes that were picked using the integer data type
    return boxes[pick].astype("int")


def evaluate(clf, X, y):
    metrics = {}
    true_positive, true_negative = 0, 0  # Correct
    false_positive, false_negative = 0, 0  # Incorrect
    for i in range(len(y)):
        prediction = clf.classify(X[i])
        if prediction == y[i]:  # Correct
            if prediction == 1:  # Face
                true_positive += 1
            else:  # No-face
                true_negative += 1
        else:  # Incorrect

            if prediction == 1:  # Face
                false_positive += 1
            else:  # No-face
                false_negative += 1

    metrics["true_positive"] = true_positive
    metrics["true_negative"] = true_negative
    metrics["false_positive"] = false_positive
    metrics["false_negative"] = false_negative

    metrics["accuracy"] = (true_positive + true_negative) / (
        true_positive + false_negative + true_negative + false_positive
    )
    metrics["precision"] = true_positive / (true_positive + false_positive)
    metrics["recall"] = true_positive / (true_positive + false_negative)
    metrics["specifity"] = true_negative / (true_negative + false_positive)
    metrics["f1"] = (2.0 * metrics["precision"] * metrics["recall"]) / (
        metrics["precision"] + metrics["recall"]
    )

    return metrics


def test(clf, name="test"):
    # Load test set
    print("\nLoading {}...".format(name))
    with open("Datasets/test_faces_cropped.pkl", "rb") as f:
        test = pickle.load(f)
    with open("Datasets/non-faces.pkl", "rb") as f:
        test2 = pickle.load(f)
    # print(len(test))
    # print(len(test2))
    test = test + test2[3000:]
    #     test = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
    #     test = [(cv2.resize(test , (19,19)) , 1)]
    #     io.imshow(test[0][0])
    np.random.shuffle(test)
    images_test = []
    labels_test = []
    for tup in test:
        images_test.append(tup[0])
        labels_test.append(tup[1])
    # Evaluate
    print("\nEvaluating...")
    metrics = evaluate(clf, images_test, np.array(labels_test))

    print("Metrics: [{}]".format(name))
    counter = 0
    for k, v in metrics.items():
        counter += 1
        if counter <= 4:
            print("\t- {}: {:,}".format(k, v))
        else:
            print("\t- {}: {:.3f}".format(k, v))


def gamma(values: np.ndarray, coeff: float = 2.2) -> np.ndarray:
    return values ** (1.0 / coeff)


def get_faces_in_multiple_scales(original_image, clf, half_window, shift=1):
    all_faces = []
    scale_w_factor = original_image.shape[0] / 19
    scale_h_factor = original_image.shape[1] / 19
    width = round(19 * scale_w_factor)
    height = round(19 * scale_h_factor)
    while width > 19 and height > 19:
        # print(scale_factor, width)
        new_image = imutils.resize(original_image, width=width)
        face_positions = get_best_faces_in_scale(
            half_window,
            new_image.shape[0],
            new_image.shape[1],
            new_image,
            clf,
            original_image.shape[0] / new_image.shape[0],
            original_image.shape[1] / new_image.shape[1],
            int(shift + shift * (max(scale_h_factor, scale_w_factor) // 10)),
        )
        # print(f"Found {len(face_positions)} faces.")
        all_faces += face_positions
        scale_w_factor /= 1.1
        width = round(19 * scale_w_factor)
        scale_h_factor /= 1.1
        height = round(19 * scale_h_factor)
    return all_faces


def get_best_faces_in_scale(
    half_window,
    rows,
    cols,
    final_image,
    clf_test,
    scale_w_factor,
    scale_h_factor,
    shift=1,
):
    # print(final_image.shape, scale_w_factor, scale_h_factor, half_window, shift)
    face_positions = []
    for row in range(half_window + 1, rows - half_window, shift):
        for col in range(half_window + 1, cols - half_window, shift):
            # walking through the image with our window size
            if (
                row + half_window <= final_image.shape[0]
                and col + half_window + 1 <= final_image.shape[1]
                and row - half_window - 1 >= 0
                and col - half_window - 1 >= 0
            ):
                window = final_image[
                    row - half_window - 1 : row + half_window + 1,
                    col - half_window - 1 : col + half_window + 1,
                ]
            else:
                continue

            probably_face = clf_test.classify(window)
            if probably_face < 0.5:
                continue
            # print((row * scale_factor, col * scale_factor, 19 * scale_factor))
            face_positions.append(
                (
                    round(row * scale_w_factor),
                    round(col * scale_h_factor),
                    round(half_window * scale_w_factor),
                    round(half_window * scale_h_factor),
                )
            )
    return face_positions
