import cv2
from ffextractor import FeaturesLocator
import numpy as np
import imutils
from collections import defaultdict

locator = FeaturesLocator(load=True, path="./results")
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def check_left_eye(keypoint, w, h):
    x, y = keypoint
    return 0 < x and x < w // 2 and 2 * h // 7 < y and y < 4 * h // 7


def check_right_eye(keypoint, w, h):
    x, y = keypoint
    return w // 2 < x and x < w and 2 * h // 7 < y and y < 4 * h // 7


def check_nose(keypoint, w, h):
    x, y = keypoint
    return w // 4 < x and x < 3 * w // 4 and 3 * h // 7 < y and y < 6 * h // 7


def check_mouth(keypoint, w, h):
    x, y = keypoint
    return w // 4 < x and x < 3 * w // 4 and h // 2 < y and y < 9 * h // 10


def angle_transform(pair, angle):
    x = pair[0] * np.cos(angle * np.pi / 180) - pair[1] * np.sin(angle * np.pi / 180)
    y = pair[0] * np.sin(angle * np.pi / 180) + pair[1] * np.cos(angle * np.pi / 180)
    pair[0] = x
    pair[1] = y
    return pair


def enhance_keypoints(face, w, h):
    all_keypoints = []
    rots = [352, 354, 356, 358, 0, 2, 4, 6, 8]
    for rot in rots:
        face_rotated = imutils.rotate(face, rot)
        keypoints = locator.findfeatures(face_rotated)
        for k, v in keypoints.items():
            keypoints[k] = angle_transform(v, -rot)

        all_keypoints.append(keypoints)

    keypoints = {"left_eye": None, "right_eye": None, "nose": None, "mouth": None}
    arr = [x["left_eye"] for x in all_keypoints if check_left_eye(x["left_eye"], w, h)]
    if len(arr) > 0:
        keypoints["left_eye"] = np.median(
            arr,
            axis=0,
        ).astype(np.uint32)
    arr = [
        x["right_eye"] for x in all_keypoints if check_right_eye(x["right_eye"], w, h)
    ]
    if len(arr) > 0:
        keypoints["right_eye"] = np.median(
            arr,
            axis=0,
        ).astype(np.uint32)
    arr = [x["nose"] for x in all_keypoints if check_nose(x["nose"], w, h)]
    if len(arr) > 0:
        keypoints["nose"] = np.median(arr, axis=0).astype(np.uint32)
    arr = [x["mouth"] for x in all_keypoints if check_mouth(x["mouth"], w, h)]
    if len(arr) > 0:
        keypoints["mouth"] = np.median(arr, axis=0).astype(np.uint32)
    return keypoints


def add_transparent_image(background, foreground, x_offset=None, y_offset=None):
    bg_h, bg_w, bg_channels = background.shape
    fg_h, fg_w, fg_channels = foreground.shape

    assert (
        bg_channels == 3
    ), f"background image should have exactly 3 channels (RGB). found:{bg_channels}"
    assert (
        fg_channels == 4
    ), f"foreground image should have exactly 4 channels (RGBA). found:{fg_channels}"

    x_offset = x_offset - foreground.shape[0] // 2
    y_offset = y_offset - foreground.shape[1] // 2
    # center by default
    if x_offset is None:
        x_offset = (bg_w - fg_w) // 2
    if y_offset is None:
        y_offset = (bg_h - fg_h) // 2

    w = min(fg_w, bg_w, fg_w + x_offset, bg_w - x_offset)
    h = min(fg_h, bg_h, fg_h + y_offset, bg_h - y_offset)

    if w < 1 or h < 1:
        return

    # clip foreground and background images to the overlapping regions
    bg_x = max(0, x_offset)
    bg_y = max(0, y_offset)
    fg_x = max(0, x_offset * -1)
    fg_y = max(0, y_offset * -1)
    foreground = foreground[fg_y : fg_y + h, fg_x : fg_x + w]
    background_subsection = background[bg_y : bg_y + h, bg_x : bg_x + w]

    # separate alpha and color channels from the foreground image
    foreground_colors = foreground[:, :, :3]
    alpha_channel = foreground[:, :, 3] / 255  # 0-255 => 0.0-1.0

    # construct an alpha_mask that matches the image shape
    alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))

    # combine the background with the overlay image weighted by alpha
    composite = (
        background_subsection * (1 - alpha_mask) + foreground_colors * alpha_mask
    )

    # overwrite the section of the background image that has been updated
    background[bg_y : bg_y + h, bg_x : bg_x + w] = composite


def render_filter_0(img):
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img_grey, 1.3, 5)
    if len(faces) == 0:
        return img

    (y, x, h, w) = faces[0]
    face = img_grey[x : x + w, y : y + h]

    keypoints = enhance_keypoints(face, w, h)
    if (
        keypoints["left_eye"] is None
        or keypoints["right_eye"] is None
        or keypoints["nose"] is None
        or keypoints["mouth"] is None
    ):
        return img
    xwidth = int(9 * img.shape[0] / 350)
    ywidth = int(4 * img.shape[1] / 350)
    thickness = (img.shape[0] * img.shape[1]) // 200000
    cv2.ellipse(
        img,
        np.add(keypoints["left_eye"], [y, x]),
        (xwidth, ywidth),
        0,
        0,
        360,
        color=255,
        thickness=thickness,
    )
    cv2.ellipse(
        img,
        np.add(keypoints["right_eye"], [y, x]),
        (xwidth, ywidth),
        0,
        0,
        360,
        color=255,
        thickness=thickness,
    )
    cv2.ellipse(
        img,
        np.add(keypoints["nose"], [y, x]),
        (xwidth, ywidth),
        0,
        0,
        360,
        color=255,
        thickness=thickness,
    )
    cv2.ellipse(
        img,
        np.add(keypoints["mouth"], [y, x]),
        (xwidth, ywidth),
        0,
        0,
        360,
        color=255,
        thickness=thickness,
    )
    cv2.rectangle(img, (y, x), (y + h, x + w), (0, 0, 255), 2)
    return img


def render_filter_1(img):
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img_grey, 1.3, 5)
    if len(faces) == 0:
        return img
    glasses = cv2.imread("./filters/glasses.png", cv2.IMREAD_UNCHANGED)
    (y, x, h, w) = faces[0]
    glasses = cv2.resize(glasses, (np.array([w, h]) * 0.8).astype(int))
    face = img_grey[x : x + w, y : y + h]
    keypoints = enhance_keypoints(face, w, h)
    if keypoints["left_eye"] is None or keypoints["right_eye"] is None:
        return img
    center = (keypoints["left_eye"] + keypoints["right_eye"]) // 2
    add_transparent_image(img, glasses, center[0] + y, center[1] + x)
    return img


def render_filter_2(img):
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img_grey, 1.3, 5)
    if len(faces) == 0:
        return img
    nose = cv2.imread("./filters/clown_nose.png", cv2.IMREAD_UNCHANGED)
    wig = cv2.imread("./filters/clown_wig.png", cv2.IMREAD_UNCHANGED)
    (y, x, h, w) = faces[0]
    nose = cv2.resize(nose, (np.array([w, h]) * 0.30).astype(int))
    wig = cv2.resize(wig, (np.array([w, h]) * 1.25).astype(int))
    face = img_grey[x : x + w, y : y + h]
    keypoints = enhance_keypoints(face, w, h)
    if (
        keypoints["left_eye"] is None
        or keypoints["right_eye"] is None
        or keypoints["nose"] is None
    ):
        return img
    center_nose = keypoints["nose"]
    center_wig = (keypoints["left_eye"] + keypoints["right_eye"]) // 2
    add_transparent_image(img, nose, center_nose[0] + y, center_nose[1] + x)
    add_transparent_image(img, wig, center_wig[0] + y, center_wig[1] + x - h // 3)
    return img


def render_filter_3(img):
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img_grey, 1.3, 5)
    if len(faces) == 0:
        return img
    mask = cv2.imread("./filters/mask.png", cv2.IMREAD_UNCHANGED)
    (y, x, h, w) = faces[0]
    mask = cv2.resize(mask, (np.array([w, h]) * 0.90).astype(int))
    face = img_grey[x : x + w, y : y + h]
    keypoints = enhance_keypoints(face, w, h)
    if keypoints["mouth"] is None:
        return img
    center = keypoints["mouth"]
    add_transparent_image(img, mask, center[0] + y, center[1] + x)
    return img


def K_means(hist, No_of_groups=10):
    # Initialize Cetroids by dividing the range of histogram to equal size Clusters
    step = int(len(hist) / No_of_groups)
    dum = [i for i in range(0, len(hist) + 1, step)]
    old_centroids = np.array(dum)
    new_centroids = np.zeros_like(old_centroids)

    while True:
        clusters = defaultdict(list)
        # Construct dictionary contains the Clusters by subtract each histogram value from all centroids and get index of
        # minimum result to know each histogram value closest to each cluster and append it to the list of this cluster
        for i in range(len(hist)):
            if hist[i] == 0:
                continue
            dis = np.abs(old_centroids - i)
            index = np.argmin(dis)
            clusters[index].append(i)

        # Calculate New Centroids by making a weighted average for each cluster
        for i, ind in clusters.items():
            if np.sum(hist[ind]) == 0:
                continue
            new_centroids[i] = int(np.sum(ind * hist[ind]) / np.sum(hist[ind]))
        # Break if we saturated and new_centroids eqls old_centroids
        if np.array(new_centroids - old_centroids).any() == False:
            break
        old_centroids = new_centroids
    return new_centroids


def cartoonize(img):
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img_grey, 1.3, 5)
    if len(faces) == 0:
        return img
    (oy, ox, h, w) = faces[0]
    frame = img[ox : ox + w, oy : oy + h]
    bi = frame
    # Use bilateral Filter to smooths flat regions while keeping edges sharp
    for i in range(5):
        bi = cv2.bilateralFilter(
            bi, d=5, sigmaColor=9, sigmaSpace=7
        )  # bilateral filter

    # img_edge = cv2.Canny(bi, 100, 200)
    img_cartoon = np.array(bi)
    img_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    img_blur = cv2.medianBlur(img_gray, 3)
    # detect Edges to use it later in finding contours
    img_edge = cv2.Canny(img_blur, 100, 200)

    x, y, c = img_cartoon.shape
    # Change photo to HSV
    img_cartoon = cv2.cvtColor(img_cartoon, cv2.COLOR_RGB2HSV)
    # Get Histogram for each channel on HSV
    histograms = []
    for i in range(c):
        if i == 0:
            histograms.append(
                np.histogram(img_cartoon[:, :, i], bins=np.arange(256))[0]
            )
        else:
            histograms.append(
                np.histogram(img_cartoon[:, :, i], bins=np.arange(200))[0]
            )

    # Get the centroids of each group in each channel( one channel has a list of centroids )
    channel_cenroids = []
    for i in range(c):
        channel_cenroids.append(K_means(histograms[i]))

    # Flatten the two dimension of each channel for the image and dims will be (width*height,3)
    img_cartoon = img_cartoon.reshape((-1, c))
    for i in range(c):
        # Get one of Channels of photo
        channel = img_cartoon[:, i]
        # Using broadcasting to subtract each pixel value (stored as colomn vector) from all centroids (stored as row vector)
        # and get index of minimum result to know each pixel value closest to which centroid and assign value of this cetroid to the pixel
        dum = np.abs(channel[:, np.newaxis] - channel_cenroids[i])
        index = np.argmin(dum, axis=1)
        img_cartoon[:, i] = channel_cenroids[i][index]
    # Rerturn Original shape of image
    img_cartoon = img_cartoon.reshape((x, y, c))
    # Return Image to RGB
    img_cartoon = cv2.cvtColor(img_cartoon, cv2.COLOR_HSV2RGB)
    # Get the contours and draw it on cartonized version
    contours, _ = cv2.findContours(img_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img_cartoon, contours, -1, 0, thickness=1)

    img[ox : ox + w, oy : oy + h] = img_cartoon
    return img
