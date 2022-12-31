from tkinter import *
from PIL import ImageTk, Image
import cv2
from ffextractor import FeaturesLocator
import numpy as np


locator = FeaturesLocator(load=True, path="./results")
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def render_filter_0(img):
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img_grey, 1.3, 5)
    if len(faces) == 0:
        return img
    (y, x, h, w) = faces[0]
    face = img_grey[x: x + w, y: y + h]
    keypoints = locator.findfeatures(face)
    cv2.ellipse(img, np.add(keypoints['left_eye'], [y, x]), (9, 4),
                0, 0, 360, color=255, thickness=1)
    cv2.ellipse(img, np.add(keypoints['right_eye'], [y, x]), (9, 4),
                0, 0, 360, color=255, thickness=1)
    cv2.ellipse(img, np.add(keypoints['nose'], [y, x]),
                (9, 4), 0, 0, 360, color=255, thickness=1)
    cv2.ellipse(img, np.add(keypoints['mouth'], [y, x]),
                (9, 4), 0, 0, 360, color=255, thickness=1)
    cv2.rectangle(img, (y, x), (y + h, x + w), (0, 0, 255), 2)
    return img


def add_transparent_image(background, foreground, x_offset=None, y_offset=None):
    bg_h, bg_w, bg_channels = background.shape
    fg_h, fg_w, fg_channels = foreground.shape

    assert bg_channels == 3, f'background image should have exactly 3 channels (RGB). found:{bg_channels}'
    assert fg_channels == 4, f'foreground image should have exactly 4 channels (RGBA). found:{fg_channels}'

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
    foreground = foreground[fg_y:fg_y + h, fg_x:fg_x + w]
    background_subsection = background[bg_y:bg_y + h, bg_x:bg_x + w]

    # separate alpha and color channels from the foreground image
    foreground_colors = foreground[:, :, :3]
    alpha_channel = foreground[:, :, 3] / 255  # 0-255 => 0.0-1.0

    # construct an alpha_mask that matches the image shape
    alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))

    # combine the background with the overlay image weighted by alpha
    composite = background_subsection * (1 - alpha_mask) + foreground_colors * alpha_mask

    # overwrite the section of the background image that has been updated
    background[bg_y:bg_y + h, bg_x:bg_x + w] = composite


def render_filter_1(img):
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img_grey, 1.3, 5)
    if len(faces) == 0:
        return img
    glasses = cv2.imread('./filters/glasses.png', cv2.IMREAD_UNCHANGED)
    (y, x, h, w) = faces[0]
    glasses = cv2.resize(glasses, (np.array([w, h]) * 0.75).astype(int))
    face = img_grey[x: x + w, y: y + h]
    keypoints = locator.findfeatures(face)
    center = (keypoints['left_eye'] + keypoints['right_eye']) // 2
    add_transparent_image(img, glasses, center[0] + y, center[1] + x)
    return img


def render_filter_2(img):
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img_grey, 1.3, 5)
    if len(faces) == 0:
        return img
    nose = cv2.imread('./filters/clown_nose.png', cv2.IMREAD_UNCHANGED)
    wig = cv2.imread('./filters/clown_wig.png', cv2.IMREAD_UNCHANGED)
    (y, x, h, w) = faces[0]
    nose = cv2.resize(nose, (np.array([w, h]) * 0.30).astype(int))
    wig = cv2.resize(wig, (np.array([w, h]) * 0.95).astype(int))
    face = img_grey[x: x + w, y: y + h]
    keypoints = locator.findfeatures(face)
    center_nose = keypoints['nose']
    center_wig = (keypoints['left_eye'] + keypoints['right_eye']) // 2
    add_transparent_image(img, nose, center_nose[0] + y, center_nose[1] + x)
    add_transparent_image(img, wig, center_wig[0] + y, center_wig[1] + x - h // 3)
    return img


root = Tk()
# Create a frame
app = Frame(root, bg="white")
app.grid()
# Create a label in the frame
lmain = Label(app)
lmain.grid()
# Capture from camera
cap = cv2.VideoCapture(0)


def main():
    filters = -1

    def filter_num0():
        nonlocal filters
        filters = 0

    def filter_num1():
        nonlocal filters
        filters = 1

    def filter_num2():
        nonlocal filters
        filters = 2

    # function for video streaming
    exit = Button(app,
                  text="QUIT",
                  fg="red",
                  command=quit)
    exit.grid(row=1, column=1, padx=4)
    filter_0 = Button(app,
                      text="Filter 0",
                      fg="blue",
                      command=filter_num0)
    filter_0.grid(row=1, column=2, padx=4)
    filter_1 = Button(app,
                      text="Filter 1",
                      fg="blue",
                      command=filter_num1)
    filter_1.grid(row=1, column=3, padx=4)
    filter_2 = Button(app,
                      text="Filter 2",
                      fg="blue",
                      command=filter_num2)
    filter_2.grid(row=1, column=4, padx=4)

    def video_stream():
        _, frame = cap.read()
        if filters == 0:
            frame = render_filter_0(frame)
        if filters == 1:
            frame = render_filter_1(frame)
        if filters == 2:
            frame = render_filter_2(frame)
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        lmain.after(1, video_stream)

    video_stream()


main()
root.mainloop()
