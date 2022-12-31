from tkinter import *

import cv2
from filters import *
from PIL import Image, ImageTk

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

    def filter_num3():
        nonlocal filters
        filters = 3

    def filter_num4():
        nonlocal filters
        filters = 4

    # function for video streaming
    exit = Button(app, text="Quit", fg="red", command=quit)
    exit.grid(row=0, column=7, padx=4)
    filter_0 = Button(app, text="Filter 1", fg="green", command=filter_num0)
    filter_0.grid(row=0, column=2, padx=4)
    filter_1 = Button(app, text="Filter 2", fg="green", command=filter_num1)
    filter_1.grid(row=0, column=3, padx=4)
    filter_2 = Button(app, text="Filter 3", fg="green", command=filter_num2)
    filter_2.grid(row=0, column=4, padx=4)
    filter_3 = Button(app, text="Filter 4", fg="green", command=filter_num3)
    filter_3.grid(row=0, column=5, padx=4)
    filter_3 = Button(app, text="Cartoonify", fg="green", command=filter_num4)
    filter_3.grid(row=0, column=6, padx=4)
    dp = None
    prev = -1

    def video_stream():
        nonlocal dp, prev
        _, frame = cap.read()
        if filters == 0:
            if prev != filters:
                dp = None
            frame, dp = render_filter_0(frame, dp)
        elif filters == 1:
            if prev != filters:
                dp = None
            frame, dp = render_filter_1(frame, dp)
        elif filters == 2:
            if prev != filters:
                dp = None
            frame, dp = render_filter_2(frame, dp)
        elif filters == 3:
            if prev != filters:
                dp = None
            frame, dp = render_filter_3(frame, dp)
        elif filters == 4:
            frame = cartoonize(frame)
        prev = filters
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        lmain.after(1, video_stream)

    video_stream()


main()
root.mainloop()
