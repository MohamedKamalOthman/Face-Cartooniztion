from tkinter import *
import tkinter.font as font
import cv2
from filters import *
from PIL import Image, ImageTk

root = Tk()
root.resizable(0, 0)
# Create a frame
app = Frame(root, bg="#393E46")
app.grid()
# Create a label in the frame
lmain = Label(app)
lmain.grid()
# button style
button_font = font.Font(family="Helvitica", size=20)
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
    exit = Button(
        app,
        text="Quit",
        bg="#F05454",
        fg="#E8E8E8",
        width=40,
        bd=0,
        font=button_font,
        command=quit,
    )
    exit.grid(row=1, column=0, padx=4)
    filter_0 = Button(
        app,
        text=" Features ",
        bg="#30475E",
        fg="#E8E8E8",
        height=2,
        bd=0,
        font=button_font,
        command=filter_num0,
    )
    filter_0.grid(row=0, column=2, padx=4)
    filter_1 = Button(
        app,
        text=" Glasses ",
        bg="#30475E",
        fg="#E8E8E8",
        height=2,
        bd=0,
        font=button_font,
        command=filter_num1,
    )
    filter_1.grid(row=0, column=3, padx=4)
    filter_2 = Button(
        app,
        text=" Clown ",
        bg="#30475E",
        fg="#E8E8E8",
        height=2,
        bd=0,
        font=button_font,
        command=filter_num2,
    )
    filter_2.grid(row=0, column=4, padx=4)
    filter_3 = Button(
        app,
        text=" Face Mask ",
        bg="#30475E",
        fg="#E8E8E8",
        height=2,
        bd=0,
        font=button_font,
        command=filter_num3,
    )
    filter_3.grid(row=0, column=5, padx=4)
    filter_3 = Button(
        app,
        text=" Cartoonify ",
        bg="#30475E",
        fg="#E8E8E8",
        height=2,
        bd=0,
        font=button_font,
        command=filter_num4,
    )
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
