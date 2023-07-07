import tkinter

import cv2
from PIL import Image, ImageTk

width, height = 800, 400
fps = 30

window = tkinter.Tk()
window.title("Rock Paper Scissors")
window.geometry(f'{width}x{height}')
window.resizable(False, False)

camera_label = tkinter.Label(window)
camera_label.pack(side="left")
result_label = tkinter.Label(window)
result_label.pack(side="right")

cap = cv2.VideoCapture(0)


def detect_gesture(frame):
    """손가락 개수를 세어서 가위, 바위, 보를 판별합니다."""
    return "rock"


def update():
    ret, frame = cap.read()
    if not ret:
        print("카메라를 열 수 없습니다.")
        return

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (400, height))
    frame = Image.fromarray(frame)
    frame = ImageTk.PhotoImage(frame)

    camera_label.configure(image=frame)
    camera_label.image = frame

    result = detect_gesture(frame)
    result_image = Image.open(f"{result}.png")
    result_image = ImageTk.PhotoImage(result_image)
    result_label.configure(image=result_image)
    result_label.image = result_image

    window.after(int(1000 / fps), update)


window.after(0, update)
window.mainloop()

cap.release()
