import tkinter

import cv2
import numpy as np
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

def detect_gesture(img):
    """손가락 개수를 세어서 가위, 바위, 보를 판별합니다."""

    # img = cv.resize(img, (720, 480))
    # img = cv2.resize(img, None, fx=0.5, fy=0.5)
    img: np.ndarray

    # clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # R, G, B = cv.split(img)
    # R = clahe.apply(R)
    # G = clahe.apply(G)
    # B = clahe.apply(B)
    # img = cv.merge((R,G,B))

    rn = cv2.blur(img, (5, 5))
    # cv.imshow("1. blur", rn)

    hsv = cv2.cvtColor(rn, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 30, 90], dtype="uint8")
    upper = np.array([50, 255, 255], dtype="uint8")
    skin_region_hsv = cv2.inRange(hsv, lower, upper)

    blurred = cv2.medianBlur(skin_region_hsv, 5)
    cv2.imshow("2. medianBlur", blurred)

    contours, hierarchy = cv2.findContours(blurred, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = max(contours, key=lambda x: cv2.contourArea(x))
    cv2.drawContours(img, [contours], -1, (0, 255, 0), 2)

    hull = cv2.convexHull(contours)
    cv2.drawContours(img, [hull], -1, (0, 0, 0), 2)
    hull = cv2.convexHull(contours, returnPoints=False)
    defects = cv2.convexityDefects(contours, hull)

    acutes = 0
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i][0]
        start = tuple(contours[s][0])
        end = tuple(contours[e][0])
        far = tuple(contours[f][0])

        a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
        c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)

        angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
        if angle <= np.pi / 2:
            acutes += 1
            cv2.circle(img, start, 4, [0, 0, 255], -1)

    fingers = acutes + 1 if acutes > 0 else acutes

    if not fingers:
        _shape = "paper"
    elif fingers == 2:
        _shape = "rock"
    elif fingers == 5:
        _shape = "scissors"
    else:
        _shape = None

    return _shape


def update():
    ret, frame = cap.read()
    if not ret:
        print("카메라를 열 수 없습니다.")
        return
    frame_orig = frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (400, height))
    frame = Image.fromarray(frame)
    frame = ImageTk.PhotoImage(frame)

    camera_label.configure(image=frame)
    camera_label.image = frame

    result = detect_gesture(frame_orig)
    if result!=None:
        result_image = Image.open(f"{result}.png")
        result_image = ImageTk.PhotoImage(result_image)
        result_label.configure(image=result_image)
        result_label.image = result_image

    window.after(int(1000 / fps), update)


window.after(0, update)
window.mainloop()

cap.release()
