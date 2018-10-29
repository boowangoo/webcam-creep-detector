import numpy as np
import cv2
import time
import pynput

from pynput.keyboard import Key, Controller

keyboard = Controller()

face_cascade = cv2.CascadeClassifier('classifiers/haarcascade_frontalface_alt.xml')
profile_cascade = cv2.CascadeClassifier('classifiers/haarcascade_profileface.xml')

capture = cv2.VideoCapture(0)

while(capture.isOpened()):
    ret, frame = capture.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(frame_gray, 1.3, 5)
    profiles = profile_cascade.detectMultiScale(frame_gray, 1.3, 5)

    if len(faces) > 0 or len(profiles) > 0:
        print('creep detected')

        # for drawing rectangles around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        for (x, y, w, h) in profiles:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        timeStr = time.strftime("%YY%mM%dd%HH%MM%SS")
        cv2.imwrite("creeps/" + timeStr + ".jpg", frame)

        keyboard.press(Key.cmd)
        keyboard.press('d')
        break

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

keyboard.release(Key.cmd)
keyboard.release('d')

capture.release()
cv2.destroyAllWindows()