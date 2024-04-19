import cv2
import numpy as np
import time

faceClassifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

videoCam = cv2.VideoCapture(0)

if not videoCam.isOpened():
    print("Camera cannot be accessed")
    exit()

qPressed = False
while (qPressed == False):
    ret, frame = videoCam.read()

    if ret == True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceClassifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=2)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # print("Number of detected faces: ", len(faces))
        text = "Number of Detected Faces = " + str(len(faces))

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, text, (0, 30), font, 1, (255, 0, 0), 1)

        cv2.imshow("Result", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            qPressed = True
            break

videoCam.release()
cv2.destroyAllWindows()
