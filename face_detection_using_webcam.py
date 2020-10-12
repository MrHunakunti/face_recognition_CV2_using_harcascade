import cv2
import numpy as np


# LOADING THE FACE DATA
trained_face_data = cv2.CascadeClassifier(r"resources/haarcascade_frontalface_default.xml")
webcam = cv2.VideoCapture(0)

while True:
    # success = Bool,
    # frame = actual frame by frame image
    success, frame = webcam.read()

    # converting the frames in grey scale for detection
    grey_scale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # face coordinates detection
    face_cordinates_detection = trained_face_data.detectMultiScale(image=frame )

    # drawing the rectangle
    for x,y,w,h in face_cordinates_detection:
        rectangle = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.imshow("webcam", frame)
    # to stop the video
    if cv2.waitKey(delay=100) & 0xff==ord("q"):
        break




