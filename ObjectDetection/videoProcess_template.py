import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)
# filename -> video clip
# 0,1 -> camera
if not cap.isOpened():
    print('can not open video clip/camera')
    exit()

while True:
    # read frame by frame
    ret, frame = cap.read()
    if not ret:
        print(' can not read video frame. Video ended?')
        break
    # your code
    cv.imshow('video', frame)


    # close clip
    if cv.waitKey(1) == ord('q'):
        break
cap.release()
cv.destroyAllWindows()