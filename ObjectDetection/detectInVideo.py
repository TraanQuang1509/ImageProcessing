import numpy as np
import cv2 as cv

# load video /  open camera
cap = cv.VideoCapture('7.mp4')
if not cap.isOpened():
    print('can not open video/camera')
    exit()

while True:
    ret, frame = cap.read() # read a frame from video/camera
    w = frame.shape[1]
    h = frame.shape[0]
    if not ret:
        print('can not receive frame or video ended')
        break
    # your code
    frame = cv.resize(frame, (int(w/5),int(h/5)))
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    low_thres = (30,100,0)
    high_thres = (60,255,255)

    b_img = cv.inRange(hsv, low_thres,high_thres)
    #morphology
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
    out = cv.morphologyEx(b_img, cv.MORPH_OPEN, kernel, iterations=1)
    out = cv.morphologyEx(out, cv.MORPH_CLOSE, kernel, iterations=5)
    contours, hierarchy = cv.findContours(out, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for c in contours:
        area = cv.contourArea(c)
        if area > 1000: # kích thước
            (x,y),radius = cv.minEnclosingCircle(c)
            
            cArea = np.pi*(radius**2)
            if (1-area/cArea) < 0.4:  # shape filter
                center = (int(x),int(y))
                radius = int(radius)
                cv.circle(frame, center, radius, (0,255,0),2)

    cv.imshow('out', frame)

    # cv.imshow('video', frame)
   
    if cv.waitKey(1) == ord('q'):
        break
cap.release()
cv.destroyAllWindows()


