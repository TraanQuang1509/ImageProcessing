import numpy as np
import cv2 as cv

# load video /  open camera
cap = cv.VideoCapture('videos/6.mp4')
if not cap.isOpened():
    print('can not open video/camera')
    exit()

while True:
    ret, frame = cap.read() # read a frame from video/camera
    # w = frame.shape[1]
    # h = frame.shape[0]
    if not ret:
        print('can not receive frame or video ended')
        break
    # your code
    # frame = cv.resize(frame, (int(w/3),int(h/3)))
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    low_thres = (30,100,0)
    high_thres = (60,255,255)

    b_img = cv.inRange(hsv, low_thres,high_thres)
    #morphology
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
    out = cv.morphologyEx(b_img, cv.MORPH_OPEN, kernel, iterations=1)
    out = cv.morphologyEx(out, cv.MORPH_CLOSE, kernel, iterations=3)
    contours, hierarchy = cv.findContours(out, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.imshow('close', out)

    for c in contours:
        area = cv.contourArea(c)
        if area > 20: # kích thước
            (x,y),radius = cv.minEnclosingCircle(c)
            
            cArea = np.pi*(radius**2)
            if (1-area/cArea) < 0.3:  # shape filter
                center = (int(x),int(y))
                radius = int(radius)
                cv.circle(frame, center, radius, (0,0,255),2)

    cv.imshow('out', frame)

    # cv.imshow('video', frame)
   
    if cv.waitKey(20) == ord('q'):
        break
cap.release()
cv.destroyAllWindows()


