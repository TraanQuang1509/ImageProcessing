import cv2 as cv
import numpy as np

img = cv.imread('images/trafficSign.jpg', cv.IMREAD_COLOR)
blur = cv.GaussianBlur(img, (5,5), 0)

hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)

# detect red color
low_thresh = (160, 100, 100)
high_thresh = (180, 255, 255)
b_img = cv.inRange(hsv, low_thresh, high_thresh)

contours, heirarchy = cv.findContours(b_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

n = 1
for c in contours:
    x, y, w, h = cv.boundingRect(c)
    if h > 1 and w > 1 :
        roi = img[y:y+h, x:x+w]
        canny = cv.Canny(roi, 150, 250)
        title = 'roi' + str(n)
        cv.imshow(title, canny) 
        cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        n += 1

cv.imshow('traffic', img)

cv.waitKey(0)
cv.destroyAllWindows()