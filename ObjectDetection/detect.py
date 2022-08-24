import cv2 as cv
import numpy as np

img = cv.imread('images/1.jpg', cv.IMREAD_COLOR)

hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)\

low_thres = (30,100,0)
high_thres = (60,255,255)

b_img = cv.inRange(hsv, low_thres,high_thres)
#morphology
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
out = cv.morphologyEx(b_img, cv.MORPH_OPEN, kernel, iterations=1)
out = cv.morphologyEx(out, cv.MORPH_CLOSE, kernel, iterations=3)
contours, hierarchy = cv.findContours(out, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

cv.imshow('binary', b_img)
cv.imshow('close', out)
