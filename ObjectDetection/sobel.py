import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# load image
img = cv.imread('Lenna.png', cv.IMREAD_COLOR)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

kernel1 = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])

kernel2 = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]])

kernel3 = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]])

kernel4= np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]])

gray = cv.GaussianBlur(gray, (5, 5), 0)

filter1 = cv.filter2D(src=gray, kernel=kernel1, ddepth=-1)
filter2 = cv.filter2D(src=gray, kernel=kernel2, ddepth=-1)
filter12 = filter1 + filter2

filter3 = cv.filter2D(src=gray, kernel=kernel3, ddepth=-1)
filter4 = cv.filter2D(src=gray, kernel=kernel4, ddepth=-1)
filter34 = filter3 + filter4

filter5 = filter12 + filter34

cv.imshow('vertical1', filter1)
cv.imshow('vertical2', filter2)
cv.imshow('vertical12', filter12)

cv.imshow('vertical3', filter3)
cv.imshow('vertical4', filter4)
cv.imshow('vertical34', filter34)

cv.imshow('vertical5', filter5)

cv.waitKey(0)