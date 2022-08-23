import cv2 as cv
import numpy as np

img = cv.imread('images/rice.jpg')

canny = cv.Canny(image, threshold1, threshold2)