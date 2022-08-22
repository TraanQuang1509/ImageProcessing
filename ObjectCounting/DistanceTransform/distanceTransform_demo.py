import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('tomato.png', cv.IMREAD_COLOR)
print(img[1])