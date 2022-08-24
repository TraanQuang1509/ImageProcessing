import cv2 as cv
import numpy as np
import imutils

img = cv.imread('images/blisterPack.jpg', cv.IMREAD_COLOR)
blur = cv.GaussianBlur(img, (5, 5), 0)
canny = cv.Canny(blur, 150, 250) 

# contours, hierarchy = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

lines = cv.HoughLinesP(canny, 1, np.pi/180, 10, minLineLength=20, maxLineGap=10)

# for line in lines:
#     x1, y1, x2, y2 = line[0]
#     cv.line(img, (x1, y1), (x2, y2), (0, 0, 255))

x1, y1, x2, y2 = lines[0][0]
cv.line(img, (x1, y1), (x2, y2), (0, 0, 255))
cv.imshow('origin', img)
cv.imshow('canny', canny)
# print(lines[0][0])

alpha = np.arctan((x2-x1)/(y2-y1))*(180/(np.pi))
# alpha = np.arctan((y2-y1)/(x2-x1))*(180/(np.pi))

print(lines[0][0])

rotated = imutils.rotate(img, angle=-alpha)
print(alpha)

cv.imshow('rotate', rotated)

cv.waitKey(0)
cv.destroyAllWindows()