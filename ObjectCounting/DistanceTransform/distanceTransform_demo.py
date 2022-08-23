import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('tomato.png', cv.IMREAD_COLOR)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# show histogram gray image
# plt.hist(gray.ravel(),256,[0,256])
# plt.show()

#convert to binary image
b_img = cv.threshold(gray, 170, 255, cv.THRESH_BINARY_INV)[1]

# Noise processing using morphology - closing and erode
kernel_sq5x5 = np.ones((7,7),dtype=np.uint8)
b_img1 = cv.morphologyEx(b_img, cv.MORPH_CLOSE, kernel_sq5x5, iterations=1)
b_img1 = cv.erode(b_img1,kernel_sq5x5,iterations=1)

# Distance transform
d_img = cv.distanceTransform(b_img, cv.DIST_L2, 3)
# nomalize to 0->1
cv.normalize(d_img, d_img, 0, 1, cv.NORM_MINMAX)
#convert to binary image
out = cv.threshold(d_img, 0.5, 1, cv.THRESH_BINARY)[1]

# convert image from float to int8
result = np.ones(out.shape, dtype=np.uint8)
for i in range(out.shape[0]):
    for j in range(out.shape[1]):
        result[i][j] = out[i][j]

print(result.dtype)
# Find contour and counting
# contours, hierarchy = cv.findContours(out, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
contours, hierarchy = cv.findContours(result, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
print('# of objects', len(contours))

n = 1
for c in contours:
    (x, y), radius = cv.minEnclosingCircle(c)
    center = (int(x), int(y))
    radius = 30
    cv.circle(img, center, radius, (0, 255, 0), 2)
    text = "#" + str(n)
    cv.putText(img, text, center, cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255))
    n += 1

cv.imshow('gray', gray)
cv.imshow('binary', b_img)
cv.imshow('binary_deliate', b_img1)
cv.imshow('distance', out)
cv.imshow('results', img)
cv.waitKey(0)
cv.destroyAllWindows()
