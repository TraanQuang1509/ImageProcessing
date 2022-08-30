#%%
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# load image
img = cv.imread('rice.jpg', cv.IMREAD_COLOR)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("gray", gray)
# Noise
gray = cv.blur(gray, (3,3))

# show histogram gray image
# plt.hist(gray.ravel(),256,[0,256])
# plt.show()

# roi = gray[0:15, 0:15]                
# thresh = np.mean(roi) - 10

# Binarize the image
thresh = 120
b_img = cv.threshold(gray, thresh, 255, cv.THRESH_BINARY)[1]
print(b_img.dtype)
cv.imshow("binary", b_img)
contours, hierarchy = cv.findContours(b_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# # so luong contour 
print('# of objects', len(contours))

#raw contours 
n = 1
for c in contours:
    (x, y), radius = cv.minEnclosingCircle(c)
    center = (int(x), int(y))
    radius = int(radius) 
    if radius > 2:
        cv.circle(img, center, radius, (0, 255, 0), 2)
        text = "#" + str(n)
        cv.putText(img, text, center, cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255))
        n += 1
print(n)
cv.imshow('result', img)
cv.waitKey(0)