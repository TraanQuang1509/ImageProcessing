#%%
import cv2 as cv
import numpy as np

img = cv.imread('halloween.jpg', cv.IMREAD_COLOR)
print('dang bien cua tung pixel:', img.dtype)

#%% lay du lieu file hinh
print('so chieu data', img.ndim)
print('tong so pixel', img.size)
print('so piexl trong tung dimension', img.shape)

#%% chieu dai va chieu rong cua tam hinh
width = img.shape[0]
height = img.shape[1]
print(width, height)

#%% doc va gan gia tri cho pixel
print('gia tri pixel tai hang 3. cot 4 =', img[0, 0])
img[3,4] = [255, 255, 255]
print('gia tri pixel tai hang 3. cot 4 =', img[3, 4])

#%% tao hieu ung cho tam hinh
out = np.zeros((height, width, 3), dtype=np.uint8)

#VD1
# for i in np.arange(height -1, -1, -1):
#     out[i, :, :] = img[i, :, :]
#     cv.imshow('Halloween', out)
#     cv.waitKey(5) 

#VD2
for i in np.arange(0, width, 1):
    out[:, i, :] = img[:, i, :]
    cv.imshow('Halloween', out)
    cv.waitKey(5) 
