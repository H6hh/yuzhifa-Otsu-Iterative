import cv2
import imutils
import numpy as np

# 在某一范围(A, B)突出灰度，其他灰度值保持不变
image = cv2.imread('D:\python huidu\gougou.png')
gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

r_left, r_right = 150, 230
r_min, r_max = 0, 255
level_img = np.zeros((gray_img.shape[0], gray_img.shape[1]), dtype=np.uint8)
for i in range(gray_img.shape[0]):
    for j in range(gray_img.shape[1]):
        if r_left <= gray_img[i, j] <= r_right:
            level_img[i, j] = r_max
        else:
            level_img[i, j] = gray_img[i, j]

cv2.imshow('origin image', imutils.resize(image, 480))
cv2.imshow('level image', imutils.resize(level_img, 480))
if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()
