import cv2
import numpy as np

img = cv2.imread("OI.png", 0)

# 大津法实现
hist, bins = np.histogram(img.ravel(), 256, [0, 256])
total_pixels = img.shape[0] * img.shape[1]

best_threshold = 0
best_variance = 0

for threshold in range(0, 256):
    class1_pixels = np.sum(hist[:threshold])
    class2_pixels = total_pixels - class1_pixels

    class1_mean = np.sum(np.arange(threshold) * hist[:threshold]) / class1_pixels if class1_pixels > 0 else 0.
    class2_mean = np.sum(np.arange(threshold, 256) * hist[threshold:]) / class2_pixels if class2_pixels > 0 else 0.

    between_class_variance = class1_pixels * class2_pixels * (class1_mean - class2_mean) ** 2
    if between_class_variance > best_variance:
        best_threshold = threshold
        best_variance = between_class_variance

output_otsu = cv2.threshold(img, best_threshold, 255, cv2.THRESH_BINARY)[1]

# 迭代法实现
init_threshold = 128
max_iterations = 1000
convergence_threshold = 1e-10

threshold = init_threshold
prev_threshold = threshold - 1

iteration = 0
error = 1.

while iteration < max_iterations and abs(error) > convergence_threshold:
    class1_pixels = img[img < threshold]
    class2_pixels = img[img >= threshold]
    class1_mean = np.mean(class1_pixels) if len(class1_pixels) > 0 else 0.
    class2_mean = np.mean(class2_pixels) if len(class2_pixels) > 0 else 0.

    new_threshold = (class1_mean + class2_mean) / 2.
    error = new_threshold - threshold

    prev_threshold = threshold
    threshold = new_threshold
    iteration += 1

output_iter = cv2.threshold(img, int(threshold), 255, cv2.THRESH_BINARY)[1]

# 显示输出图像
cv2.imshow("Original Image", img)
cv2.imshow("Otsu Thresholding", output_otsu)
cv2.imshow("Iterative Thresholding", output_iter)

cv2.waitKey(0)