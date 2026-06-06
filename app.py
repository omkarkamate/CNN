import cv2
import os

img = cv2.imread("dataset/helmet/" + os.listdir("dataset/helmet")[0])

print(img.shape)