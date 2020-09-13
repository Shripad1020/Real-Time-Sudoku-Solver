import cv2
import numpy as np
import imutils
from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
import Extract__cell as ext
import Predict_digit as pre
import tensorflow as tf
count = 0

frame=cv2.imread(r"D:\TRF IP TASK\suduko\sudo.jpg")
img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
thresh = cv2.GaussianBlur(img, (55, 55), 3)

thresh = cv2.adaptiveThreshold(thresh, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
thresh = cv2.bitwise_not(thresh)

cnts= cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

max_contourArea = 0
max_contourCorner = [0]

for contour in cnts:
    perimeterImg = cv2.arcLength(contour, True)
    epsilon = 0.1 * perimeterImg
    approx = cv2.approxPolyDP(contour, epsilon, True)

    if len(approx) == 4:
        puzzleCnt = approx
        break

cv2.imwrite("Output.jpg", frame)


puzzle = four_point_transform(frame, puzzleCnt.reshape(4, 2))
warped = four_point_transform(img, puzzleCnt.reshape(4, 2))

cv2.imwrite("Output.jpg", puzzle1)
cv2.imshow("puzzle",puzzle1)
l=pre.predict(puzzle1)
print(l)



cv2.waitKey(0)