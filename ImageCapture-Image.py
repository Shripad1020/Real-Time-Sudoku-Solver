import cv2
import numpy as np
import imutils
from imutils.perspective import four_point_transform

count = 0

    # converting image to grayscale and then applying blurs to remove noise
frame=cv2.imread("sudoku.jpg")
img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
thresh = cv2.GaussianBlur(img, (55, 55), 3)

thresh = cv2.adaptiveThreshold(thresh, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
thresh = cv2.bitwise_not(thresh)
cv2.imshow('thresh', thresh)

# finding contours in image
cnts= cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

# max_contourCorner - maximum contours
# max_contourArea - maximum area of contours
max_contourArea = 0
max_contourCorner = [0]

for contour in cnts:
    perimeterImg = cv2.arcLength(contour, True)
    epsilon = 0.1 * perimeterImg
    approx = cv2.approxPolyDP(contour, epsilon, True)

        # checking maximum contour
    '''if (cv2.contourArea(contour) > 1000) and (cv2.contourArea(contour) > max_contourArea) and (len(approx) == 4):
         max_contourArea = cv2.contourArea(contour)
         max_contourCorner = approx'''

    if len(approx) == 4:
        puzzleCnt = approx
        break

cv2.drawContours(frame, [puzzleCnt], -1, (255, 0, 0), 5)


cv2.imshow("Output",frame)
cv2.imwrite("Output.jpg", frame)


puzzle = four_point_transform(frame, puzzleCnt.reshape(4, 2))
warped = four_point_transform(img, puzzleCnt.reshape(4, 2))
cv2.imshow("Puzzle Transform", puzzle)

#---------------------------------------------------------------

row=[]

h,w,c=puzzle.shape
xlen=int(w/9)
ylen=int(h/9)
xlen=xlen+1
ylen=ylen+1
for i in range(9):
    for j in range (9):
        xs=j*xlen
        ys=i*ylen
        xe=(j+1)*xlen
        ye=(i+1)*ylen
        puzzle=cv2.rectangle(puzzle,(xs,ys),(xe,ye),(0,255,0),2)
cv2.imshow("Final",puzzle)
cv2.waitKey(0)