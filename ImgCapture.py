import cv2
import numpy as np


cap = cv2.VideoCapture(0)

count = 0
while (True):
    ret, frame = cap.read()
    
    # converting image to grayscale and then applying blurs to remove noise
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh = cv2.GaussianBlur(img, (55, 55), 3)

    thresh = cv2.adaptiveThreshold(thresh, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    cv2.imshow('thresh',thresh)

    # finding contours in image
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # max_contourCorner - maximum contours
    # max_contourArea - maximum area of contours
    max_contourArea = 0
    max_contourCorner = [0]

    for contour in contours: 
    	perimeterImg = cv2.arcLength(contour, True)
    	epsilon = 0.1 * perimeterImg
    	approx = cv2.approxPolyDP(contour, epsilon, True)

    	# checking maximum contour
    	if (cv2.contourArea(contour) > 1000) and (cv2.contourArea(contour) > max_contourArea) and (len(approx) == 4) :
    		max_contourArea = cv2.contourArea(contour)
    		max_contourCorner = approx

    if len(max_contourCorner) == 4:
    	count += 1
    	cv2.drawContours(frame, [max_contourCorner], -1, (255, 0, 0), 5)
    	cv2.drawContours(frame, max_contourCorner, -1, (0, 255, 0), 8)  	
    else:
    	count = 0
    
    cv2.imshow("All", frame)
    if count == 4:
    	cv2.imwrite("Output.jpg", frame)

    key = cv2.waitKey(1)
    if key == 27 :
        break

cap.release()
cv2.destroyAllWindows()