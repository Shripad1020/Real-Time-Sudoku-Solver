import cv2
import numpy as np
import imutils
from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border


def cell(img1):
    img1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    img = cv2.threshold(img1, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    img = clear_border(img)
    #img = cv2.bitwise_not(img)
    cnts= cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    if(len(cnts)==0):
        return 0
    else:
        print(cnts)
    #print(cnts[0][0][1][0][1])
    cv2.imshow("cell",img)
    return img