import cv2
import numpy as np
import imutils
from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border


class Extract_digit:
    def cell(img1,k):

        img = cv2.threshold(img1, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        img = clear_border(img)

        cnts= cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        if (len(cnts) == 0):
            return None

        c = max(cnts, key=cv2.contourArea)
        mask = np.zeros(img.shape, dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)

        (h, w) = img.shape

        percentFilled = cv2.countNonZero(mask) / float(w * h)
        img = cv2.bitwise_and(img, img, mask=mask)
        cv2.imshow(str(k), img)
        
        return img


