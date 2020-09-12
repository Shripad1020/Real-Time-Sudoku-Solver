import cv2
import numpy as np
from operator import itemgetter

cap = cv2.VideoCapture(0)

adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
blockSize = 11
while (True):
	ret, frame = cap.read()
	if ret:
		# converting image to grayscale and then applying blurs to remove noise
		img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		thresh = cv2.GaussianBlur(img, (55, 55), 3)

		thresh = cv2.adaptiveThreshold(thresh, 255, adaptiveMethod, cv2.THRESH_BINARY, blockSize, 2)
		cv2.imshow('thresh',thresh)

		txt = "MEAN_C" if adaptiveMethod == cv2.ADAPTIVE_THRESH_MEAN_C else "GAUSSIAN_C"
		cv2.putText(frame, "blockSize: {:2d} | adaptiveMethod: {}".format(blockSize, txt), (10, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (0, 0, 0), 1)

		# finding contours in image
		_, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		# max_contourCorner - maximum contours
		# max_contourArea - maximum area of contours
		max_contourArea = 0
		max_contourCorner = None

		for contour in contours: 
			perimeterImg = cv2.arcLength(contour, True)
			epsilon = 0.1 * perimeterImg
			approx = cv2.approxPolyDP(contour, epsilon, True)

			# checking maximum contour
			if (cv2.contourArea(contour) > 20000) and (cv2.contourArea(contour) > max_contourArea) and (len(approx) == 4) :
				max_contourArea = cv2.contourArea(contour)
				max_contourCorner = approx

		if max_contourCorner is not None:
			cv2.drawContours(frame, [max_contourCorner], -1, (255, 0, 0), 5)
			cv2.drawContours(frame, max_contourCorner, -1, (0, 255, 0), 8) 

			# Finding 4-Corner Co-ordinates
			pts = np.vstack(max_contourCorner).squeeze()
			pts = sorted(pts, key = itemgetter(1))
			if pts[0][0] < pts[1][0]:
				if pts[3][0] < pts[2][0]:
					pts1 = np.float32([pts[0], pts[1], pts[3], pts[2]])
				else:
					pts1 = np.float32([pts[0], pts[1], pts[2], pts[3]])
			else:
				if pts[3][0] < pts[2][0]:
					pts1 = np.float32([pts[1], pts[0], pts[3], pts[2]])
				else:
					pts1 = np.float32([pts[1], pts[0], pts[2], pts[3]])

			# Four point transform
			pts2 = np.float32([[0, 0], [500, 0], [0, 500], [500, 500]])
			matrix = cv2.getPerspectiveTransform(pts1, pts2)
			pazzle = cv2.warpPerspective(frame, matrix, (500, 500))
			cv2.imshow("pazzle", pazzle)

			#cv2.putText(frame, "1", (pts[0][0], pts[0][1]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 1)
			#cv2.putText(frame, "2", (pts[1][0], pts[1][1]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 1)
			#cv2.putText(frame, "3", (pts[2][0], pts[2][1]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 1)
			#cv2.putText(frame, "4", (pts[3][0], pts[3][1]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 1)
		
		cv2.imshow("All", frame)
		
		key = cv2.waitKey(1) & 0xFF
		if key == 27 :
			cv2.imwrite("Output.jpg", pazzle)
			break
		if key == ord('p'):
			blockSize = min(21, blockSize + 2)
		if key == ord('m'):
			blockSize = max(3, blockSize - 2)
		if key == ord('o'):
			if adaptiveMethod == cv2.ADAPTIVE_THRESH_GAUSSIAN_C:
				adaptiveMethod = cv2.ADAPTIVE_THRESH_MEAN_C
			else:
				adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C

cap.release()
cv2.destroyAllWindows()