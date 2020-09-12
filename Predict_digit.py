import cv2
import numpy as np
import imutils
from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
import Extract__cell as ext

def predict(puzzle):
    row = []
    l=np.zeros((9,9),dtype=np.float32)

    h, w, c = puzzle.shape
    xlen = int(w / 9)
    ylen = int(h / 9)
    xlen = xlen
    ylen = ylen
    for i in range(9):
        for j in range(9):
            xs = j * xlen
            ys = i * ylen
            xe = (j + 1) * xlen
            ye = (i + 1) * ylen
            row.append([xs, ys, xe, ye])
            img1 = puzzle[xs:xe, ys:ye]
            final=ext.cell(img1)
            # puzzle1=cv2.rectangle(puzzle1,(xs,ys),(xe,ye),(0,255,0),2)
            if final is not None:
                cv2.imshow("Final",final)
                final = cv2.resize(final, (28, 28))
                test = tf.keras.utils.normalize(final, axis=1)
                model = tf.keras.models.load_model("DigitRecognition.model")
                prect = model.predict([[test]])
                print("ans -", np.argmax(prect[0]))
                l[(i,j)]=np.argmax(prect[0])
    print(l)
    #cv2.imshow("Final", puzzle1)
    #cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #img1 = puzzle[row[0][0]:row[0][2], row[0][1]:row[0][3]]