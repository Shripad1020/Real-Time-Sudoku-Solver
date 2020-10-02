import cv2
import numpy as np
import imutils
from imutils.perspective import four_point_transform
from tensorflow.keras.preprocessing.image import img_to_array
from skimage.segmentation import clear_border
from Extract__cell import Extract_digit
import tensorflow as tf

class Prediction:
    def predict(puzzle):
        puzzle=cv2.resize(puzzle,(480,480))
        row = []
        l = np.zeros((9, 9), dtype="int")
        print(l)
        h, w = puzzle.shape
        print(puzzle.shape)
        xlen = int(w / 9)
        ylen = int(h / 9)
        xlen = xlen
        ylen = ylen

        k=0

        model = tf.keras.models.load_model("DigitRecognition.model")

        for i in range(2,3,1):
            for j in range(0,9,1):
                xs = i * xlen
                ys = j * ylen
                xe = (i + 1) * xlen
                ye = (j + 1) * ylen
                row.append([xs, ys, xe, ye])
                img1 = puzzle[xs:xe,ys:ye]
                h, w = img1.shape
                k=k+1
                final=Extract_digit.cell(img1,k)

                if final is not None:

                    final = cv2.resize(final, (28, 28))
                    final = final.astype("float") / 255.0
                    final = np.expand_dims(final, axis=0)
                    pred = model.predict(final)
                    print("ans -", np.argmax(pred[0]))
                    l[i,j]=np.argmax(pred[0])
                    print(l)
                else:
                    continue
        return l
