import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

img = cv2.imread("testNum.png", 0)
img = cv2.bitwise_not(img)
print(img.size)
print(img.shape)

plt.imshow(img, cmap = plt.cm.binary)
plt.show()

img = cv2.resize(img, (28, 28))
plt.imshow(img, cmap = plt.cm.binary)
plt.show()

test = tf.keras.utils.normalize(img, axis=1)
model = tf.keras.models.load_model("DigitRecognition.model")
prect = model.predict([[test]])

for i in range (10):
    b = prect[0][i]
    print("prob", i, b)
    
print("ans -", np.argmax(prect[0]))