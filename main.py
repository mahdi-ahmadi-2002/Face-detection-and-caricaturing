import cv2 as cv
import numpy as np
import math

img = cv.imread('image1.jpg')
img = cv.resize(img, (500, 500), interpolation=cv.INTER_CUBIC)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

haar_cascade = cv.CascadeClassifier('haar_face.xml')

faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=2)


def transition(img, matrix, name):
    temp = img.copy()

    for (x, y, w, h) in faces_rect:
        transMat = np.float32(matrix)
        dimensions = (temp[y:y + w, x: x + h].shape[1], temp[y:y + w, x: x + h].shape[0])
        temp[y:y + w, x: x + h] = cv.warpAffine(temp[y:y + w, x: x + h], transMat, dimensions)
    temp = cv.resize(temp, (500, 500), interpolation=cv.INTER_CUBIC)
    gray = cv.cvtColor(temp, cv.COLOR_BGR2GRAY)
    ret, thresh1 = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV)
    dst_TELEA = cv.inpaint(temp, thresh1, 10, cv.INPAINT_TELEA)
    cv.imshow(name, dst_TELEA)


transition(img, [[1.2, 0, 0],
                 [0, 1.2, 0]],
           "bigger faces")

transition(img, [[0.7, 0, 0],
                 [0, 0.7, 0]],
           "smaller faces")

transition(img, [[0.866, 1 / 2, 1],
                 [-1 / 2, 0.866, 0]],
           "rotation")

transition(img, [[1, 0, 0],
                 [0, 1, 15]],
           "shift")

transition(img, [[2, 0, 0],
                 [0, 1 / 2, 0]],
           "smaller faces in one dimension")

transition(img, [[1, 0, 0],
                 [0.4, 1, 0]],
           "incurve horizontally")

transition(img, [[1, 0.4, 0],
                 [0, 1, 0]],
           "incurve vertically")

cv.waitKey(0)
