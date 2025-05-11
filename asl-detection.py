import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
# import tensorflow as tf
# model = tf.keras.models.load_model("Model/keras_model.h5")
# model.save("Model/keras_model_fixed.h5")
# classifier = Classifier("Model/keras_model_fixed.h5", "Model/labels.txt")
#
# from tensorflow.keras.layers import DepthwiseConv2D
# custom_objects = {"DepthwiseConv2D": DepthwiseConv2D}
# model = tf.keras.models.load_model("Model/keras_model.h5", custom_objects=custom_objects)

cap = cv2.VideoCapture(1)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300
folder = "Images/C"
counter = 0
while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        imgCropShape = imgCrop.shape
        aspectRatio = h / w
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite)
        print(counter)