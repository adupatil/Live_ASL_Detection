import cv2
import torch
import torchvision
from PIL import Image
from matplotlib.image import imread
import numpy as np

# from cvzone.HandTrackingModule import HandDetector
from Hand_tracing import HandDetector
from matplotlib.image import imread
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
# img = "../inline_image_preview.jpg"
# img_arr = imread(img)
while True:
    success,img = cap.read()
    hands,img,img_cropped = detector.findHands(img)
    cv2.imshow("Image",img)
    cv2.imshow("Image_2",img_cropped)
    cv2.waitKey(1)