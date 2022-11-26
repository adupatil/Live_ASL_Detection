import cv2
from PIL import Image
from matplotlib.image import imread
import numpy as np
import tensorflow as tf
from tensorflow import keras
from Hand_tracing import HandDetector
from matplotlib.image import imread
import sys
from test_acc import get_test_acc

debug = True

model = keras.models.load_model("../Project/Hand-Gestures/model/asl_classifier_baseline_normalized.h5")

# Creating generator
#test_transform = keras.preprocessing.image.ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input)
## Defining Classes
classes = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','del','nothing','space']

if debug:
    img = "C:/Users/ADU/Pictures/Camera Roll/asl_alphabet_test/asl_alphabet_test/"
else:
    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=1)



while True:
    if debug:
        accuracy = get_test_acc(model,img)
        print(accuracy)
        break
    else:
        success,img = cap.read()
        hands,img,img_cropped = detector.findHands(img)
        cv2.imshow("Image",img)
        cv2.imshow("Image_2",img_cropped)
    
        # Rescaling
        # rescale = tf.keras.layers.Rescaling(1./127.5, offset=-1)
        img_1 = 2*((img_1 - np.amin(img_1)) / (np.amax(img_1) - np.amin(img_1)))-1
        img_1.resize((200,200,3),refcheck=False)
        pred = model.predict(np.expand_dims(img_1,axis=0))
        print(classes[np.argmax(pred,axis=-1)[0]])
        cv2.waitKey(100)