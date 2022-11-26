import glob 
import numpy as np
from matplotlib.image import imread

def get_test_acc(model,path):
    classes = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','del','nothing','space']
    count=0
    total = 0
    for i in glob.glob(path+"/*"):
        label = i.split('/')[-1].split('\\')[-1].split('.')[0].split('_')[0]
        img_1 = imread(i)
        img_1 = 2*((img_1 - np.amin(img_1)) / (np.amax(img_1) - np.amin(img_1)))-1
        pred = model.predict(np.expand_dims(img_1,axis=0))
        pred = classes[np.argmax(pred,axis=-1)[0]]
        print("Prediction",pred)
        print("Label",label)
        if(pred==label):
            count+=1 
        total+=1   
    return ((count/total)*100)

