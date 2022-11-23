import cv2
import torch
import torchvision
from torchvision import datasets,models
from torchvision import transforms
from PIL import Image
from matplotlib.image import imread
import numpy as np
from Hand_tracing import HandDetector
from matplotlib.image import imread

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
model = models.resnet34(pretrained=False)
# Number of Input Features in the Last Fully Connected Layer
in_features = model.fc.in_features
# Replacing the Last Fully Connected Layer
fc = torch.nn.Linear(in_features=in_features, out_features=29)
model.fc = fc

#model.load_state_dict(torch.load('./tut1-model/tut1-model.pt'))

checkpoint = torch.load("../Project/Hand-Gestures/checkpoint_90.89")
model.load_state_dict(checkpoint['model_state_dict'])  
model.eval()

test_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])
# img = "../inline_image_preview.jpg"
# img_arr = imread(img)
i=0
while True:
    success,img = cap.read()
    hands,img,img_cropped = detector.findHands(img)
    img_1 = Image.fromarray(img_cropped)
    img_2 = test_transforms(img_1)
    pred = model(torch.unsqueeze(img_2,dim=0))
    #print(pred)
    print(torch.max(pred,dim=1)[1])
    cv2.imshow("Image",img)
    cv2.imshow("Image_2",img_cropped)
    cv2.waitKey(100)