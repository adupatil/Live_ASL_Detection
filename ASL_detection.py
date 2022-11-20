# import opencv
import cv2
#Loading cascade
smile_cascade = cv2.CascadeClassifier("Hand_Recognition\Hand_haar_cascade.xml")

def detect(gray,frame):
    smiles = smile_cascade.detectMultiScale(gray,1.7,25)
    for (x,y,w,h) in smiles:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
    return frame

# Recognizing smiles
video_capture = cv2.VideoCapture(0)

while True:
    _,frame = video_capture.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    canvas = detect(gray,frame)
    cv2.imshow('Video',canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()