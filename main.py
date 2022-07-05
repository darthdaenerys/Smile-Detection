import cv2
import numpy as np

print(cv2.__version__)

width=1280
height=720
camera=cv2.VideoCapture(0,cv2.CAP_DSHOW)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
camera.set(cv2.CAP_PROP_FRAME_WIDTH,width)
camera.set(cv2.CAP_PROP_FPS,30)
camera.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))

face_cascade=cv2.CascadeClassifier('haar\data\haarcascade_frontalface_default.xml')
smile_cascade=cv2.CascadeClassifier('haar\data\haarcascade_smile.xml')

while True:
    _,frame=camera.read()
    framegrey=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(framegrey,1.3,5)
    for face in faces:
        x,y,w,h=face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        smileframe=framegrey[y:y+h,x:x+w]
        smiles=smile_cascade.detectMultiScale(smileframe,1.8,20)
        for smile in smiles:
            xs,ys,ws,hs=smile
            cv2.rectangle(frame,(x+xs,y+ys),(x+xs+ws,y+ys+hs),(255,0,0),2)
            cv2.rectangle(frame,(x,y+h),(x+w,y+h+30),(0,255,0),-1)
            cv2.putText(frame,'Smile Detected',(x+5,y+h+25),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255),2)
    cv2.imshow('WebCam',frame)
    if cv2.waitKey(1) & 0xff==ord('q'):
        cv2.destroyAllWindows()
        break
camera.release()