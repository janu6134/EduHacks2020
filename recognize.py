# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 16:19:08 2020

@author: JANAKI
"""
import os
import cv2
import pandas as pd

def recog_faces():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("Labels/Trained.yml")
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath)
    df = pd.read_csv("Ids/faceids.csv")
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Initialize and start realtime video capture
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  # set video width
    cam.set(4, 480)  # set video height
    # Define min window size to be recognized as a face
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)

    while True:
        ret, im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5, minSize = (int(minW), int(minH)),flags = cv2.CASCADE_SCALE_IMAGE)
        for(x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x+w, y+h), (10, 159, 255), 2)
            Id, conf = recognizer.predict(gray[y:y+h, x:x+w])
            if conf < 100:
                aa = df.loc[df['Id'] == Id]['Name'].values
                print(aa)
                confstr = "  {0}%".format(round(100 - conf))
                tt = str(Id)+"-"+aa
                
            else:
                Id = '  Unknown  '
                tt = str(Id)
                confstr = "  {0}%".format(round(100 - conf))

            tt = str(tt)[2:-2]
            if(100-conf) > 67:
                tt = tt + "[Pass]"
                cv2.putText(im, str(tt), (x+5,y-5), font, 1, (255, 255, 255), 2)
            else:
                cv2.putText(im, str(tt), (x + 5, y - 5), font, 1, (255, 255, 255), 2)

            if (100-conf) > 67:
                cv2.putText(im, str(confstr), (x + 5, y + h - 5), font,1, (0, 255, 0),1 )
            elif (100-conf) > 50:
                cv2.putText(im, str(confstr), (x + 5, y + h - 5), font, 1, (0, 255, 255), 1)
            else:
                cv2.putText(im, str(confstr), (x + 5, y + h - 5), font, 1, (0, 0, 255), 1)

        cv2.imshow('Attendance', im)
        if (cv2.waitKey(1) == ord('q')):
            break
    print("Successful")
    cam.release()
    cv2.destroyAllWindows()

recog_faces()