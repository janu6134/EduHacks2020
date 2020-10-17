# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 19:14:48 2020
@author: JANAKI
"""

import tkinter as tk
import cv2,os
import pandas as pd
from create import capture

window = tk.Tk()
window.title("Student Log-in System")

window.configure(background='black')
window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)

message = tk.Label(window, text="Student Log-in system" ,bg="black"  ,fg="white"  ,width=50  ,height=3,font=('times', 30, 'bold underline')) 
message.place(x=200, y=20)

studentid = tk.Label(window, text="Enter ID",width=20  ,height=2  ,fg="red"  ,bg="black" ,font=('times', 15, ' bold ') ) 
studentid.place(x=400, y=200)
studenttext = tk.Entry(window,width=20  ,bg="white" ,fg="red",font=('times', 15, ' bold '))
studenttext.place(x=700, y=215)

name = tk.Label(window, text="Enter Name",width=20  ,fg="red"  ,bg="black"    ,height=2 ,font=('times', 15, ' bold ')) 
name.place(x=400, y=300)
nametext = tk.Entry(window,width=20  ,bg="white"  ,fg="red",font=('times', 15, ' bold ')  )
nametext.place(x=700, y=315) 
    
def captureimage():
    capture()
    message.configure(text= "Images Saved for ID : " + studenttext +" Name : "+ nametext)
    
def Login():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("TrainedModel\Model.yml")

    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");    
    
    df=pd.read_csv("Studentcsv\StudentDetails.csv")
    videocapture = cv2.VideoCapture(0)
    
    while True:
        ret, im = videocapture.read()
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        faces=faceCascade.detectMultiScale(gray, 1.2,5)    
        
        for(x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
            student_id, confidence= recognizer.predict(gray[y:y+h,x:x+w])                                   
            
            if(confidence < 50):
                valuecheck = df.loc[df['Id'] == student_id]['Name'].values
                finalres = str(student_id) + "-" + valuecheck
                
            else:
                student_id = 'Unknown'                
                finalres = str(student_id) 
            
            cv2.putText(im, str(finalres), (x,y+h), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)        
        
        cv2.imshow('Login',im) 
        if (cv2.waitKey(1)==ord('q')):
            break
        
    videocapture.release()
    cv2.destroyAllWindows()
   
login = tk.Button(window, text="Login", command=Login  ,fg="red"  ,bg="black"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
login.place(x=200, y=500)

reg = tk.Button(window, text="Register", command=captureimage  ,fg="red"  ,bg="black"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
reg.place(x=500, y=500)

agedetect = tk.Button(window, text="Age Detection", command=Login  ,fg="blue"  ,bg="black"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
agedetect.place(x=800, y=500)

quitWindow = tk.Button(window, text="Quit", command=window.destroy  ,fg="red"  ,bg="black"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
quitWindow.place(x=1100, y=500)

 
window.mainloop()