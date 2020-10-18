# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 19:14:48 2020
@author: JANAKI
"""

import tkinter as tk
import cv2, os
import pandas as pd
from create import capture

tk_window = tk.Tk()
tk_window.title("Student Log-in System")

tk_window.configure(background = 'black')
tk_window.grid_rowconfigure(0, weight=1)
tk_window.grid_columnconfigure(0, weight=1)

message = tk.Label(tk_window, text="Student Log-in system", bg = "black", fg = "white", width = 35, height = 4, font = ('times', 40, 'bold underline')) 
message.place(x = 200, y = 20)

studentid = tk.Label(tk_window, text="Enter ID", width = 25, height = 2, fg = "red", bg = "black", font = ('Helvetica', 20, 'bold')) 
studentid.place(x = 400, y = 190)
studenttext = tk.Entry(tk_window, width = 25, bg = "white", fg = "red", font = ('Helvetica', 20, 'bold'))
studenttext.place(x = 700, y = 200)

name = tk.Label(tk_window, text = "Enter Name", width = 25, fg="red", bg = "black", height = 2, font = ('Helvetica', 20, 'bold')) 
name.place(x = 400, y = 300)
nametext = tk.Entry(tk_window, width = 25, bg = "white", fg = "red", font = ('Helvetica', 20, 'bold'))
nametext.place(x = 700, y = 315) 
    
def captureimage():
    capture()
    message.configure(text = "Images Captured for " +nametext)
    
def Login():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("TrainedModel\Model.yml")

    haarcascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");    
    
    df = pd.read_csv("Studentcsv\LoginDetails.csv")
    videocapture = cv2.VideoCapture(0)
    
    while True:
        ret, im = videocapture.read()
        face_images = haarcascade.detectMultiScale(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY), 1.2, 5)    
        
        for(x,y,w,h) in face_images:
            cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
            student_id, confidence = recognizer.predict((cv2.cvtColor(im,cv2.COLOR_BGR2GRAY), 1.2, 5)[y:y+h,x:x+w])                                   
            
            if(confidence < 50):
                finalres = str(student_id) + "-" + df.loc[df['Id'] == student_id]['Name'].values
                
            else:
                student_id = 'Unknown'                
                finalres = str(student_id) 
            
            cv2.putText(im, str(finalres), (x,y+h), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)        
        
        cv2.imshow('Login', im) 
        if (cv2.waitKey(1)==ord('r')):
            break
        
    videocapture.release()
    cv2.destroyAlltk_windows()
   
login = tk.Button(tk_window, text = "Login", command = Login, fg = "red", bg = "black", width = 25, height = 4, activebackground = "Red", font = ('Helvetica', 15, 'bold'))
login.place(x = 200, y = 500)

reg = tk.Button(tk_window, text = "Register", command = captureimage, fg = "red", bg = "black", width = 25, height = 4, activebackground = "Red" , font = ('Helvetica', 15, 'bold'))
reg.place(x = 500, y = 500)

agedetect = tk.Button(tk_window, text = "Age Detection", command = Login, fg = "blue", bg = "black", width = 25, height = 4, activebackground = "Red" ,font = ('Helvetica', 15, 'bold'))
agedetect.place(x = 800, y = 500)

close = tk.Button(tk_window, text = "Quit", command = tk_window.destroy, fg = "red", bg = "black", width = 25, height = 4, activebackground = "Red" ,font = ('Helvetica', 15, 'bold'))
close.place(x = 1200, y = 500)

 
tk_window.mainloop()