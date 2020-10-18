from PIL import Image
import cv2,os
import csv
import numpy as np

def train():
    recognizer = cv2.face_LBPHFaceRecognizer.create()
    img_pathlist = [os.path.join("TrainImages", f) for f in os.listdir("TrainImages")] 
    face_Student_Ids = []
    face_images = []
    
    for imgpath in img_pathlist:
        face_image = Image.open(imgpath).convert('L')
        Student_Id = int(os.path.split(imgpath)[-1].split(".")[1])
        face_images.append(np.array(face_image,'uint8'))
        face_Student_Ids.append(Student_Id)
        
    recognizer.train(face_images, np.array(Student_Id))
    
    recognizer.save("TrainedModel\Model.yml")

def capture(studenttext, nametext):        
    Student_Id = (studenttext.get())
    Student_Name= (nametext.get())
    
    vStudent_Ideocapture = cv2.VStudent_IdeoCapture(0)
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    count = 0
    
    while(True):
        ret, img = vStudent_Ideocapture.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_images = detector.detectMultiScale(gray, 1.3, 5)
        
        for (x,y,w,h) in face_images:
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)        
            count = count+1
            cv2.imwrite("TrainImages\ "+Student_Name+"."+Student_Id +'.'+ str(count) + ".jpg", gray[y:y+h,x:x+w])
            cv2.imshow('Register', img)

        if cv2.waitKey(120) & 0xFF == ord('r'):
            break
        
        elif count > 100:
            break
        
    vStudent_Ideocapture.release()
    cv2.destroyAllWindows()
    
    with open('Studentcsv\LoginDetails.csv','a+') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow([Student_Id , Student_Name])
    csvFile.close()
    
    #Let us start training now that we have captured the images
    train()
    
