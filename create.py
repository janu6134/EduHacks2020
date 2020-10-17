import cv2,os
import csv
import numpy as np
from PIL import Image

def train():
    recognizer = cv2.face_LBPHFaceRecognizer.create()
    harcascadePath = "haarcascade_frontalface_default.xml"
    path = "TrainImages"
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
    faces=[]
    Ids=[]
    
    for imagePath in imagePaths:
        pilImage=Image.open(imagePath).convert('L')
        #Now we are converting the PIL image into numpy array
        imageNp=np.array(pilImage,'uint8')
        #getting the Id from the image
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces.append(imageNp)
        Ids.append(Id)
    recognizer.train(faces, np.array(Id))
    recognizer.save("TrainedModel\Trainner.yml")

def capture(studenttext, nametext):        
    Id=(studenttext.get())
    name=(nametext.get())
    
    videocapture = cv2.VideoCapture(0)
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector=cv2.CascadeClassifier(harcascadePath)
    count=0
    while(True):
        ret, img = videocapture.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)        
            count=count+1
            cv2.imwrite("TrainImages\ "+name +"."+Id +'.'+ str(count) + ".jpg", gray[y:y+h,x:x+w])
            cv2.imshow('Register',img)
        #wait for 100 miliseconds 
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        
        elif count>100:
            break
    videocapture.release()
    cv2.destroyAllWindows() 
    row = [Id , name]
    with open('StudentDetails\StudentDetails.csv','a+') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row)
    csvFile.close()
    train()
    
