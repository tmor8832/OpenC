import os #needed to navigate the directories on the pc
import cv2 as cv
import numpy as np

#list of names of the people and the directory address where images are stored

people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']
DIR = r'C:\Coding\OpenCV\Resources\Faces\train'

haar_cascade = cv.CascadeClassifier('haar_face.xml') #saved xml file from openCV github for front face detection

features = [] #empty lists for storing features and labels for the face detection
labels = []

def create_train():
    for person in people: #get the names from the list of people generated above
        path = os.path.join(DIR, person)#navigate to each specific folder by naming convention e.g.
        # C:\Coding\OpenCV\Resources\Faces\train\Ben Affleck or Maddonna etc
        label = people.index(person) #label with the index so Ben Affleck would be 0, Elton 1 etc

        for img in os.listdir(path): #for each image found in the specificied path
            img_path = os.path.join(path,img) 

            img_array = cv.imread(img_path) #get the image array
            if img_array is None:
                continue 
                
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY) #convert image to grayscale

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
            #create the facial rectangle, scale up and min neighbors determines how hard it is to detect

            for (x,y,w,h) in faces_rect: #using the coordinates from the detected face
                faces_roi = gray[y:y+h, x:x+w] #crop out the region which is just the detected face
                features.append(faces_roi) #append that region
                labels.append(label) #append the corresponding label in numeric form

create_train()
print('Training done ---------------')

features = np.array(features, dtype='object') #arrays of features and labels
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create() #create local binary pattern histogram model

# Train the Recognizer on the features list and the labels list
face_recognizer.train(features,labels) #pass the model the trained features and labels

face_recognizer.save('face_trained.yml') #save the trained model as a yml file for re-use
np.save('features.npy', features) #save these as numpy files
np.save('labels.npy', labels)