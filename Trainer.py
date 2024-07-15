import os
import numpy as np
import cv2 
recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier("//Users//krishilparikh//CODING//FaceRecog//haarcascade_frontalface_default copy.xml")
path = '/Users/krishilparikh/CODING/FaceRecog/samples'
def Image_and_Labels(path):
    imagePaths = [os.path.join(path , f) for f in os.listdir(path)]
    faceSamples = []
    i_d = []
    for imagePath in imagePaths:
        gray_image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
        img_arr = np.array(gray_image , 'uint8')
        iDList = imagePath.split("_")[0].split("/")
        
        faces = face_cascade.detectMultiScale(img_arr)
        for (x,y,w,h) in faces:
            faceSamples.append(img_arr[y:y+h , x:x+w])
            ids = iDList[-1]
            ids = int(ids)
            i_d.append(ids)

    return faceSamples , i_d

faces , i_d = Image_and_Labels(path)



# print(faces)
# print(i_d_encoded)
recognizer.train(faces , np.array(i_d)) 

data = recognizer.train(faces , np.array(i_d))
recognizer.write("/Users/krishilparikh/CODING/FaceRecog/Trainer/trainer.yml")