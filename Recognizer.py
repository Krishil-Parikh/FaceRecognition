import cv2      

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("/Users/krishilparikh/CODING/FaceRecog/Trainer/trainer.yml")
face_cascade = cv2.CascadeClassifier("/Users/krishilparikh/CODING/FaceRecog/haarcascade_frontalface_default copy.xml")

id = 4

names = ['' , 'Krishil' , 'Roshni', 'Dev']

cap = cv2.VideoCapture(0)

while True:
    _ , frame = cap.read()
    frame = cv2.flip(frame , 1)
    gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray , 1.2 , 5)
    for (x , y , w , h) in faces:
        cv2.rectangle(frame , (x,y) , (x+w , y+h) , (0 , 255 , 0) , 2)
        id , accuracy = recognizer.predict(gray[y:y+h , x:x+w])
        if accuracy<60:
            i_d = names[id]
            text_01 = f'{i_d}  Accuracy : {100-accuracy}'
        else:
            i_d = 'unknown'
            text_01 = f'{i_d}  Accuracy : {100-accuracy}'

        cv2.putText(frame , text_01 , (x+5 , y-5) , cv2.FONT_HERSHEY_COMPLEX , 1 , (0 , 255 , 0) , 2)
    
    cv2.imshow("frame" , frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()