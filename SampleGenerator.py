import cv2

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("//Users//krishilparikh//CODING//FaceRecog//haarcascade_frontalface_default copy.xml")
i_d = input("Enter you ID : ")
count = 0

while True:
    _ , frame = cap.read()
    gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray , 1.1 , 3)

    for (x , y , w , h) in faces:
        cv2.rectangle(frame , (x , y) , (x+w , y+h) , (0 , 255 ,0) , 2)
        img_name = f"//Users//krishilparikh//CODING//FaceRecog//samples//{i_d}_{count}.jpg"
        face = gray[y:y+h , x:x+w]

        cv2.imwrite(img_name , face)
        count = count+1
        print(count)

    cv2.imshow("Frame" , frame)

    if cv2.waitKey(1) == ord('q'):
        break
    elif count>1000:
        break

cap.release()
cv2.destroyAllWindows()