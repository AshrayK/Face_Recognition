import numpy as np
import cv2 as cv

haar_cascade=cv.CascadeClassifier('F:\python\save files\project_main\haar_face.xml')

people=['Cristiano Ronaldo','Donald Trump','Selena Gomez']
# features = np.load('features.npy', allow_pickle=True)
# labels = np.load('labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')


  

choice=int(input ("Enter the image you want to detect (1-3): "))
if choice==1:
    img=cv.imread(r'F:\python\save files\project_main\validation\Cristiano Ronaldo\cr.jpg')
elif choice==2:
    img=cv.imread(r'F:\python\save files\project_main\validation\Donald Trump\DT.jpg')
elif choice==3:
    img=cv.imread(r'F:\python\save files\project_main\validation\Selena Gomez\sg.jpg')
else:
    print("Invalid input!")


gray_scale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Gray conversion of the image', gray_scale)

# Detecting the face in the provided image
faces_recognized = haar_cascade.detectMultiScale(gray_scale, 1.1, 9)

for (x,y,w,h) in faces_recognized:
    faces_roi = gray_scale[y:y+h,x:x+w]

    label,confidence= face_recognizer.predict(faces_roi)
    print(f'Person Detected = {people[label]}')

    cv.putText(img, str(people[label]), (x,y-10), cv.FONT_HERSHEY_COMPLEX, 1.0, (255,0,0), thickness=2)
    cv.rectangle(img, (x,y), (x+w,y+h), (255,0,0), thickness=2)

cv.imshow('Recognized face', img)

cv.waitKey(0)