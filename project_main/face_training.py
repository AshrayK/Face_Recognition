import os
import cv2 as cv
import numpy as np
print('Training started')

people=['Cristiano Ronaldo','Donald Trump','Selena Gomez']
dir=r'F:\python\save files\project_main\train'

haar_cascade=cv.CascadeClassifier('project_main\haar_face.xml')

feature=[]
labels=[]

def create_train():
    for person in people:
        path=os.path.join(dir,person)
        label=people.index(person)

        for img in os.listdir(path):
            img_path=os.path.join(path,img)

            img_array=cv.imread(img_path)
            gray_scale=cv.cvtColor(img_array,cv.COLOR_BGR2GRAY)

            face_rect=haar_cascade.detectMultiScale(gray_scale,scaleFactor=1.1,minNeighbors=4)

            for(x,y,w,h) in face_rect:
                face_roi=gray_scale[y:y+h,x:x+w]
                feature.append(face_roi)
                labels.append(label)

create_train()


feature=np.array(feature,dtype='object')
labels=np.array(labels)

face_recognizer=cv.face.LBPHFaceRecognizer_create()

#Training the Recognizer on the feature list and the labels list
face_recognizer.train(feature,labels)


#saving for reuse
face_recognizer.save('face_trained.yml')

np.save('Features.npy',feature)
np.save('labels.npy',labels)
print('Training ended')
















# print(f'Lenght of the features in the images={len(feature)}')
# print(f'Lenght of the labels in the images= {len(labels)}')



# people_train=[]
# for i in os.listdir(r'F:\python\save files\project_main\train'):
#     people_train.append(i)

# print(people_train)