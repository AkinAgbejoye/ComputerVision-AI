import numpy
import face_recognition
import cv2
import os

import numpy as np

path = 'ImagesAttendance/'

images = []
classNames = []
myList = os.listdir(path)
print(myList)

for cl in myList:
    currentImg  = cv2.imread(f'{path}/{cl}')
    images.append(currentImg)
    classNames.append(os.path.splitext(cl)[0])

print(classNames)


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

codeListKnown = findEncodings(images)

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgs = cv2.resize(img,(0,0),None, 0.25,0.25)
    imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)

    facescurFrame = face_recognition.face_locations(imgs)
    encodescurFrame = face_recognition.face_encodings(imgs,facescurFrame)

    for encodeface, faceloc in zip(encodescurFrame,facescurFrame):
        matches = face_recognition.compare_faces(codeListKnown,encodeface)
        facedis = face_recognition.face_distance(codeListKnown,encodeface)
        print(facedis)
        matchIndex = np.argmin(facedis)

        if matches[matchIndex]:
            name  = classNames[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceloc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)

    cv2.imshow("webcam",img)
    cv2.waitKey(1)