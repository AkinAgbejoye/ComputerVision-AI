import numpy
import face_recognition
import cv2
ellon = face_recognition.load_image_file('Elon_Musk.jpg')
ellon = cv2.cvtColor(ellon, cv2.COLOR_BGR2RGB)


ellonTest = face_recognition.load_image_file('bill_gate.jpeg')
ellonTest = cv2.cvtColor(ellonTest, cv2.COLOR_BGR2RGB)

faceloc = face_recognition.face_locations(ellon)[0]
faceencoding = face_recognition.face_encodings(ellon)[0]
cv2.rectangle(ellon,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255),2)


facelocTest = face_recognition.face_locations(ellonTest)[0]
faceencodingTest = face_recognition.face_encodings(ellonTest)[0]
cv2.rectangle(ellonTest,(facelocTest[3],facelocTest[0]),(facelocTest[1],facelocTest[2]),(255,0,255),2)

results  = face_recognition.compare_faces([faceencoding],faceencodingTest)
facedist = face_recognition.face_distance([faceencoding],faceencodingTest)
print(results,facedist)
cv2.putText(ellonTest,f'{results} ,{round(facedist[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow("Elon Musk",ellon)
cv2.imshow("Elon Test",ellonTest)
cv2.waitKey(0)