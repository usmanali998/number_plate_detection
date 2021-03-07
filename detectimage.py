import cv2
import numpy as np



faceCascade = cv2.CascadeClassifier('haarcascades/haarcascade_russian_plate_number.xml')

img = cv2.imread('car.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

faces = faceCascade.detectMultiScale(img,scaleFactor=1.2,
    minNeighbors = 5, minSize=(25,25))

for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)
    cv2.putText(img, 'Num Plate', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (36, 255, 12), 2)

    plate = img[y: y+h, x:x+w]
    # plate = cv2.blur(plate,ksize=(20,20))
    # print(plate)
    # put the blurred plate into the original image
    img[y: y+h, x:x+w] = plate

cv2.imshow('plates',img)
if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()