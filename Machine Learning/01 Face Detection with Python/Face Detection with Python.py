# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 06:29:22 2021

@author: Sabeer PH

@Project : Face Detection

**Face Detection Vs Face Recognition**

Face detection is the process of detecting faces, from an image or a video does not matter. 
The program does nothing more than finding the faces. 
But on the other hand in the task of face recognition, 
the program finds faces and can also tell which face belongs to which.
 So it’s more informative than just detecting them. There is more programming, 
 in other words, more training in the process.


"""
import cv2

'''OpenCV library in python is blessed with many pre-trained classifiers for face, eyes, smile, etc. 
These XML files are stored in a folder. We will use the face detection model. 
You can download the pre-trained face detection model from 
https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'''

face_cascade = cv2.CascadeClassifier('face_detector.xml')


'''The next step is to choose an image on which you want to test your code. 
Make sure there is at least one face in the image so that the face detection program can 
find at least one face.'''

img = cv2.imread('Arbaaz.jpg')

print(img)
#Detect Faces
#You will be amazed at how short the face detection code is. 
#Thanks to the people who contribute to OpenCV. Here is the code that detects faces in an image:

faces = face_cascade.detectMultiScale(img, 1.1, 4)


#Now the last step is to draw rectangles around the detected faces, which can be easily done with the following code:

for (x, y, w, h) in faces: 
  cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
cv2.imwrite("face_detected.png", img) 
print('Successfully saved')
