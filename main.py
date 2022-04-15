## The following is written for ipynb - Google Colab
## To run it on your personal computer, change the cv2_inshow function to a compatible function

import cv2
import numpy as np
from google.colab.patches import cv2_imshow
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/content/shape_predictor_68_face_landmarks.dat")
img = cv2.imread("/content/Michelle_Obama_2013_official_portrait.jpg")
cv2_imshow(img)
gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
faces = detector(gray)

polypoints = []

for face in faces:
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()

    landmarks = predictor(image=gray, box=face)

    for n in range(0, 68):

        x = landmarks.part(n).x
        y = landmarks.part(n).y

        polypoints.append((x,y))

points = np.array(polypoints, np.int32)
points = points.reshape((-1, 1, 2))

imgB = cv2.imread("/content/44ah0f.jpg")

img = cv2.polylines(img, points, True, (0,0,255), 3)

imgB = cv2.polylines(imgB, points, True, (0,0,255), 3)


cv2_imshow(img)

cv2_imshow(imgB)
