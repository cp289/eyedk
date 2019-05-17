import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cv2.namedWindow("preview")
lasers = cv2.imread('laser-eye-meme.png')
vc = cv2.VideoCapture(0)

# Check if camera opened successfully
if (vc.isOpened() == False):
  print("Unable to read camera feed")

while True:
    # Capture frame-by-frame
    ret, frame = vc.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        for(ex, ey, ew, eh) in eyes:
            lasers = cv2.resize(lasers, (ew, eh))
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 2)
            # frame = cv2.addWeighted(roi_color,0.9,lasers,0.1,0)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
vc.release()
cv2.destroyAllWindows()
