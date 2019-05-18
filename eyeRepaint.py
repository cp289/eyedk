import numpy as np
import cv2
from itertools import count

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Load the overlay image: laser.png
imglaser = cv2.imread('laser-eye-meme.png')

#Check if the files opened
if imglaser is None:
    exit("Could not open the image")
if face_cascade.empty():
    exit("Missing: haarcascade_frontalface_default.xml")
if eye_cascade.empty():
    exit("Missing: haarcascade_eye.xml")


# Create the mask for the laser
imglaserGray = cv2.cvtColor(imglaser, cv2.COLOR_BGR2GRAY)
ret, orig_mask = cv2.threshold(imglaserGray, 10, 255, cv2.THRESH_BINARY)

# Create the inverted mask for the laser
orig_mask_inv = cv2.bitwise_not(orig_mask)

# Convert laser image to BGR
# and save the original image size (used later when re-sizing the image)
imglaser = imglaser[:, :, 0:3]
origlaserHeight, origlaserWidth = imglaser.shape[:2]


video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    exit('The Camera is not opened')
ret, frame = video_capture.read()


while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes[:4]:
            laserWidth = 3*ew
            laserHeight = laserWidth * origlaserHeight / origlaserWidth

            # Center the laser on the bottom of the nose
            x1 = int(ex - (laserWidth/4))
            x2 = int(ex + ew + (laserWidth/4))
            y1 = int(ey + eh - (laserHeight/2))
            y2 = int(ey + eh + (laserHeight/2))

            # Check for clipping
            if x1 < 0:
                x1 = 0
            if y1 < 0:
                y1 = 0
            if x2 > w:
                x2 = w
            if y2 > h:
                y2 = h

            # Re-calculate the width and height of the laser imagex   
            laserWidth = x2 - x1
            laserHeight = y2 - y1

            # Re-size the original image and the masks to the laser sizes
            # calcualted above
            laser = cv2.resize(
                imglaser, (laserWidth, laserHeight), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(
                orig_mask, (laserWidth, laserHeight), interpolation=cv2.INTER_AREA)
            mask_inv = cv2.resize(
                orig_mask_inv, (laserWidth, laserHeight), interpolation=cv2.INTER_AREA)

            # take ROI for laser from background equal to size of laser image
            roi = roi_color[y1:y2, x1:x2]

            # roi_bg contains the original image only where the laser is not
            # in the region that is the size of the laser.
            roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

            # roi_fg contains the image of the laser only where the laser is
            roi_fg = cv2.bitwise_and(laser, laser, mask=mask)

            # join the roi_bg and roi_fg
            dst = cv2.add(roi_bg, roi_fg)

            # place the joined image, saved to dst back over the original image
            roi_color[y1:y2, x1:x2] = dst
        
    #Display the resulting frame
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
