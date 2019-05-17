import numpy as np
import cv2
from itertools import count

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Load the overlay image: lasers.png
imglasers = cv2.imread('laser-eye-meme.png')

#Check if the files opened
if imglasers is None:
    exit("Could not open the image")
if face_cascade.empty():
    exit("Missing: haarcascade_frontalface_default.xml")
if eye_cascade.empty():
    exit("Missing: haarcascade_eye.xml")


# Create the mask for the lasers
imglasersGray = cv2.cvtColor(imglasers, cv2.COLOR_BGR2GRAY)
ret, orig_mask = cv2.threshold(imglasersGray, 10, 255, cv2.THRESH_BINARY)

#orig_mask = imglasers[:,:,3]

# Create the inverted mask for the lasers
orig_mask_inv = cv2.bitwise_not(orig_mask)

# Convert lasers image to BGR
# and save the original image size (used later when re-sizing the image)
imglasers = imglasers[:, :, 0:3]
origlasersHeight, origlasersWidth = imglasers.shape[:2]

#cv2.imshow('Video', imglasers)
#cv2.waitKey()


video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    exit('The Camera is not opened')
ret, frame = video_capture.read()

#counter = count(1)

while True:
    #print("Iteration %d" % next(counter))
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        # cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        #cv2.imshow('Video', roi_gray)
        #cv2.waitKey()

        #print 'X:%i, Y:%i, W:%i, H:%i' % (x, y, w, h)
        # for (ex, ey, ew, eh) in eyes[:2]:
        #     cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

        for (ex, ey, ew, eh) in eyes[:2]:
            lasersWidth = 3*ew
            lasersHeight = lasersWidth * origlasersHeight / origlasersWidth

            # Center the lasers on the bottom of the nose
            x1 = int(ex - (lasersWidth/4))
            x2 = int(ex + ew + (lasersWidth/4))
            y1 = int(ey + eh - (lasersHeight/2))
            y2 = int(ey + eh + (lasersHeight/2))

            # Check for clipping
            if x1 < 0:
                x1 = 0
            if y1 < 0:
                y1 = 0
            if x2 > w:
                x2 = w
            if y2 > h:
                y2 = h

            # Re-calculate the width and height of the lasers imagex   
            lasersWidth = x2 - x1
            lasersHeight = y2 - y1

            # Re-size the original image and the masks to the lasers sizes
            # calcualted above
            lasers = cv2.resize(
                imglasers, (lasersWidth, lasersHeight), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(
                orig_mask, (lasersWidth, lasersHeight), interpolation=cv2.INTER_AREA)
            mask_inv = cv2.resize(
                orig_mask_inv, (lasersWidth, lasersHeight), interpolation=cv2.INTER_AREA)

            # take ROI for lasers from background equal to size of lasers image
            roi = roi_color[y1:y2, x1:x2]

            # roi_bg contains the original image only where the lasers is not
            # in the region that is the size of the lasers.
            roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

            # roi_fg contains the image of the lasers only where the lasers is
            roi_fg = cv2.bitwise_and(lasers, lasers, mask=mask)

            # join the roi_bg and roi_fg
            dst = cv2.add(roi_bg, roi_fg)

            # place the joined image, saved to dst back over the original image
            roi_color[y1:y2, x1:x2] = dst
        break
    #Display the resulting frame
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
