import cv2     # import opencv library
import picamera
from time import sleep
from picamera.array import PiRGBArray
from picamera import PiCamera
import numpy as np
import glob
import string
import sys
import math

#take an image
#find the 4 corners
# warp it.
# increase brightness.
#create a line between the 4 corners.
# divide by 8 to create 8 new points and thus create new squares.
#After image is warped create a perfect fit image and read what is inside
# each square.
#

# Function to find state of the board
##def findstateboard(stateimage):
##    # Arrays to store object points and image points from all the images.
##    objpoints = [] # 3d points in real world space
##    imgpoints = [] # 2d points in image plane.
##    imagename = '/home/pi/Pictures/' + stateimage
##    grey_image = cv2.imread(imagename, 0)
##    found, corners = cv2.findchessboardcorners(grey_image, (6,9))
##    corners = np.int0(corners)
##
##    for i in corners:
##        x,y = i.ravel()
##        cv2.circle(grey_image,(x,y),3,(0,0,255),-1)
##
##    cv2.imshow('corners',grey_image)
##
##    #return matrix
##
##def statechange(stateimage1, stateimage2):
##    print()
##    #do the comparision
##    #return matrix
    
# Beg. of code:
camera = PiCamera() #initialize the camera
camera.resolution = (2592, 1944)
camera.brightness = 70
camera.contrast = 70

# Take an Image as state initial.
camera.start_preview(alpha=200)
sleep(2)
camera.capture('/home/pi/Pictures/initialstate.jpg') #/home/pi/Pictures
camera.stop_preview()

# Take an image as state initial + 1 for comparision.
camera.start_preview(alpha=200)
sleep(2)
camera.capture('/home/pi/Pictures/newstate.jpg')
camera.stop_preview()
imagename = '/home/pi/Pictures/newstate.jpg'
grey_image = cv2.imread(imagename, 0)

##found, corners = cv2.findChessboardCorners(grey_image, (6,9))
##img = cv2.drawChessboardCorners(grey_image, (6,9), corners, found)
##print(found)
##print(corners)
cv2.imshow('img',grey_image)
cv2.waitKey(500)
sleep(2)
cv2.destroyAllWindows()

# Run function to compare to old state
##findstateboard('newstate.jpg')

#rawCapture = PiRGBArray(camera)
#time.sleep(0.1)


#camera.start_preview()


# Take an image as new state of board. This will be used for comparision
# with old state.







##"""
##Corner Finding
##"""
### termination criteria 
##criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
##
### Prepare object points, like (0,0,0), (1,0,0), ....,(6,5,0)
##objp = np.zeros((5*5,3), np.float32)
##objp[:,:2] = np.mgrid[0:5,0:5].T.reshape(-1,2)
##
### Arrays to store object points and image points from all images
##objpoints = []
##imgpoints = []
##
##counting = 0
##
### Import Images
##images = glob.glob('dir/sub dir/Images/*')
##
##for fname in images:
##
##    img = cv2.imread(fname)     # Read images
##    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # Convert to grayscale
##
##
##
##    # Find the chess board corners
##    ret, corners = cv2.findChessboardCorners(gray, (5,5), None)
##
##    # if found, add object points, image points (after refining them)
##    if ret == True:
##        objpoints.append(objp)
##
##        cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
##        imgpoints.append(corners)
##
##        #Draw and display corners
##        cv2.drawChessboardCorners(img, (5,5), corners, ret)
##        counting += 1
##
##        print (counting) + ' Viable Image(s)'
##
##        cv2.imshow('img', img)
##        cv2.waitKey(500)
##
##cv2.destroyAllWindows()        
##
##
### Calibrate Camera    
##ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
##


##
##
##### initialize the camera and grab a reference to the raw camera capture
####camera = PiCamera()
####rawCapture = PiRGBArray(camera)
#### 
##### allow the camera to warmup
####time.sleep(0.1)
#### 
##### grab an image from the camera
####camera.capture(rawCapture, format="bgr")
####image = rawCapture.array
#### 
##### display the image on screen and wait for a keypress
####cv2.imshow("Image", image)
####cv2.waitKey(0)
####
##
### initialize the camera and grab a reference to the raw camera capture
##camera = PiCamera()
##camera.resolution = (640, 480)
##camera.framerate = 32
##rawCapture = PiRGBArray(camera, size=(640, 480))
## 
### allow the camera to warmup
##time.sleep(0.1)
## 
### capture frames from the camera
##for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
##	# grab the raw NumPy array representing the image, then initialize the timestamp
##	# and occupied/unoccupied text
##	image = frame.array
## 
##	# show the frame
##	cv2.imshow("Frame", image)
##	key = cv2.waitKey(1) & 0xFF
## 
##	# clear the stream in preparation for the next frame
##	rawCapture.truncate(0)
## 
##	# if the `q` key was pressed, break from the loop
##	if key == ord("q"):
##		break
##
##
####gray = cv2.imread('lines.jpg')
####edges = cv2.Canny(gray,50,150,apertureSize = 3)
####cv2.imwrite('edges-50-150.jpg',edges)
####minLineLength=100
####lines = cv2.HoughLinesP(image=edges,rho=1,theta=np.pi/180, threshold=100,lines=np.array([]), minLineLength=minLineLength,maxLineGap=80)
####
####a,b,c = lines.shape
####for i in range(a):
####    cv2.line(gray, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 3, cv2.LINE_AA)
####    cv2.imwrite('houghlines5.jpg',gray)
##
#####reading the image
####img=cv2.imread('messi.png',0)#read the image as grayscale image
#####display in screen
####cv2.imshow('messi',img)
####k=cv2.waitKey(0) & 0xFF
####
####if k==27:
####    cv2.destroyAllWindows()#close all display windows
####    
####elif k==ord('s'): #wait for 's' to be pressed
#####save the image into new file
####    cv2.imwrite('messigray.png',img)
####    cv2.destroyAllWindows()
