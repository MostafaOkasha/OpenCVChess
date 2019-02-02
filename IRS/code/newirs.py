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
import imutils
import argparse

# Beg. of code:
camera = PiCamera() #initialize the camera
camera.resolution = (2592, 1944)
camera.brightness = 70
camera.contrast = 70
sleep(1)
camera.capture('/home/pi/Pictures/temp.jpg')
imagename = '/home/pi/Pictures/temp.jpg'
grey_image = cv2.imread(imagename, 0)

##cv2.namedWindow("output", cv2.WINDOW_AUTOSIZE)       
##imS = cv2.resize(grey_image, (960, 540))
##cv2.imshow("output", imS)
##cv2.waitKey(0)


class ShapeDetector:
    def __init__(self):
	    pass
 
    def detect(self, c):
        # initialize the shape name and approximate the contour
        shape = "unidentified"
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)

    if len(approx) == 4:
        # compute the bounding box of the contour and use the
        # bounding box to compute the aspect ratio
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)

        # a square will have an aspect ratio that is approximately
        # equal to one, otherwise, the shape is a rectangle
        shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
        
image = cv2.imread(imagename, 0)
resized = imutils.resize(image, width=300)
ratio = image.shape[0] / float(resized.shape[0])
 
# convert the resized image to grayscale, blur it slightly,
# and threshold it
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
 
# find contours in the thresholded image and initialize the
# shape detector
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
sd = ShapeDetector()

# loop over the contours
for c in cnts:
	# compute the center of the contour, then detect the name of the
	# shape using only the contour
	M = cv2.moments(c)
	cX = int((M["m10"] / M["m00"]) * ratio)
	cY = int((M["m01"] / M["m00"]) * ratio)
	shape = sd.detect(c)
 
	# multiply the contour (x, y)-coordinates by the resize ratio,
	# then draw the contours and the name of the shape on the image
	c = c.astype("float")
	c *= ratio
	c = c.astype("int")
	cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
	cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
		0.5, (255, 255, 255), 2)
 
	# show the output image
	cv2.imshow("Image", image)
	cv2.waitKey(0)
