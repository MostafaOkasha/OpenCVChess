import cv2     # import opencv library
import picamera
from time import sleep
import numpy as np
from matplotlib import pyplot as plt
import glob
from picamera.array import PiRGBArray
import sys
import random

BLUE = [255,0,0]
colours=['yellow','orange','red','green','black','white']
uppers=[[20,100,100],[5,100,100],[0,100,100],[180,255,50],[255,255,255]]
lowers=[[30,255,255],[15,255,255],[6,255,255],[0,0,0],[200,200,200]]

# Beg. of code:
camera = picamera.PiCamera() #initialize the camera
##camera.resolution = (2592, 1944)
##camera.brightness = 70
##camera.contrast = 70
sleep(0.1)
# camera.capture('/home/pi/Pictures/temp.jpg')
# allow the camera to warmup
sleep(0.1)
imagename = '/home/pi/Pictures/temp.jpg'
###########################################################################

#read captured image into img
img = cv2.imread(imagename)

#grey_image = cv2.imread(imagename, 0) # Converts RGB image to Gray scale

grey_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # converts BGR to gray

kernel = np.ones((5,5),np.uint8)
#erosion = cv2.erode(grey_image,kernel,iterations=1)
opening = cv2.morphologyEx(grey_image, cv2.MORPH_OPEN, kernel)
#use cv2.MORPH_CLOSE for dilation followed by erosion.


#perform morphological-closing aka opening
#dilation
#erosion
#^^This is to remove any holes in the image. removes noisy dots.

#set a threshold efficiency to 80%. use hierarchical clustering to merge connnected
#components. 


#threshold(gs_img, thresholdvalue,[0,255] - 0 for black 255 for white. brightness
#threshold valuye shjould be 127. pixel 255.
ret, thresh_img = cv2.threshold(opening,128,255,cv2.THRESH_BINARY)

#find contours in the image in order to calculate areas and detect squares
#im2,contours,hierarchy=cv2.findContours(separated,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
im2,contours,h = cv2.findContours(thresh_img,1,2)

#approximating each contour to check number of elements in all the shapes.
#square should have 4 contours

for cnt in contours:
    approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
    cont_area = cv2.contourArea(cnt)
    #print (len(approx))
    if cont_area >=0:
        if len(approx)==4:
            #print ("square")
            cv2.drawContours(img,[cnt],0,(0,0,255),-1)

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('img',opening)
cv2.waitKey(0)
cv2.destroyAllWindows()







'''
corners = cv2.goodFeaturesToTrack(gray,30,0.05,50)
corners = np.int0(corners)

for i in corners:
    x,y = i.ravel()
    cv2.circle(gray,(x,y),10,165,10)

##plt.imshow(gray),plt.show()
plt.subplot(),plt.imshow(gray,'gray'),plt.title('CONSTANT')
plt.show()

'''

