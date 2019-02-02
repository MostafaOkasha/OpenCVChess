import cv2     # import opencv library
import picamera
from time import sleep
import numpy as np
from matplotlib import pyplot as plt

BLUE = [255,0,0]

# Beg. of code:
camera = picamera.PiCamera() #initialize the camera
camera.resolution = (2592, 1944)
camera.brightness = 70
camera.contrast = 70
sleep(1)
camera.capture('/home/pi/Pictures/temp.jpg')
imagename = '/home/pi/Pictures/temp.jpg'
grey_image = cv2.imread(imagename, 0)

#BGR pixel: px = grey_image[100,100]

# selecting a region: ball = img[280:340, 330:390] #AKA image ROI


cv2.namedWindow("output", cv2.WINDOW_AUTOSIZE)       
imS = cv2.resize(grey_image, (960, 540))
cv2.imshow("output", imS)
cv2.waitKey(0)
cv2.destroyAllWindows()

# cv2.imwrite('messigray.png',img) to write image

# Create a black image
img = np.zeros((512,512,3), np.uint8)

# Draw a diagonal blue line with thickness of 5 px
img = cv2.line(img,(0,0),(511,511),(255,0,0),5)


#  top-left corner and bottom-right corner
img = cv2.rectangle(img,(384,0),(510,128),(0,255,0),3)

# enter coordinates and radius circle
img = cv2.circle(img,(447,63), 63, (0,0,255), -1)

## putting text
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,'OpenCV',(10,500), font, 4,(255,255,255),2,cv2.LINE_AA)

constant= cv2.copyMakeBorder(img,10,10,10,10,cv2.BORDER_CONSTANT,value=BLUE)
plt.subplot(236),plt.imshow(constant,'gray'),plt.title('CONSTANT')
plt.show()

cv2.imshow("output", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

laplacian = cv2.Laplacian(grey_image,cv2.CV_64F)
plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.show()


edges = cv2.Canny(grey_image,100,200)
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()


filename = '/home/pi/Pictures/temp.jpg'
img = cv2.imread(filename)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.04)

#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[0,0,255]

imS = cv2.resize(img, (960, 540))
cv2.imshow("output", imS)

cv2.waitKey(0)
cv2.destroyAllWindows()

img = cv2.imread('simple.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(gray,20,0.1,6)
#(pic,number to find, quality, min distance between corners)
corners = np.int0(corners)

for i in corners:
    x,y = i.ravel()
    cv2.circle(img,(x,y),3,255,-1)

plt.imshow(img),plt.show()


'''
from matplotlib import pyplot as plt

img = cv2.imread('messi5.jpg',0)
plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()
'''
