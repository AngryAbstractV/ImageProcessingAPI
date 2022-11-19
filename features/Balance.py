import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt 

"""
Find keypoints in a photo
"""
im = cv.imread('coins.jpg')
gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
sift = cv.SIFT_create()
kp, des = sift.detectAndCompute(gray,None)
img = cv.drawKeypoints(gray,kp,img)
plt.imshow(img)
print(len(kp))
#des[0]

"""
Find circles in an image
"""
img = cv.imread('coins.jpg',0)
img = cv.medianBlur(img,5)
cimg = cv.cvtColor(img,cv.COLOR_GRAY2BGR)
min_dist = min(img.shape[0],img.shape[1])
circles = cv.HoughCircles(img,cv.HOUGH_GRADIENT,2.0,int(min_dist/2),
                            param1=200,param2=100,minRadius=10,maxRadius=int(min_dist/2))
circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
plt.figure()
plt.imshow(cimg)

"""
ToDo
Find how to see what keypoints are: Do they have just x and y or is there more?
If keypoint does not have direction can we check if description does?
Description has 128 bins can we find direction through a vector?
Check second paper more
Use OpenCV documentation feature matching and do it on match pair of features
"""
