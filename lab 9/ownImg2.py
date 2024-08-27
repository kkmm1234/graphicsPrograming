import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('building.jpg')
imgHarris = img.copy()
imgShiTomasi = img.copy()

# Rows and Columns
nrows = 2
ncols = 2

# Gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

## Detects Corners Using Harris
dst = cv2.cornerHarris(gray, 2, 3, 0.04)

## Threshold to obtain the corners
threshold = 0.01
for i in range(len(dst)):
    for j in range(len(dst[i])):
        if dst[i][j] > (threshold*dst.max()):
            cv2.circle(imgHarris,(j,i),3,(255),-1)

## Detects Corners Using Tomasi
corners = cv2.goodFeaturesToTrack(gray,50,0.01,10)

# Corner plotting
for i in corners:
    x,y = i.ravel()
    x, y = int(x), int(y)  # Convert coordinates to integer
    cv2.circle(imgShiTomasi,(x,y),3,(255),-1)

# Orb
orb = cv2.ORB_create()
kp = orb.detect(img,None)
kp1, des1 = orb.compute(img, kp)

imgOrb = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)


# GreyScale image
plt.subplot(nrows, ncols,1),plt.imshow(gray, cmap = 'gray')
plt.title('Graycale'), plt.xticks([]), plt.yticks([])

# Harris Corner Detection
plt.subplot(nrows, ncols,2),plt.imshow(cv2.cvtColor(imgHarris, cv2.COLOR_BGR2RGB), cmap = 'gray')
plt.title('Harris'), plt.xticks([]), plt.yticks([])

# Tomasi Corner Detection
plt.subplot(nrows, ncols,3),plt.imshow(cv2.cvtColor(imgShiTomasi, cv2.COLOR_BGR2RGB), cmap = 'gray')
plt.title('ShiTomasi'), plt.xticks([]), plt.yticks([])

# Orb Detection
plt.subplot(nrows, ncols,4),plt.imshow(cv2.cvtColor(imgOrb, cv2.COLOR_BGR2RGB), cmap = 'gray')
plt.title('Orb'), plt.xticks([]), plt.yticks([])


plt.show()