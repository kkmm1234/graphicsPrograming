import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('ATU1.jpg')
imgHarris = img.copy()

# Rows and Columns
nrows = 2
ncols = 1

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


# GreyScale image
plt.subplot(nrows, ncols,1),plt.imshow(gray, cmap = 'gray')
plt.title('Graycale'), plt.xticks([]), plt.yticks([])

# Harris Corner Detection
plt.subplot(nrows, ncols,2),plt.imshow(cv2.cvtColor(imgHarris, cv2.COLOR_BGR2RGB), cmap = 'gray')
plt.title('Harris'), plt.xticks([]), plt.yticks([])


plt.show()