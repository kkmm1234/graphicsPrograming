import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('ATU1.jpg')
imgHarris = img.copy()

# Rows and Columns
nrows = 1
ncols = 1

# Gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# GreyScale image
plt.subplot(nrows, ncols,1),plt.imshow(gray, cmap = 'gray')
plt.title('Graycale'), plt.xticks([]), plt.yticks([])

## Detects Corners Using Harris
dst = cv2.cornerHarris(gray, 2, 3, 0.04)

plt.show()