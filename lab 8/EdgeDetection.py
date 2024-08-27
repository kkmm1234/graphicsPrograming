import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('ATU.jpg')

# Rows and Columns
nrows = 2
ncols = 1

#gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#original image
plt.subplot(nrows, ncols,1),plt.imshow(rgb, cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])

# GreyScale image
plt.subplot(nrows, ncols,2),plt.imshow(gray, cmap = 'gray')
plt.title('Graycale'), plt.xticks([]), plt.yticks([])

plt.show()
