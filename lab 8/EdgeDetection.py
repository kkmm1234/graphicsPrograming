import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('ATU.jpg')

# Rows and Columns
nrows = 2
ncols = 3

#gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Blur
blur3x3 = cv2.GaussianBlur(gray,(3, 3),0)
blur13x13 = cv2.GaussianBlur(gray,(13, 13),0)

# Sobel
sobelHorizontal = cv2.Sobel(blur3x3,cv2.CV_64F,1,0,ksize=5) # x dirextion
sobelVertical = cv2.Sobel(blur3x3,cv2.CV_64F,0,1,ksize=5) # y direction

#Atu image plot
#original image
plt.subplot(nrows, ncols,1),plt.imshow(rgb, cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])

# GreyScale image
plt.subplot(nrows, ncols,2),plt.imshow(gray, cmap = 'gray')
plt.title('Graycale'), plt.xticks([]), plt.yticks([])

# Blur 3x3
plt.subplot(nrows, ncols,3),plt.imshow(blur3x3, cmap = 'gray')
plt.title('Blur 3x3'), plt.xticks([]), plt.yticks([])

# Blur 13x13
plt.subplot(nrows, ncols,4),plt.imshow(blur13x13, cmap = 'gray')
plt.title('Blur 13x13'), plt.xticks([]), plt.yticks([])

#sobelHorizontal
plt.subplot(nrows, ncols,5),plt.imshow(sobelHorizontal, cmap = 'gray')
plt.title('Sobel Horizontal'), plt.xticks([]), plt.yticks([])

#sobelVertical 
plt.subplot(nrows, ncols,6),plt.imshow(sobelVertical, cmap = 'gray')
plt.title('Sobel Vertical'), plt.xticks([]), plt.yticks([])

plt.show()
