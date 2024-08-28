import cv2
import numpy as np
from matplotlib import pyplot as plt
import random as rng

rng.seed(12345)

img = cv2.imread('building.jpg')
imgHarris = img.copy()
imgShiTomasi = img.copy()

img2 = cv2.imread('building2.jpg')
img2Contours = img2.copy()

# Rows and Columns
nrows = 2
ncols = 6

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

# Image Matching
gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

kp2, des2 = orb.detectAndCompute(img2,None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
# Draw first 10 matches.
img3 = cv2.drawMatches(img,kp1,img2,kp2,matches[:10],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Spliting RGB
M = np.asarray(img)

# Tomasi
## Detects Corners Using Tomasi
cornersBuilding = cv2.goodFeaturesToTrack(gray,50,0.01,10)

## corner plotting
for i in cornersBuilding:
    x,y = i.ravel()
    x, y = int(x), int(y)  # Convert coordinates to integer
    cv2.circle(imgShiTomasi,(x,y),3,(255),-1)

# Orb
orbBuilding = cv2.ORB_create()
kp = orbBuilding.detect(img,None)
kp, des = orbBuilding.compute(img, kp)

imgOrbBuilding = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)

# Contours
img2ContoursGray = cv2.cvtColor(img2Contours, cv2.COLOR_BGR2GRAY)
img2ContoursGray = cv2.blur(img2ContoursGray, (3, 3))

# Canny
canny_output = cv2.Canny(img2ContoursGray, 100, 200)

# Find contours
contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours
for i in range(len(contours)):
    color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
    cv2.drawContours(img2Contours, contours, i, color, 2, cv2.LINE_8, hierarchy, 0)

# HSV
# Convert the image from BGR to HSV
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Split the HSV image into its three channels
H, S, V = cv2.split(hsv_img)



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

# RGB image
plt.subplot(nrows, ncols,5)
plt.imshow(M[:, :, 0], cmap='Reds', vmin=0, vmax=255)
plt.title('Red Channel'), plt.xticks([]), plt.yticks([])

plt.subplot(nrows, ncols,6)
plt.imshow(M[:, :, 1], cmap='Greens', vmin=0, vmax=255)
plt.title("Green Channel"), plt.xticks([]), plt.yticks([])

plt.subplot(nrows, ncols,7)
plt.imshow(M[:, :, 2], cmap='Blues', vmin=0, vmax=255)
plt.title("Blue Channel"), plt.xticks([]), plt.yticks([])

# Contours
plt.subplot(nrows, ncols, 8), plt.imshow(cv2.cvtColor(img2Contours, cv2.COLOR_BGR2RGB))
plt.title('Contours'), plt.xticks([]), plt.yticks([])

# HSV
# Hue channel
plt.subplot(nrows, ncols, 9)
plt.imshow(H, cmap='gray')
plt.title('Hue Channel')
plt.axis('off')

# Saturation channel
plt.subplot(nrows, ncols, 10)
plt.imshow(S, cmap='gray')
plt.title('Saturation Channel')
plt.axis('off')

# Value channel
plt.subplot(nrows, ncols, 11)
plt.imshow(V, cmap='gray')
plt.title('Value Channel')
plt.axis('off')

# Image Matching
plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB), cmap = 'gray')
plt.title('Image Matching ATU'), plt.xticks([]), plt.yticks([])

plt.show()