import cv2 
import numpy as np
from matplotlib import pyplot as plt


from interactive_binarization import interactive_binarization

#-----------------------------------------------#
#------------- 1. Read image --------------#
#-----------------------------------------------#

img = cv2.imread(r'Images\test_assemblage_04_normal_zoom200_HDR.jpg') 
    
    
print(img .shape) # image rows, columns, and channels
## 1.2 Show the image
cv2.imshow("img", img)
cv2.waitKey(0) # touch 0 to close
cv2.destroyAllWindows() # close the window and de-allocate any associated memory usage

#----------------------------------------------------------#
#------------- 2. Image processing -----------------#
#----------------------------------------------------------#


## 2.1 Convert to gray scale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
## 2.2 Image blurring -> image convolution with a low-pass filter kernel 
# cv2.GaussianBlur: blurs an image using a Gaussian filter.
# specify the width and height of the kernel which should be positive and odd
# specify the standard deviation in the X and Y directions
img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)
## 2.3 Image thersholding
## Interactive thresholding
#[thmin, thmax] = interactive_binarization(img_blur) # press 0 to close
#print(thmin, thmax)

## Manual thresholding
thmin = 222
thmax = 255
ret, img_bw = cv2.threshold(img_blur, thmin, thmax, cv2.THRESH_BINARY)
## 2.4 Plot original and post-processed images
fig, axs = plt.subplots(2,2)
axs = axs.flatten()
plt.set_cmap('gray')
for ax in axs: 
 ax.set_axis_off()
original = axs[0].imshow( img )
grayim = axs[1].imshow( img_gray )
blurred = axs[2].imshow( img_blur )
binarize = axs[3].imshow( img_bw ) 
axs[0].set_title('Original image')
axs[1].set_title('Gray scale')
axs[2].set_title('Blurred')
axs[3].set_title('Binarized')
plt.savefig('Image processing.jpg') # save the figure
plt.show()


## 3.3 Find contours
# step 1: create a copy of the image
imgcopy = np.copy( cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR) ) 
# step 2: find contours
# cv2.RETR_LIS retrieves all of the contours without establishing any 
# hierarchical relationships
(contours,hierarchy) = cv2.findContours(img_bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
# step 3: draw all the contours 
cv2.drawContours(image=imgcopy, contours=contours, contourIdx=-1, \
 color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
cv2.imshow('All contours', imgcopy)
cv2.waitKey(0)
cv2.destroyAllWindows()

## 3.4 Select interesting contours (tracks)
# step 1: select larger contours
track =[]
for i in range(0,len(contours)):
    if len(contours[i])>100: 
        track.append(contours[i])
# step 2: draw selected contours
imgcopy1 = np.copy( cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR) ) 
cv2.drawContours(image=imgcopy1, contours=track, contourIdx=-1, \
 color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
cv2.imshow('Rails', imgcopy1)
cv2.waitKey(0)
cv2.destroyAllWindows()


#---------------------------------------------------#
#------------- 4. Fit contours ------------------#
#---------------------------------------------------#
## 4.1 Linear fit
def plot_contours(img,cnt):
 rows,cols = img.shape[:2]
 
 # (vx, vy) is a normalized vector collinear to the line and 
 # (x, y) is a point on the line
 # the equation of the line is (x,y) = (x0,y0) + t*(vx,vy)
 [vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01) # Fit the edge with a line
 
 # Now find two extreme points on the line to draw line
 lefty = int((-x*vy/vx) + y)
 righty = int(((cols-x)*vy/vx)+y)
 
 imgline = cv2.line(img,(cols-1,righty),(0,lefty),(0,0,255),2) # Plot the line
 cv2.imshow('FitLine', imgline)
 cv2.imwrite('Linear fit.jpg',imgline)
 return vx, vy
m = []
for i in range(0,len(track)):
    [vx,vy] = plot_contours(imgcopy, track[i])
    m.append(vy/vx) # line slope
cv2.waitKey(0)
cv2.destroyAllWindows()

for i in m:
    print(f"{i}")
    