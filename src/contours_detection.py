import cv2 as cv 
import numpy as np
import setup as stp

from setup import RED, GREEN, BLUE

blur_path = stp.CNTRS_OUTPUT_PATH + 'blur.jpg'
bw_path = stp.CNTRS_OUTPUT_PATH + 'bw.jpg'

edges_path = stp.CNTRS_OUTPUT_PATH + 'edges.jpg'
green_edges_path = stp.CNTRS_OUTPUT_PATH + 'green_edges.jpg'

img_name = stp.BEFORE_FRET + stp.IMG_NAME

img = cv.imread(img_name, cv.IMREAD_GRAYSCALE)
img = cv.resize(src=img, dsize=None, fx=0.05, fy=0.05, interpolation=cv.INTER_AREA)

#-------------------------- IMAGE PROCESSING --------------------------# 
img_blur = cv.GaussianBlur(img, (9, 9), 0)
cv.imwrite(blur_path, img_blur)

(ret, img_bw) = cv.threshold(img_blur, stp.TH_MIN, stp.TH_MAX, cv.THRESH_BINARY + cv.THRESH_OTSU)
cv.imwrite(bw_path, img_bw)

#-------------------------- EDGE DETECTION --------------------------#
img_edges = cv.Canny(image=img_bw, threshold1=100, threshold2=255) 

img_rgb_edges = cv.cvtColor(img_edges, cv.COLOR_GRAY2RGB) 
img_rgb_edges *= np.array((0, 1, 0), np.uint8)
img_rgb = cv.cvtColor(img, cv.COLOR_GRAY2RGB)

img_green_edges = cv.addWeighted(img_rgb, 0.25, img_rgb_edges, 0.75, 0)

cv.imwrite(edges_path, img_edges)
cv.imwrite(green_edges_path, img_green_edges)



