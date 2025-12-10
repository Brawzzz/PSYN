import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import glob
import setup as stp


IMG_INDEX = 0

image_names = sorted(glob.glob(stp.BEFORE_FRET + '*.bmp'))

img = cv.imread(image_names[IMG_INDEX], cv.IMREAD_GRAYSCALE)
img = cv.resize(src=img, dsize=None, fx=0.1, fy=0.1, interpolation=cv.INTER_AREA)

f_img = np.fft.fft2(img)
f_shift = np.fft.fftshift(f_img)

angle = np.angle(f_shift)
mag = np.abs(f_shift)

mag_log = 20 * np.log(1 + mag)

img_module_norm = cv.normalize(mag_log, None, 0, 255, cv.NORM_MINMAX)
img_module_norm = np.uint8(img_module_norm)

# cv.imshow("Image originale", img)
# cv.waitKey(0)

# cv.imshow("Phase de l'image", angle)
# cv.waitKey(0) 

# cv.imshow("Module de l'image", img_module_norm)
# cv.waitKey(0) 

# cv.destroyAllWindows()