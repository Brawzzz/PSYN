import colorsys
import numpy as np
import cv2 as cv
import setup as stp

#----------------
def is_pow_2(n):
    return((n and (n-1)) == 0)

#----------------
def generate_colors(n):

    colors = []

    for i in range(n):

        hue = i / n
        saturation = 1.0
        lightness = 1.0
        
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, lightness)

        colors.append((int(b*255), int(g*255), int(r*255)))
        
    return colors

#----------------
def angle_color(angle : float):

    normalize_angle = (angle % 360)
    hue = int(normalize_angle / 2)
    hsv = np.uint8([[[hue, 255, 255]]])

    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)[0][0]

    return (int(bgr[2]), int(bgr[1]), int(bgr[0]))

#----------------
def img_empty(img):

    if img is None:
        return True
    
    if img.size == 0:
        return True
        
    return False

#----------------
def f_pass(x):
    pass