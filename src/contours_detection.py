import cv2 as cv 
import numpy as np
import setup as stp
import random

from setup import RED, GREEN, BLUE
from utilis import generate_colors


#------------------------------------------------------------------------------------#
#------------------------------------- FUNCTION -------------------------------------#
#------------------------------------------------------------------------------------#
def select_countours(contours, len_min):

    selected_contours = []
    for cntrs in contours:
        if(len(cntrs) > len_min):
            selected_contours.append(cntrs)

    return selected_contours

def draw_regions(img, contours):

    img_cpy = np.copy(cv.cvtColor(img, cv.COLOR_GRAY2RGB))

    cnt_index = 0
    cntrs_colors = generate_colors(len(contours))
    for cnt in contours : 

        red = random.randint(140, 255)
        green = random.randint(0, 60)
        blue = random.randint(50, 155)

        cv.drawContours(image=img_cpy, contours=[cnt], contourIdx=-1, 
                        color=cntrs_colors[cnt_index], thickness=1, lineType=cv.LINE_AA)

        cnt_index += 1

    return img_cpy

#------------------------------------------------------------------------------------#
#------------------------------------- PATH -----------------------------------------#
#------------------------------------------------------------------------------------#
blur_path = stp.CNTRS_OUTPUT_PATH + 'blur.jpg'
bw_path = stp.CNTRS_OUTPUT_PATH + 'bw.jpg'

edges_path = stp.CNTRS_OUTPUT_PATH + 'edges.jpg'
green_edges_path = stp.CNTRS_OUTPUT_PATH + 'green_edges.jpg'

img_name = stp.BEFORE_FRET + stp.IMG_NAME

#------------------------------------------------------------------------------------#
#----------------------------------- VARIABLES --------------------------------------#
#------------------------------------------------------------------------------------#
CNTRS_LEN_MIN = 500 

#------------------------------------------------------------------------------------#
#--------------------------------------- MAIN ---------------------------------------#
#------------------------------------------------------------------------------------#
img = cv.imread(img_name, cv.IMREAD_GRAYSCALE)
img = cv.resize(src=img, dsize=None, fx=0.05, fy=0.05, interpolation=cv.INTER_AREA)

#------------------------- IMAGE PROCESSING -------------------------# 
img_blur = cv.GaussianBlur(img, (9, 9), 0)
(ret, img_bw) = cv.threshold(img_blur, stp.TH_MIN, stp.TH_MAX, cv.THRESH_BINARY + cv.THRESH_OTSU)

#-------------------------- EDGE DETECTION --------------------------#
(contours, hierarchy) = cv.findContours(img_bw, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
selected_contours = select_countours(contours, CNTRS_LEN_MIN) 

#------------------------------ SAVING ------------------------------#
img_filtered_edges = draw_regions(img, selected_contours)

cv.imwrite(green_edges_path, img_filtered_edges)




