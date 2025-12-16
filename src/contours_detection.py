import cv2 as cv 
import numpy as np
import setup as stp
import random

from setup import RED, GREEN, BLUE
from utilis import generate_colors, f_pass


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

        cv.drawContours(image=img_cpy, contours=[cnt], contourIdx=-1, 
                        color=cntrs_colors[cnt_index], thickness=1, lineType=cv.LINE_AA)
        cnt_index += 1

    return img_cpy

def interactive_th(img_blur, f_pass=f_pass):
    
    cv.namedWindow('Thresh settings')
    cv.createTrackbar('Thresh', 'Thresh settings', 100, 255, f_pass)

    while True:

        thresh_val = cv.getTrackbarPos('Thresh', 'Thresh settings')
        (ret, img_bw )= cv.threshold(img_blur, thresh_val, stp.TH_MAX, cv.THRESH_TOZERO)
        cv.imshow('Thresh settings', img_bw)
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cv.destroyAllWindows()
    print(f"Selecterd alue for thresh : {thresh_val}")

    return(thresh_val, img_bw)

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
CNTRS_LEN_MIN = 1000 

#------------------------------------------------------------------------------------#
#--------------------------------------- MAIN ---------------------------------------#
#------------------------------------------------------------------------------------#
img = cv.imread(img_name, cv.IMREAD_GRAYSCALE)
img = cv.resize(src=img, dsize=None, fx=0.05, fy=0.05, interpolation=cv.INTER_AREA)

#------------------------- IMAGE PROCESSING -------------------------# 
img_blur = cv.medianBlur(img, 5) # cv.GaussianBlur(img, (3, 3), sigmaX = 0)
cv.imwrite(blur_path, img_blur)

(th_val, img_bw) = interactive_th(img_blur)
# (ret, img_bw) = cv.threshold(img_blur, stp.TH_MIN, stp.TH_MAX, cv.THRESH_TOZERO)

cv.imwrite(bw_path, img_bw)

#-------------------------- EDGE DETECTION --------------------------#
(contours, hierarchy) = cv.findContours(img_bw, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
selected_contours = select_countours(contours, CNTRS_LEN_MIN) 

#------------------------------ SAVING ------------------------------#
img_filtered_edges = draw_regions(img, selected_contours)

cv.imwrite(green_edges_path, img_filtered_edges)




