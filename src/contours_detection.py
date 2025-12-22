import os
import cv2 as cv 
import numpy as np
import setup as stp
import Sample as spl

from setup import RED, GREEN, BLUE
from utilis import generate_colors, angle_color, f_pass


#------------------------------------------------------------------------------------#
#------------------------------------ FUNCTIONS -------------------------------------#
#------------------------------------------------------------------------------------#
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
    print(f"Selecterd value for thresh : {thresh_val}")

    return(thresh_val, img_bw)

#----------------
def select_countours(contours : np.ndarray, len_min : int) -> list:

    selected_contours = []
    for cntrs in contours:
        if(len(cntrs) > len_min):
            selected_contours.append(cntrs)

    return selected_contours

#----------------
def compute_orientation(cnt : np.ndarray) -> tuple[float, tuple[int, int]]:

    f_cnt = cnt.reshape(-1, 2).astype(np.float64)

    (mean, eigen_vectors) = cv.PCACompute(f_cnt, mean=None)

    center = (int(mean[0,0]), int(mean[0,1]))
    main_vec = eigen_vectors[0]

    angle_rad = np.arctan2(main_vec[1], main_vec[0])
    angle = np.degrees(angle_rad)

    return(angle, center)

#----------------
def draw_regions(img : np.ndarray, contours : np.ndarray) -> np.ndarray:

    img_cpy = np.copy(cv.cvtColor(img, cv.COLOR_GRAY2RGB))

    for cnt in contours : 

        (angle, center) = compute_orientation(cnt)
        color = angle_color(angle)

        cv.drawContours(image=img_cpy, contours=[cnt], contourIdx=-1, 
                        color=color, thickness=1, lineType=cv.LINE_AA)

    return img_cpy

#----------------
def detect_regions(sample : spl.Sample) -> list[np.ndarray]: 

    for sub_index in range(len(sample.split_list)):

        img_blur = cv.GaussianBlur(sample.split_list[sub_index], (9, 9), sigmaX = 0)
        (ret, img_bw) = cv.threshold(img_blur, stp.TH_MIN, stp.TH_MAX, cv.THRESH_TOZERO + cv.THRESH_OTSU)

        (contours, hierarchy) = cv.findContours(img_bw, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        selected_contours = select_countours(contours, stp.CNTRS_LEN_MIN) 

        sample.split_list[sub_index] = draw_regions(sample.split_list[sub_index], selected_contours)

    sample.join()


