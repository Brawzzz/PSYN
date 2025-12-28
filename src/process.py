#----------------------------------------------------------------------------------------------#
#------------------------------------------- IMPORT -------------------------------------------#
#----------------------------------------------------------------------------------------------#
import cv2 as cv 
import numpy as np

import Sample
import Fiber

import setup as stp

#----------------------------------------------------------------------------------------------#
#----------------------------------------- FUNCTIONS ------------------------------------------#
#----------------------------------------------------------------------------------------------#
def detect_fibers(sample : Sample.Sample, n_split_index : int = 0) -> list[Fiber.Fiber] : 

    #---------------
    print("\nFinding contours ...", end="\r")

    thresh_img_path = sample.thresh_path + stp.THRESH + str(n_split_index) + stp.OUTPUT_EXTENSION
    thresh_img = cv.imread(thresh_img_path, cv.IMREAD_GRAYSCALE)

    (contours, _) = cv.findContours(thresh_img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    fibers = []
    for cnt in contours:
        
        fib = Fiber.Fiber(cnt)
        fibers.append(fib)

    fibers = Fiber.select_fiber(fibers)

    return fibers