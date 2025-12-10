# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 21:56:37 2020

@author: zakaria
updated 16/12/2020 @lofeudos
"""

import cv2

def interactive_binarization(img):

    def trackChaned(x):
        pass
    
    cv2.namedWindow('Select a threshold')

    cv2.createTrackbar("Max","Select a threshold",0,255,trackChaned)
    cv2.createTrackbar("Min","Select a threshold",0,255,trackChaned) 
        
    while(True):
        thmax = cv2.getTrackbarPos("Max", "Select a threshold")
        thmin = cv2.getTrackbarPos("Min", "Select a threshold")
        
        ret , bw = cv2.threshold(img, thmin, thmax, cv2.THRESH_BINARY)
        
        cv2.namedWindow("binarized", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("binarized", img.shape[0], img.shape[1]) # Resize window 
        cv2.imshow("binarized",bw)
        
        if cv2.waitKey(1) == ord('0'): # press 0 to close
            break
    
    cv2.destroyAllWindows() 
    
    return thmin, thmax