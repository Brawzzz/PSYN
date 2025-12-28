#----------------------------------------------------------------------------------------------#
#------------------------------------------- IMPORT -------------------------------------------#
#----------------------------------------------------------------------------------------------#
import os
import cv2 as cv
import numpy as np

import tools
import setup as stp
import Region as reg


#----------------------------------------------------------------------------------------------#
#------------------------------------------- CLASS --------------------------------------------#
#----------------------------------------------------------------------------------------------#
class Sample :

    def __init__(self, n_id):

        self.id             = n_id
        self.img            = None
        self.img_path       = ""  
        self.name           = "" 
        self.before_fret    = None 

        self.output_path    = stp.OUTPUT_PATH

        self.split_path     = ""
        self.row            = 0
        self.col            = 0

        self.thresh_path    = "" 

        self.regions        = []
        self.regions_path   = ""   

    #--------------------------------------------------------------------------------#
    def set_path(self, n_bf=True):

        self.before_fret = n_bf

        if(self.before_fret):
            self.name = "hxtl_p" + self.id + "_pre.bmp"
            self.img_path = stp.DATA_PATH + "sample_" + str(self.id) + "/before_fretting/" + self.name
        else:
           self.name = "hxtl_p" + self.id + "_post.bmp"
           self.img_path = stp.DATA_PATH + "sample_" + str(self.id) + "/after_fretting/" + self.name 
        
        self.name           = os.path.splitext(self.name)[0]
        self.output_path    = self.output_path + self.name + "/"

        self.split_path     = self.output_path + stp.SPLIT_PATH
        self.thresh_path    = self.output_path + stp.THRESH_PATH
        self.regions_path   = self.output_path + stp.REGION_PATH
    
    #--------------------------------------------------------------------------------#
    def load_img(self):

        if(self.img_path == None) :
            raise ValueError(f"Impossible to load img : img_path = {self.img_path}")

        elif(not os.path.exists(self.img_path)) :
            raise FileExistsError(f"Impossible to load img : img_path do not exist : img_path = {self.img_path}")

        else :
            self.img = cv.imread(self.img_path, cv.IMREAD_GRAYSCALE)

    #--------------------------------------------------------------------------------#
    def compute_row_col(self, n : int):

        if(n == 2):
            self.row = 2
            self.col = 1
        
        self.row = 2
        self.col = int(n / self.row)

        while(self.col % 2 == 0 and self.col > 4):

            self.row *= 2
            self.col = int(self.col / 2)        

    #--------------------------------------------------------------------------------#
    def split(self, nb_split : int = 2):
        
        if(os.path.exists(self.split_path) and len(os.listdir(self.split_path)) ==  nb_split):
            self.compute_row_col(nb_split) 
            return
        
        if(nb_split % 2 != 0):
            raise ValueError(f"nb_split must be even : nb_split = {nb_split}")
        
        if(tools.img_empty(self.img)):
            raise ValueError(f"impossible to split : self.img = {self.img} (is empty) ")
        
        os.makedirs(self.split_path, exist_ok=True)
        
        (img_h, img_w) = self.img.shape[:2]

        self.compute_row_col(nb_split) 
        h_step = img_h // self.row
        w_step = img_w // self.col

        i = 0
        for y in range(self.row):
            for x in range(self.col):
                
                y_start = y * h_step
                y_end = (y + 1) * h_step

                x_start = x * w_step
                x_end = (x + 1) * w_step

                if (y == self.row - 1):
                    y_end = img_h

                if (x == self.col - 1):
                    x_end = img_w

                img_xy = self.img[y_start:y_end, x_start:x_end]
                
                file_name = self.split_path + stp.SPLIT + str(i) + stp.OUTPUT_EXTENSION
                cv.imwrite(file_name, img_xy)

                i += 1

    #--------------------------------------------------------------------------------#
    def join(self) -> np.ndarray:

        strip = []
        for y in range(self.row):
                
            start_index = y * self.col
            end_index = start_index + self.col

            row_img = self.split_list[start_index:end_index]
            strip_i = np.hstack(row_img)
            strip.append(strip_i) 

        img = np.vstack(strip)

        cv.imwrite(stp.RECON_PATH + os.path.splitext(self.name)[0] + "_recon.jpg", img)

        return img        
    
    #--------------------------------------------------------------------------------#
    def tresh_img(self, thresh_method : int = stp.THRESH_METHOD):
        
        os.makedirs(self.thresh_path, exist_ok=True)

        for i in range(len(os.listdir(self.split_path))):

            split_img   = self.split_path + stp.SPLIT + str(i) + stp.OUTPUT_EXTENSION
            thresh_img  = self.thresh_path + stp.THRESH + str(i) + stp.OUTPUT_EXTENSION

            img             = cv.imread(split_img, cv.IMREAD_GRAYSCALE)
            img_blur        = cv.GaussianBlur(img, stp.KERNEL_SIZE, sigmaX = 0)
            (ret, img_bw)   = cv.threshold(img_blur, stp.TH_MIN, stp.TH_MAX, thresh_method)

            cv.imwrite(thresh_img, img_bw)

    #--------------------------------------------------------------------------------#
    def print(self):

        print(f"#================ SAMPLE {self.id} ================#")
        print(f"name            : {self.name}")
        print(f"path            : {self.img_path}")
        print(f"split_path      : {self.split_path}")
        print(f"thresh_path     : {self.thresh_path}")
        print(f"regions_path    : {self.regions_path}")
        print(f"before fret     : {self.before_fret}")

        print(f"\nsplit (row, col) : ({self.row}, {self.col})")

    #--------------------------------------------------------------------------------#
    def save(self):

        return