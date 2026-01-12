#----------------------------------------------------------------------------------------------#
#------------------------------------------- IMPORT -------------------------------------------#
#----------------------------------------------------------------------------------------------#
import os
import glob
import cv2 as cv
import numpy as np

import tools
import Region

import setup as stp

#----------------------------------------------------------------------------------------------#
#------------------------------------------- CLASS --------------------------------------------#
#----------------------------------------------------------------------------------------------#
class Sample :

    def __init__(self, n_id, n_split):

        self.id             = n_id
        self.img            = None
        self.img_path       = ""  
        self.name           = "" 
        self.before_fret    = None 

        self.output_path    = stp.OUTPUT_PATH

        if(n_split % 2 != 0):
            raise ValueError(f"__init__() Sample.py line 29 : nb_split must be even : nb_split = {n_split}")
        
        self.split_path     = ""
        self.nb_split       = n_split
        self.row            = 0
        self.col            = 0

        self.thresh_path    = "" 

        self.regions        = []
        self.regions_path   = ""  

        self.recon_path     = "" 

    #--------------------------------------------------------------------------------#
    def set_path(self, n_bf=True):

        self.before_fret = n_bf

        if(self.before_fret):
            self.name       = "hxtl_p" + self.id + "_pre.bmp"
            self.img_path   = stp.DATA_PATH + "sample_" + str(self.id) + "/before_fretting/" + self.name
        else:
           self.name        = "hxtl_p" + self.id + "_post.bmp"
           self.img_path    = stp.DATA_PATH + "sample_" + str(self.id) + "/after_fretting/" + self.name 
        
        #---------------
        self.name           = os.path.splitext(self.name)[0]
        self.output_path    = self.output_path + self.name + "/"

        self.split_path     = self.output_path + stp.SPLIT_PATH
        self.thresh_path    = self.output_path + stp.THRESH_PATH
        self.regions_path   = self.output_path + stp.REGION_PATH
        self.recon_path     = self.output_path + stp.RECON_PATH

        #---------------
        os.makedirs(self.split_path, exist_ok=True)
        os.makedirs(self.thresh_path, exist_ok=True)
        os.makedirs(self.regions_path, exist_ok=True)
        os.makedirs(self.recon_path, exist_ok=True)

        tools.clear_folder(self.regions_path)

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

        if(n <= stp.MIN_COL):
            self.row = 2
            self.col = int(n / self.row)
            return
        
        else :
            self.row = 2
            self.col = int(n / self.row)

            while(self.col % 2 == 0 and self.col > stp.MIN_COL):

                self.row *= 2
                self.col = int(self.col / 2)        

    #--------------------------------------------------------------------------------#
    def split(self):
        
        sucess = tools.verifiy_folder(self.split_path, folder_len=self.nb_split)
        if(not sucess):
            self.compute_row_col(self.nb_split) 
            return
        
        if(tools.img_empty(self.img)):
            raise ValueError(f"impossible to split : self.img = {self.img} (is empty) ")
        
        (img_h, img_w) = self.img.shape[:2]

        self.compute_row_col(self.nb_split) 
        h_step = img_h // self.row
        w_step = img_w // self.col

        #---------------
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
                
                file_name = self.split_path + stp.SPLIT_ + str(i) + stp.OUTPUT_EXTENSION
                cv.imwrite(file_name, img_xy)

                i += 1

    #--------------------------------------------------------------------------------#
    def join(self, n_row : int = -1) -> None:

        suffix = "_all" + stp.OUTPUT_EXTENSION           
        recon_name = os.path.join(self.recon_path, self.name + suffix)

        all_split = []
        strip = []

        #---------------
        if(n_row <= 0):
            nb_row = self.row
        else:
            nb_row = n_row

        nb_split = self.col * nb_row
        for i in range(nb_split):
            
            folder_name     = stp.SPLIT_ + str(i)
            file_name       = stp.SPLIT_ + str(i) + suffix
            split_img_path  = os.path.join(self.regions_path, folder_name, file_name)

            split_img = None

            if os.path.exists(split_img_path):
                split_img = cv.imread(split_img_path, cv.IMREAD_COLOR)
            
            if split_img is None:
                raise ValueError(f"join() Sample.py line 164  : split_image_{i} do not exit {split_img_path}")

            all_split.append(split_img)

        #---------------
        if not all_split:
            raise ValueError(f"join() Sample.py line 169 : no image found ")

        for y in range(nb_row):

            start_index = y * self.col
            end_index = start_index + self.col

            row_images = all_split[start_index:end_index]
            
            if not row_images:
                print(f"Erreur reconstruction ligne {y} : liste vide")
                continue

            try:
                strip_i = np.hstack(row_images)
                strip.append(strip_i)
            except ValueError as e:
                print(f"hstack error ligne {y} : {e}")

        if strip:
            img_join = np.vstack(strip)
            cv.imwrite(recon_name, img_join)
        else:
            raise ValueError(f"join() Sample.py line 199 : vstack error {strip}")    
    
    #--------------------------------------------------------------------------------#
    def tresh_img(self, 
                  thresh_type : int = stp.THRESH_TYPE,
                  blur_method : int = stp.GAUSSIAN_BLUR,
                  thresh_method : int = stp.CLASSIC_THRESH_METHOD):
        
        os.makedirs(self.thresh_path, exist_ok=True)

        #---------------
        for i in range(len(os.listdir(self.split_path))):

            split_img   = self.split_path + stp.SPLIT_ + str(i) + stp.OUTPUT_EXTENSION
            thresh_img  = self.thresh_path + stp.THRESH_ + str(i) + stp.OUTPUT_EXTENSION

            img = cv.imread(split_img, cv.IMREAD_GRAYSCALE)

            #---------------
            if(blur_method == stp.GAUSSIAN_BLUR):
                img_blur = cv.GaussianBlur(img, stp.KERNEL_SIZE, sigmaX = 0)
            else:
                img_blur = cv.GaussianBlur(img, stp.KERNEL_SIZE, sigmaX = 0)

            #---------------
            if(thresh_method == stp.GAUSSIAN_BLUR):
                (ret, img_bw) = cv.threshold(img_blur, stp.TH_MIN, stp.TH_MAX, thresh_type)
            else:
                (ret, img_bw) = cv.threshold(img_blur, stp.TH_MIN, stp.TH_MAX, thresh_type)

            kernel = cv.getStructuringElement(cv.MORPH_CROSS, (5,5))
            img_bw_clean = cv.morphologyEx(img_bw, cv.MORPH_OPEN, kernel)

            cv.imwrite(thresh_img, img_bw_clean)

    #--------------------------------------------------------------------------------#
    def print(self):

        print(f"#=================== SAMPLE {self.id} ====================#")
        print(f"name            : {self.name}")
        print(f"path            : {self.img_path}")
        print(f"split_path      : {self.split_path}")
        print(f"thresh_path     : {self.thresh_path}")
        print(f"regions_path    : {self.regions_path}")
        print(f"before fret     : {self.before_fret}\n")

        print(f"split           = {self.nb_split}")
        print(f"(row, col)      = ({self.row}, {self.col})")
        print(f"#==================================================#\n")

    #--------------------------------------------------------------------------------#
    def save_region(self, region : Region.Region) -> None:
        self.regions.append(region)
    
#----------------------------------------------------------------------------------------------#
#------------------------------------------- STATIC -------------------------------------------#
#----------------------------------------------------------------------------------------------#
def init(n_id : int, n_split : int = 8) -> Sample:

    sample = Sample(n_id, n_split=n_split)
    sample.set_path(n_bf=True)

    print("\nLoading image .........", end="\r")
    sample.load_img()
    print("Loading image ........... Done")

    print("Spliting image .......... ", end="\r")
    sample.split()
    print("Spliting image .......... Done")

    print("Thresholding images ..... ", end="\r")
    sample.tresh_img(thresh_type=stp.THRESH_TYPE,
                    blur_method=stp.GAUSSIAN_BLUR,
                    thresh_method=stp.CLASSIC_THRESH_METHOD)
    print("Thresholding images ..... Done\n")

    return sample