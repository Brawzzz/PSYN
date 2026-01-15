#----------------------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------- IMPORT ----------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------------#
import os
import glob
import cv2 as cv
import numpy as np

import Fiber
import Region

import tools
import setup as stp

#----------------------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------- CLASS -----------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------------#
class Sample :

    def __init__(self, n_id, n_split, n_row_roi : tuple[int, int]):

        self.id             = n_id
        self.name           = ""

        self.img_path       = ""  
        self.img            = None

        self.before_fret    = None 

        self.output_path    = stp.OUTPUT_PATH
        
        self.split_path     = ""
        self.thresh_path    = "" 
        self.regions_path   = ""  
        self.recon_path     = "" 

        if(n_split % 2 != 0):
            raise ValueError(f"__init__() Sample.py line 29 : nb_split must be even : nb_split = {n_split}")
        
        self.nb_split           = n_split
        (self.row, self.col)    = self.compute_row_col(self.nb_split)

        if((n_row_roi[1] <= self.row) and (n_row_roi[1] > n_row_roi[0])):
            self.row_roi = n_row_roi 
        else:
            self.row_roi = (0, self.row)
            Warning(f"we could have : {n_row_roi[1]} > {n_row_roi[0]} : default row_roi = (0, {self.row})")

        self.nb_strip      = (self.row_roi[1] - self.row_roi[0])  
        self.working_split = self.nb_strip  * self.col

        self.regions : list[Region.Region] = [] 

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
    def compute_row_col(self, n : int) -> tuple[int, int]:

        if(n <= stp.MIN_COL):

            row = 2
            col = int(n / row)
            return(row, col)
        
        else :

            row = 2
            col = int(n / row)

            while(col % 2 == 0 and col > stp.MIN_COL):

                row *= 2
                col = int(col / 2)        

            return(row, col)
        
    #--------------------------------------------------------------------------------#
    def split(self):

        if(tools.img_empty(self.img)):
            raise ValueError(f"impossible to split : self.img = {self.img} (is empty) ")
        
        (img_h, img_w) = self.img.shape[:2]

        self.compute_row_col(self.nb_split) 
        h_step = img_h // self.row
        w_step = img_w // self.col

        (start_row, end_row) = (self.row_roi[0], self.row_roi[1])
        
        #---------------
        i = 0
        for y in range(start_row, end_row):
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
        for i in range(self.working_split):
            
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

        for y in range(self.nb_strip):

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
                  thresh_method : int = stp.CLASSIC_THRESH ):
        
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
            if(thresh_method == stp.CLASSIC_THRESH ):
                (ret, img_bw) = cv.threshold(img_blur, stp.TH_MIN, stp.TH_MAX, thresh_type)
            else:
                (ret, img_bw) = cv.threshold(img_blur, stp.TH_MIN, stp.TH_MAX, thresh_type)

            #---------------
            kernel = cv.getStructuringElement(cv.MORPH_CROSS, (5,5))
            img_bw_clean = cv.morphologyEx(img_bw, cv.MORPH_OPEN, kernel)

            cv.imwrite(thresh_img, img_bw_clean)

    #--------------------------------------------------------------------------------#
    def process_regions(self):

        thresh_split_names = sorted(glob.glob(self.thresh_path + "*" + stp.OUTPUT_EXTENSION), key=tools.extract_number)

        #---------------
        split_index = 0
        nb_split = len(thresh_split_names)

        for thresh_split in thresh_split_names:
            
            print(f"Process split {split_index+1}/{nb_split} ... ", end="\r")

            fibers          = Fiber.detect_fibers(thresh_split)
            sorted_fibers   = Fiber.sort_fibers(fibers)

            #---------------
            for i in range(len(sorted_fibers)):

                reg_i = Region.Region(fibers = sorted_fibers[i], 
                                    n_split_index = split_index, 
                                    sample_regions_path = self.regions_path)
                
                self.regions.append(reg_i)

            split_index += 1
        
        print(f"Process split {split_index}/{nb_split} ... Done\n")

    #--------------------------------------------------------------------------------#
    def print(self, region=False):

        if(region):
            print(f"#=================== SAMPLE {self.id} ====================#\n")

            for reg in self.regions:
                reg.print()

            print(f"#==================================================#\n")

        else:
            print(f"#=================== SAMPLE {self.id} ====================#")
            print(f"name            : {self.name}")
            print(f"path            : {self.img_path}")
            print(f"split_path      : {self.split_path}")
            print(f"thresh_path     : {self.thresh_path}")
            print(f"regions_path    : {self.regions_path}")
            print(f"before fret     : {self.before_fret}\n")

            print(f"split           = {self.nb_split}")
            print(f"(row, col)      = ({self.row}, {self.col})")

            print(f"row_roi         = {self.row_roi}")
            print(f"#==================================================#\n")

    #--------------------------------------------------------------------------------#
    def render(self, render_type : int = stp.DRAW_FIBER):
        
        prev_split_index = 0

        split_img_path    = self.split_path + stp.SPLIT_ + str(prev_split_index) + stp.OUTPUT_EXTENSION
        split_img         = cv.imread(split_img_path, cv.IMREAD_COLOR_BGR)

        print(f"Rendering regions ... ", end="\r")

        #---------------
        for reg in self.regions:

            split_img_path  = self.split_path + stp.SPLIT_ + str(reg.split_index) + stp.OUTPUT_EXTENSION
            region_img      = cv.imread(split_img_path, cv.IMREAD_COLOR_BGR)

            #---------------
            if(prev_split_index != reg.split_index):
                
                suffix          = stp.SPLIT_ + str(prev_split_index)
                all_img_path    = os.path.join(self.regions_path + suffix, suffix + stp.ALL_REGIONS + stp.OUTPUT_EXTENSION)

                cv.imwrite(all_img_path, split_img)

                split_img        = cv.imread(split_img_path, cv.IMREAD_COLOR_BGR)
                prev_split_index = reg.split_index

            #---------------
            reg.render(all_img      = split_img,
                       region_img   = region_img,
                       render_type  = render_type)
        
        suffix       = stp.SPLIT_ + str(prev_split_index)
        all_img_path = os.path.join(self.regions_path + suffix, suffix + stp.ALL_REGIONS + stp.OUTPUT_EXTENSION)

        cv.imwrite(all_img_path, split_img)

        print(f"Rendering regions ... Done\n")

#----------------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------ STATIC METHODS ------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------------#
def init(n_id : int, n_split : int = 8) -> Sample:

    sample = Sample(n_id, n_split=n_split, n_row_roi=stp.ROW_ROI)
    sample.set_path(n_bf=True)

    tools.clear_folder(sample.split_path)
    tools.clear_folder(sample.thresh_path)

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