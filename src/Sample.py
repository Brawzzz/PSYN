#----------------------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------- IMPORT ----------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------------#
import os
import cv2 as cv
import numpy as np
import time
from tqdm import tqdm

import Fiber
import Region
import Split

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

        self.output_path    = stp.OUTPUT_PATH
        
        self.before_fret    = None 

        #---------------
        if(n_split % 2 != 0):
            if(n_split != 1):
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

        #---------------
        self.splits  : list[Split.Split]   = [] 
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
        
        if(n == 1):
            row = 1
            col = 1
            return(row, col)
        
        elif(n <= stp.MIN_COL):

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
        
        if(self.nb_split <= 1):

            print("Splitting image ... ", end="\r")
            split_i = Split.Split(n_id=0, sample_path=self.output_path)
            cv.imwrite(split_i.img_path, self.img)
            print("Splitting image ... Done")

            return 
        
        (img_h, img_w) = self.img.shape[:2]

        self.compute_row_col(self.nb_split) 
        h_step = img_h // self.row
        w_step = img_w // self.col

        (start_row, end_row) = (self.row_roi[0], self.row_roi[1])
        
        #---------------
        split_idx = 0
        total_iterations = (end_row - start_row) * self.col
        with tqdm(total=total_iterations, desc="Splitting images     ", unit="img") as pbar:
                                               
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
                    
                    split_i = Split.Split(n_id=split_idx, sample_path=self.output_path)
                    self.splits.append(split_i)

                    cv.imwrite(split_i.img_path, img_xy)

                    split_idx += 1
                    pbar.update(1)

    #--------------------------------------------------------------------------------#
    def join(self) -> None:

        suffix = "_all" + stp.OUTPUT_EXTENSION           
        recon_file = os.path.join(self.output_path, self.name + suffix)

        strip     = []
        all_split = []

        for split in self.splits:

            split_img_path = os.path.join(split.regions_dir, split.prefix + suffix)

            if os.path.exists(split_img_path):
                split_img = cv.imread(split_img_path, cv.IMREAD_COLOR_BGR)
            else:
                raise ValueError(f"join() Sample.py line 164 : split_image_{str(split.id)} do not exit : {split_img_path}")
            
            all_split.append(split_img)

        #---------------
        if not all_split:
            raise ValueError(f"join() Sample.py line 169 : no image found ")

        for y in tqdm(range(self.nb_strip), desc="Join           ", unit="row"):

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
            cv.imwrite(recon_file, img_join)

        else:
            raise ValueError(f"join() Sample.py line 199 : vstack error {strip}")    
    
    #--------------------------------------------------------------------------------#
    def tresh_img(self, 
                  thresh_type : int = stp.THRESH_TYPE,
                  blur_method : int = stp.GAUSSIAN_BLUR,
                  thresh_method : int = stp.CLASSIC_THRESH ):
        
        STEPS_LOOP = 2 
        actions = self.working_split * STEPS_LOOP
        with tqdm(total=actions, desc="Thresholding images  ", unit="itms") as pbar:

            #---------------
            for i in range(self.working_split):

                split_i = Split.Split(n_id=i, sample_path=self.output_path)
                img     = cv.imread(split_i.img_path, cv.IMREAD_GRAYSCALE)

                #---------------
                if(blur_method == stp.GAUSSIAN_BLUR):
                    img_blur = cv.GaussianBlur(img, stp.KERNEL_SIZE, sigmaX = 0)
                else:
                    img_blur = cv.GaussianBlur(img, stp.KERNEL_SIZE, sigmaX = 0)

                pbar.update(1)

                #---------------
                if(thresh_method == stp.CLASSIC_THRESH ):
                    (ret, img_bw) = cv.threshold(img_blur, stp.TH_MIN, stp.TH_MAX, thresh_type)
                else:
                    (ret, img_bw) = cv.threshold(img_blur, stp.TH_MIN, stp.TH_MAX, thresh_type)

                pbar.update(1)

                #---------------
                kernel = cv.getStructuringElement(cv.MORPH_CROSS, (5,5))
                img_bw_clean = cv.morphologyEx(img_bw, cv.MORPH_OPEN, kernel)

                cv.imwrite(split_i.thresh_path, img_bw_clean)

    #--------------------------------------------------------------------------------#
    def process_regions(self):
        
        STEPS_LOOP = 2 
        actions = self.working_split * STEPS_LOOP
        with tqdm(total=actions, desc="Process split  ", unit="itm") as pbar:
            
            for split_idx in range(self.working_split) :
                                                        
                split_i         = Split.Split(n_id=split_idx, sample_path=self.output_path)
                fibers          = Fiber.detect_fibers(split_i.thresh_path)
                sorted_fibers   = Fiber.sort_fibers(fibers)
                
                pbar.update(1)

                #---------------
                for i in range(len(sorted_fibers)):

                    reg_i = Region.Region(fibers = sorted_fibers[i], n_split_index = split_idx)
                    split_i.add_region(reg_i)

                self.splits.append(split_i)

                pbar.update(1)

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
            
            print(f"before fret     : {self.before_fret}\n")

            print(f"nb split        = {self.nb_split}")
            print(f"(row, col)      = ({self.row}, {self.col})")

            print(f"row_roi         = {self.row_roi}")

            print(f"splits.len      = {len(self.splits)}")
            print(f"#==================================================#\n")

    #--------------------------------------------------------------------------------#
    def render(self, render_type : int = stp.DRAW_FIBER):

        for split in tqdm(self.splits, desc="Rendering      ", unit="img"):
            split.save()

#----------------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------ STATIC METHODS ------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------------#
def init(n_id : int, n_split : int = 8) -> Sample:

    sample = Sample(n_id, n_split=n_split, n_row_roi=stp.ROW_ROI)
    sample.set_path(n_bf=True)

    tools.clear_folder(sample.output_path)

    print("\n")

    sample.load_img()
    sample.split()

    sample.tresh_img(thresh_type=stp.THRESH_TYPE,
                     blur_method=stp.GAUSSIAN_BLUR,
                     thresh_method=stp.CLASSIC_THRESH)

    print("\n")

    return sample