#----------------------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------- IMPORT ----------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------------#
import os
import glob
import json
import cv2 as cv
import numpy as np
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

    def __init__(self, n_id, n_split):

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
        self.process_split      = self.nb_split

        #---------------
        self.splits  : list[Split.Split]   = [] 
        self.regions : list[Region.Region] = [] 

        self.main_angles = []

        self.color_config = ""

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

        self.color_config = os.path.join(self.output_path, self.name + "_color_config.json")

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
        
        MIN_COL = 12

        if(n == 1):
            row = 1
            col = 1
            return(row, col)
        
        elif(n <= MIN_COL):

            row = 2
            col = int(n / row)
            return(row, col)
        
        else :

            row = 2
            col = int(n / row)

            while(col % 2 == 0 and col > MIN_COL):

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

        h_step = img_h // self.row
        w_step = img_w // self.col

        #---------------
        y_max = int((self.process_split-1) // self.col) 
        x_max = int((self.process_split-1) % self.col)

        split_idx = 0
        with tqdm(total=self.process_split, desc="Splitting images     ", unit="img") as pbar:
                                               
            for y in range(0, y_max+1):
                for x in range(0, x_max+1):
                    
                    y_start = y * h_step
                    y_end = (y + 1) * h_step

                    x_start = x * w_step
                    x_end = (x + 1) * w_step

                    if (y == self.row - 1):
                        y_end = img_h

                    if (x == self.col - 1):
                        x_end = img_w

                    img_xy = self.img[y_start:y_end, x_start:x_end]
                    
                    origin = [x_start, y_start]
                    split_i = Split.Split(n_id=split_idx, n_origin=origin, sample_path=self.output_path)
                    self.splits.append(split_i)

                    cv.imwrite(split_i.img_path, img_xy)

                    split_idx += 1
                    pbar.update(1)

    #--------------------------------------------------------------------------------#
    def join(self) -> None:

        if(self.process_split % self.col != 0):
            raise ValueError(f"\nSample.py join() line 158 : process_split is not a multiple of self.col = {self.process_split % self.col}\n")
        
        suffix          = "_all" + stp.OUTPUT_EXTENSION 
        search_pattern  = os.path.join(self.output_path, "**", "*" + suffix)
        all_split_files = sorted(glob.glob(search_pattern, recursive=True), key=tools.extract_number)
        recon_file      = self.output_path + self.name + suffix

        strip     = []
        all_split = []

        for file in all_split_files:

            if os.path.exists(file):
                split_img = cv.imread(file, cv.IMREAD_COLOR_BGR)
            else:
                raise ValueError(f"join() Sample.py line 164 : {file} do not exist")
            
            all_split.append(split_img)

        #---------------
        if not all_split:
            raise ValueError(f"join() Sample.py line 169 : no image found ")

        y_max = int((self.process_split-1) // self.col) 
        for y in tqdm(range(y_max+1), desc="Join           ", unit="row"):

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
            raise ValueError(f"join() Sample.py line 213 : vstack error {strip}")    
        
    #--------------------------------------------------------------------------------#
    def tresh_img(self,
                  blur_method   : int = stp.GAUSSIAN_BLUR,
                  thresh_method : int = stp.CLASSIC_THRESH):

        with tqdm(total=self.process_split, desc="Thresholding images  ", unit="itm") as pbar:

            #---------------
            for split in self.splits:

                img = cv.imread(split.img_path, cv.IMREAD_GRAYSCALE)

                #---------------
                if(blur_method == stp.GAUSSIAN_BLUR):
                    img_blur = cv.GaussianBlur(img, stp.KERNEL_SIZE, sigmaX = 0)

                elif(blur_method == stp.BILATERAL_BLUR):
                    img_blur = cv.bilateralFilter(img, d=stp.BILATERAL_D, 
                                                  sigmaColor=stp.BILATERAL_SIGMA_COLOR, 
                                                  sigmaSpace=stp.BILATERAL_SIGMA_SPACE)

                else :
                    img_blur = cv.blur(img, stp.KERNEL_SIZE)

                cv.imwrite(split.dir_path + split.prefix + "_blur" + stp.OUTPUT_EXTENSION, img_blur)

                #---------------
                if(thresh_method == stp.CLASSIC_THRESH):
                    (_, img_bw) = cv.threshold(img_blur, stp.TH_MIN, stp.TH_MAX, stp.THRESH_TYPE)

                else:
                    img_bw = cv.adaptiveThreshold(
                        img_blur, 
                        maxValue=stp.MAX, 
                        adaptiveMethod=cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
                        thresholdType=cv.THRESH_BINARY,
                        blockSize=stp.K_SIZE,
                        C=stp.C        
                    )

                kernel = cv.getStructuringElement(shape=cv.MORPH_ELLIPSE, ksize=(3,3))
                img_bw = cv.morphologyEx(img_bw, cv.MORPH_OPEN, kernel)

                cv.imwrite(split.thresh_path, img_bw)

                pbar.update(1)

    #--------------------------------------------------------------------------------#
    def process_regions(self):
        
        with tqdm(total=self.process_split, desc="Process split        ", unit="itm") as pbar:
            
            for split in self.splits :
                                                        
                fibers          = Fiber.detect_fibers(split.thresh_path)
                sorted_fibers   = Fiber.group_fibers(fibers)

                #---------------
                for i in range(len(sorted_fibers)):

                    reg_i = Region.Region(fibers = sorted_fibers[i], n_split_index = split.id)
                    if(reg_i.mean_angle != -1):
                        (split.regions).append(reg_i)

                self.regions = self.regions + split.regions

                pbar.update(1)

        angles           = [reg.mean_angle for reg in self.regions]    
        self.main_angles = tools.get_peaks(angles, min_peak_height=0, sigma_smoth=1.0)

    #--------------------------------------------------------------------------------#
    def group_regions(self):

        regions_group = {idx : [] for idx in self.main_angles}
        
        for reg in self.regions:

            ang         = reg.mean_angle
            best_peak   = -1
            min_dist    = float('inf')

            for peak in self.main_angles:
                
                d1 = abs(peak - ang)
                d2 = stp.MAX_ANGLE - d1
                d  = min(d1, d2)

                if(d < min_dist):

                    min_dist = d
                    best_peak = peak

            if(min_dist <= stp.DELTA_ANGLE):
                regions_group[best_peak].append(reg)

        regions = []
        for peak in regions_group: 
            
            group        : list[Region.Region] = regions_group[peak]
            mapped_group : list[Region.Region] = [] 
            for reg in group:
                
                current_split = self.splits[reg.split_index]

                mapped_fibers : list[Fiber.Fiber] = []
                for fib in reg.fibers:
                    mapped_fib = fib.map_fiber(current_split) 
                    mapped_fibers.append(mapped_fib)

                mapped_reg = Region.Region(mapped_fibers, n_split_index=-1)
                mapped_group.append(mapped_reg)

            region_peak       = Region.merge_regions(mapped_group)
            region_peak.shape = region_peak.compute_shape()
            
            regions.append(region_peak)
            
        return regions

    #--------------------------------------------------------------------------------#
    def render_config(self):

        color_config = []
        for ang in self.main_angles:

            config = {
                "angle": ang,
                "color": tools.angle_to_color(ang)
            }

            color_config.append(config)

        with open(self.color_config, "w", encoding="utf-8") as f:

            final_json = {
                "description": self.name + " Color Config",
                "colors": color_config
            }

            json.dump(final_json, f, indent=4)

    #--------------------------------------------------------------------------------#
    def render(self, img : np.ndarray = None):

        if tools.img_empty(img):
            for split in tqdm(self.splits, desc="Rendering      ", unit="img"):
                split.save(self.color_config)
        
        else:
            for reg in self.regions:
                reg.render_fibers(img, self.color_config)

    #--------------------------------------------------------------------------------#
    def print(self, region=False):

        if(region):
            print(f"\n#=================== SAMPLE {self.id} ====================#\n")

            for reg in self.regions:
                reg.print()

            print(f"#==================================================#\n")

        else:
            print(f"\n#=================== SAMPLE {self.id} ====================#")
            print(f"name            : {self.name}")
            print(f"path            : {self.img_path}")
            
            print(f"before fret     : {self.before_fret}\n")

            print(f"nb split        = {self.nb_split}")
            print(f"(row, col)      = ({self.row}, {self.col})")
            print(f"splits.len      = {len(self.splits)}")

            print(f"regions.len     = {len(self.regions)}")
            print(f"main angles     = {self.main_angles}")
            print(f"#==================================================#\n")

#----------------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------ STATIC METHODS ------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------------#
def init(n_id : int, n_split : int = 8) -> Sample:

    sample = Sample(n_id, n_split=n_split)
    sample.set_path(n_bf=True)

    if(os.path.exists(sample.output_path)):
        tools.clear_folder(sample.output_path)
    else:
        os.makedirs(sample.output_path, exist_ok=True)

    print("\n")

    sample.load_img()
    sample.split()

    sample.tresh_img(blur_method=stp.BILATERAL_BLUR,
                     thresh_method=stp.CLASSIC_THRESH)

    return sample