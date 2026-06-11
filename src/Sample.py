#============================================================================================================================#
#---------------------------------------------------------- IMPORT ----------------------------------------------------------#
#============================================================================================================================#
import os
import glob
import json
import cv2 as cv
import numpy as np

from tqdm import tqdm
from datetime import datetime

import Fiber
import Region
import Split

import tools
import setup as stp


#============================================================================================================================#
#---------------------------------------------------------- CLASS -----------------------------------------------------------#
#============================================================================================================================#
class Sample :

    """
    The Sample class contains the tribological data and the image analysis.

    It is defined by mainly :

    n_id    : the id of the sample (exemple : 22, 23, 24, 25) 
    img     : the image from the tribological study
    regions : a list of Region instaces from the image analysis
    """

    #------------------------------
    def __init__(self, n_id, n_split):

        self.id             = n_id
        self.name           = ""

        self.img_path       = ""  
        self.img            = None

        self.output_path    = ""
        self.splits_path    = ""
        self.regions_path   = ""
        
        self.fretting       = None 
        self.config         = "" 

        #------------------------------
        if(n_split % 2 != 0):
            if(n_split != 1):
                raise ValueError(f"__init__() Sample.py line 29 : nb_split must be even : nb_split = {n_split}")
        
        self.nb_split           = n_split
        (self.row, self.col)    = self.compute_row_col(self.nb_split)
        self.process_split      = self.nb_split

        #------------------------------
        self.main_angles                   = []
        self.splits  : list[Split.Split]   = [] 
        self.regions : list[Region.Region] = [] 
        self.regions_data                  = []

    #================================================================================#
    def set_path(self, n_fret=False):

        """
        Set the different path needed for a Sample

        n_fret : True if running affter fretting
        """

        #------------------------------
        self.fretting = n_fret

        if(not self.fretting):
            self.name       = "hxtl_p" + self.id + "_pre.bmp"
            self.img_path   = stp.DATA_PATH + "sample_" + str(self.id) + "/before_fretting/" + self.name
        else:
           self.name        = "hxtl_p" + self.id + "_post.bmp"
           self.img_path    = stp.DATA_PATH + "sample_" + str(self.id) + "/after_fretting/" + self.name 
        
        #------------------------------
        self.name           = os.path.splitext(self.name)[0]

        self.output_path    = stp.OUTPUT_PATH + self.name + "/"
        self.splits_path    = self.output_path + "splits/"
        self.regions_path   = self.output_path + "regions/"

        date_str        = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        config_suffix   = "_config_" + date_str + ".json" 
        self.config     = os.path.join(self.output_path, self.name + config_suffix)

    #================================================================================#
    def load_img(self):

        """
        load the corresponding image from the sample 
        """

        #------------------------------
        if(self.img_path == None) :
            raise ValueError(f"Impossible to load img : img_path = {self.img_path}")

        elif(not os.path.exists(self.img_path)) :
            raise FileExistsError(f"Impossible to load img : img_path do not exist : img_path = {self.img_path}")

        else :
            self.img = cv.imread(self.img_path, cv.IMREAD_GRAYSCALE)

    #================================================================================#
    def compute_row_col(self, n : int) -> tuple[int, int]:
        
        """
        compute and set the row/col attributes

        n : number of split 
        """

        #------------------------------
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
        
    #================================================================================#
    def split(self, save : bool = True):

        """
        split the sample in a list of Split object

        save : if true it saves the correspondind split's image otherwise 
               it return a list containing all the splits images (default : true)
        """

        #------------------------------
        if(tools.img_empty(self.img)):
            raise ValueError(f"impossible to split : self.img = {self.img} (is empty) ")
        
        if(self.nb_split <= 1):

            with tqdm(total=self.process_split, desc="Splitting image      ", unit="img") as pbar:
            
                split_i = Split.Split(n_id=0, n_origin=(0,0), sample_path=self.output_path)
                self.splits.append(split_i)

                cv.imwrite(split_i.img_path, self.img)
                pbar.update(1)

            return 
        
        #------------------------------
        (img_h, img_w) = self.img.shape[:2]

        h_step = img_h // self.row
        w_step = img_w // self.col

        y_max = int((self.process_split-1) // self.col) 
        x_max = int((self.process_split-1) % self.col)

        split_idx = 0
        with tqdm(total=self.process_split, desc="Splitting image      ", unit="img") as pbar:

            images = []                             
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

                    split_i = Split.Split(n_id=split_idx, n_origin=origin, sample_path=self.splits_path)
                    self.splits.append(split_i)
                    
                    if save:
                        cv.imwrite(split_i.img_path, img_xy)
                    else:
                        images.append(img_xy)

                    split_idx += 1
                    pbar.update(1)

        if images :
            return images
        
    #================================================================================#
    def join(self) -> None:

        """
        reconstruct the initial image form the splits
        """
                
        #------------------------------
        if(self.process_split % self.col != 0):
            raise ValueError(f"\nSample.py join() line 158 : process_split is not a multiple of self.col = {self.process_split % self.col}\n")
        
        suffix          = "_all" + stp.OUTPUT_EXTENSION 
        search_pattern  = os.path.join(self.splits_path, "**", "*" + suffix)
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

        #------------------------------
        if not all_split:
            raise ValueError(f"join() Sample.py line 169 : no image found ")

        y_max = int((self.process_split-1) // self.col) 

        for y in tqdm(range(y_max+1), desc="Join                 ", unit="split"):
                                           
            start_index = y * self.col
            end_index   = start_index + self.col
            row_images  = all_split[start_index:end_index]
            
            if not row_images:
                print(f"Erreur reconstruction ligne {y} : liste vide")
                continue

            try:
                strip_i = np.hstack(row_images)
                strip.append(strip_i)
            except ValueError as e:
                print(f"hstack error ligne {y} : {e}")

        #------------------------------
        if strip:
            img_join = np.vstack(strip)
            cv.imwrite(recon_file, img_join)

        else:
            raise ValueError(f"join() Sample.py line 213 : vstack error {strip}")    
        
    #================================================================================#
    def tresh_img(self,
                  blur_method   : int = stp.GAUSSIAN_BLUR,
                  thresh_method : int = stp.CLASSIC_THRESH):

        """
        Thresholding of the different splits, the function do both blur and thresh

        blur_method : tell the wanted blur method (default : cv.GaussianBlur())
        thresh_method : tell the wanted thresh method (default : cv.threshold())
        """

        #------------------------------
        with tqdm(total=self.process_split, desc="Thresholding images  ", unit="itm") as pbar:
                                                 
            #------------------------------
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

                #---------------
                cv.imwrite(split.blur_path, img_blur)
                cv.imwrite(split.thresh_path, img_bw)

                pbar.update(1)

    #================================================================================#
    def process_splits(self):
        
        """
        process the split : detection / seltion of fiber
        """
        
        #------------------------------  
        with tqdm(total=self.process_split, desc="Processing split     ", unit="itm") as pbar:
            
            for split in self.splits :
                                                        
                fibers          = Fiber.detect_fibers(split.thresh_path)
                sorted_fibers   = Fiber.group_fibers(fibers)

                #------------------------------
                for i in range(len(sorted_fibers)):

                    reg_i = Region.Region(n_fibers = sorted_fibers[i], n_split_index = split.id)
                    if(reg_i.mean_angle != -1):
                        (split.regions).append(reg_i)

                self.regions = self.regions + split.regions

                pbar.update(1)

        angles           = [reg.mean_angle for reg in self.regions]    
        self.main_angles = tools.get_peaks(angles, min_peak_height=0, sigma_smoth=2)

    #================================================================================#
    def group_regions(self):

        """
        Classify all the fiber depending on their angles
        """
        
        #------------------------------
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

        #------------------------------
        regions = []

        with tqdm(total=len(regions_group), desc="Computing global regions ", unit="reg") as pbar:

            for peak in regions_group: 
                
                group        : list[Region.Region] = regions_group[peak]
                mapped_group : list[Region.Region] = [] 

                #---------------
                for reg in group:
                    
                    current_split = self.splits[reg.split_index]

                    mapped_fibers : list[Fiber.Fiber] = []
                    for fib in reg.fibers:
                        mapped_fib = fib.map_fiber(current_split) 
                        mapped_fibers.append(mapped_fib)

                    mapped_reg = Region.Region(mapped_fibers, n_split_index=-1)
                    mapped_group.append(mapped_reg)

                #---------------
                region_peak = Region.merge_regions(mapped_group)

                if(len(region_peak.fibers) > stp.REGION_MIN_FIBER):
                    regions.append(region_peak)

                pbar.update(1)
                
            self.regions = regions

            return regions
    
    #================================================================================#
    def global_shape(self) -> None:

        """
        compute the shape attributes of each global regions in self.regions
        """
        
        #------------------------------
        with tqdm(total=len(self.regions), desc="Computing global shapes  ", unit="reg") as pbar:

            for reg in self.regions:

                reg.shape       = reg.compute_shape(method=stp.SHAPE_METHOD)  
                reg.area        = reg.shape.area
                reg.centroid    = (reg.shape.centroid.x,reg.shape.centroid.y)
                pbar.update(1)
        
        return
    
    #================================================================================#
    def get_statistics(self, filename=None, graph=False):

        for reg in self.regions:

            reg_data = (reg.mean_angle, 
                        reg.nb_fibers, 
                        reg.mean_fibers_len,
                        reg.mean_fibers_width,
                        reg.area,
                        reg.centroid)
            
            self.regions_data.append(reg_data)

        return(self.regions_data)

    #================================================================================#
    def save_config(self) -> None:

        """
        configuration file as an output summary of the params used for a particular run

        file name : sample.name + JJ_MM_AAAA_HH_MIN_S .json (exemple : hxtl_p25_pre_config_26_02_2026_13_51_51.json)
        """
        
        #------------------------------
        color_config = []
        for ang in self.main_angles:

            config = {
                "angle": ang,
                "color": tools.angle_to_color(ang)
            }

            color_config.append(config)

        #------------------------------
        params = {
            "nb_split"        : stp.NB_SPLIT,

            "delta_angle"     : stp.DELTA_ANGLE,

            "fiber_perimeter_min"   : stp.FIBRE_PERIMETER_MIN,  
            "fiber_len_min"         : stp.FIBER_LEN_MIN,  
            "fiber_width_max"       : stp.FIBER_WIDTH_MAX,  
            "fiber_ratio_min"       : stp.FIBER_RATIO_MIN, 

            "blur_method"     : stp.BLUR_STR,
            "thresh_method"   : stp.THRESH_STR,  

            "bilateral_d"           : stp.BILATERAL_D,          
            "bilateral_sigma_color" : stp.BILATERAL_SIGMA_COLOR, 
            "bilateral_sigma_space" : stp.BILATERAL_SIGMA_SPACE,

            "tresh_type"            : stp.THRESH_TYPE_STR,

            "region_min_fiber": stp.REGION_MIN_FIBER,
            "region_min_area" : stp.REGION_MIN_AREA, 
            "hole_min_area"   : stp.HOLE_MIN_AREA, 

            "dbscan_eps"         : stp.DBSCAN_EPS,
            "dbscan_min_samples" : stp.DBSCAN_MIN_SAMPLES,

            "shape_method"    : stp.SHAPE_STR,
            
            "r_dilate"        : stp.R_DILATE,
            "r_erode"         : stp.R_ERODE,

            "alpha"           : stp.ALPHA,
            "point_step"      : stp.POINT_STEP,

            "render"          : stp.RENDER_STR
        }
        
        #------------------------------
        with open(self.config, "w", encoding="utf-8") as f:

            final_json = {
                "description": self.name + " Config",
                "params": params,
                "colors": color_config
            }

            json.dump(final_json, f, indent=4)

    #================================================================================#
    def render(self, n_render : int = stp.RENDER_FIBER, render_splits : bool = False) -> np.ndarray:

        """
        render of all the regions in self.regions
        create a file for each region and return an image containing all th contours 

        n_render : indicates the desired rendering type 
        render_split : if true do the render on each spilt of self.splits
        """
        
        #------------------------------
        if render_splits:

            for split in tqdm(self.splits, desc="Rendering            ", unit="img"):
                split.save(self.config)
        
        #------------------------------
        else:

            if(stp.BACKGROUND == -1):
                regions_img = cv.cvtColor(self.img, cv.COLOR_GRAY2RGB)
            else:
                regions_img = np.ones_like(cv.cvtColor(self.img, cv.COLOR_GRAY2RGB)) * stp.BACKGROUND

            #---------------
            for reg in tqdm(self.regions, desc="Rendering regions     ", unit="img") :

                if(stp.BACKGROUND == -1):
                    reg_img = cv.cvtColor(self.img, cv.COLOR_GRAY2RGB)
                else:
                    reg_img = np.ones_like(cv.cvtColor(self.img, cv.COLOR_GRAY2RGB)) * stp.BACKGROUND

                regions_img = reg.render(regions_img, n_config_path=self.config, render_type=n_render)   
                reg_img     = reg.render(reg_img, n_config_path=self.config, render_type=n_render)   

                cv.imwrite(self.regions_path + self.name + "_region_" + str(int(reg.mean_angle)) + ".png", reg_img)

            return regions_img
        
    #================================================================================#
    def print(self, region=False):

        """
        print all the informations about a Sample 

        region : if true print the information about all the Regions in self.regions[]
        """
        
        #------------------------------
        if(region):
            print(f"\n#=================== SAMPLE {self.id} ====================#\n")

            for reg_data in self.regions_data :
                print(reg_data)

            print(f"#==================================================#\n")

        else:
            print(f"\n#=================== SAMPLE {self.id} ====================#")
            print(f"name            : {self.name}")
            print(f"path            : {self.img_path}")
            
            print(f"splits_path     : {self.splits_path}")
            print(f"regions_path    : {self.regions_path}")

            print(f"before fret     : {self.fretting}\n")

            print(f"nb split        = {self.nb_split}")
            print(f"(row, col)      = ({self.row}, {self.col})")
            print(f"splits.len      = {len(self.splits)}")

            print(f"regions.len     = {len(self.regions)}")
            print(f"main angles     = {self.main_angles}")
            print(f"#==================================================#\n")

    #================================================================================#
    def save(self):
        
        """
        save all the Region in self.regions[]
        """

        #------------------------------
        with tqdm(total=len(self.regions), desc="Export ROI               ", unit="reg") as pbar:
                                                
            for reg in self.regions:
                
                if reg.shape == None:

                    pbar.update(1)
                    continue
                
                reg.save(n_regions_path=self.regions_path)
                pbar.update(1)

        self.get_statistics()
        
#============================================================================================================================#
#------------------------------------------------------ STATIC METHODS ------------------------------------------------------#
#============================================================================================================================#
def init(n_id : int, n_split : int = 8, n_fret = False) -> Sample:

    """
    initialisation of a Sample

    n_split : indicates the desired number of splits
    n_fret  : indicates whether the study is before of after thr fretting
    """
    
    #------------------------------
    sample = Sample(n_id, n_split=n_split)
    sample.set_path(n_fret=n_fret)

    #------------------------------
    if(os.path.exists(sample.output_path)):
        tools.clear_folder(sample.output_path)
    else:
        os.makedirs(sample.output_path, exist_ok=True)

    #------------------------------
    print("\n")

    sample.load_img()
    sample.split()

    sample.tresh_img(blur_method=stp.BLUR_METHOD,
                     thresh_method=stp.THRESH_METHOD)

    os.makedirs(sample.regions_path, exist_ok=True)

    return sample