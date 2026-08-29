#============================================================================================================================#
#---------------------------------------------------------- IMPORT ----------------------------------------------------------#
#============================================================================================================================#
import os
import glob
import json
import copy
import roifile
import pickle as pkl
import cv2 as cv
import numpy as np

from tqdm import tqdm
from datetime import datetime
from shapely import Point, Polygon, unary_union

import Fiber
import Region
import Shape
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

    :params n_id:       the id of the sample (exemple : 22, 23, 24, 25)
    :params img:        the image from the tribological study
    :params regions:    a list of Region instances from the image analysis
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
        self.data_path      = ""
        
        self.fretting       = None 
        self.params_config  = None
        self.saving_config  = "" 

        #------------------------------
        if(n_split % 2 != 0):
            if(n_split != 1):
                raise ValueError(f"__init__() Sample.py line 29 : nb_split must be even : nb_split = {n_split}")
        
        self.nb_split           = n_split
        (self.row, self.col)    = tools.compute_row_col(self.nb_split)
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

        :params n_fret: True if running after fretting
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
        self.data_path      = self.output_path + "data/"

        date_str            = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        config_suffix       = "_config_" + date_str + ".json" 
        self.saving_config  = os.path.join(self.output_path, self.name + config_suffix)

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
    def copy(self):
        return copy.deepcopy(self)
        
    #================================================================================#
    def split(self, save : bool = True):

        """
        split the sample in a list of Split object

        :params save: if true it saves the correspondind split's image otherwise it return a list containing all the splits images (default : true)
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
        Threshold each Splits, the function do both blur and thresh

        :params blur_method:    tell the wanted blur method (default : cv.GaussianBlur())
        :params thresh_method:  tell the wanted thresh method (default : cv.threshold())
        """

        #------------------------------

        STD_MIN            = 12                               
        global_blur        = cv.GaussianBlur(self.img, stp.KERNEL_SIZE, 0)
        (global_thresh, _) = cv.threshold(global_blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)


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

                    if img_blur.std() < STD_MIN:
                        (_, img_bw) = cv.threshold(img_blur, global_thresh, stp.TH_MAX, cv.THRESH_BINARY)
                        
                    else:
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
    def full_mask(self, heal_seams: bool = True, save: bool = True) -> np.ndarray:

        """
        Assemble binary threshold masks (split.thresh_path) into a
        single full-size mask
        Fibers can be detected once -- in global coordinates

        :params heal_seams:    light morphological close to reconnect a fiber that a threshold
        :params save:          if true it saves the full mask otherwise it return it (default : true)
        """

        #------------------------------
        if not self.splits:
            raise ValueError("full_mask() : self.splits is empty (call split() first)")

        if len(self.splits) % self.col != 0:
            raise ValueError(f"full_mask() : nb splits ({len(self.splits)}) not a multiple of col ({self.col})")

        #------------------------------
        strips = []
        for r in range(0, len(self.splits), self.col):

            row_tiles = []
            for split in self.splits[r : r + self.col]:

                tile = cv.imread(split.thresh_path, cv.IMREAD_GRAYSCALE)

                if tools.img_empty(tile):
                    raise ValueError(f"full_mask() : missing mask {split.thresh_path}")
                
                row_tiles.append(tile)

            strips.append(np.hstack(row_tiles))

        full_mask = np.vstack(strips)

        #------------------------------
        if heal_seams:
            kernel    = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
            full_mask = cv.morphologyEx(full_mask, cv.MORPH_CLOSE, kernel)

        if save:
            cv.imwrite(self.output_path + self.name + "_full_mask" + stp.OUTPUT_EXTENSION, full_mask)

        return full_mask

    #================================================================================#
    def process_sample(self):

        """
        This function process a sample by :
            - detecting every fiber (on the full reassembled mask)
            - computing group of fibers classified by orientation. 
        """

        #------------------------------
        full_mask = self.full_mask(heal_seams=False)
        
        fibers          = Fiber.detect_fibers(full_mask)
        sorted_fibers   = Fiber.group_fibers(fibers)

        #------------------------------
        self.regions = []
        for group in sorted_fibers:

            reg = Region.Region(n_fibers=group, n_split_index=-1)
            
            if reg.mean_angle != -1:
                self.regions.append(reg)

        #------------------------------
        angles           = [reg.mean_angle for reg in self.regions]
        self.main_angles = tools.get_peaks(angles, min_peak_height=0, sigma_smoth=2)

    #================================================================================#
    def group_regions(self) -> list[Region.Region]:

        """
        Merge regions of similar orientation into one region per main-angle peak.

        :return region: list of new regions
        """

        #------------------------------
        regions_group = {idx: [] for idx in self.main_angles}

        for reg in self.regions:

            best_peak = -1
            min_dist  = float('inf')

            for peak in self.main_angles:

                d1 = abs(peak - reg.mean_angle)
                d2 = stp.MAX_ANGLE - d1
                d  = min(d1, d2)

                if d < min_dist:
                    min_dist  = d
                    best_peak = peak

            if min_dist <= stp.DELTA_ANGLE:
                regions_group[best_peak].append(reg)

        #------------------------------
        regions = []
        with tqdm(total=len(regions_group), desc="Computing global regions ", unit="reg") as pbar:

            for peak, group in regions_group.items():

                if group:
                    region_peak = Region.merge_regions(group)   
                    if len(region_peak.fibers) > stp.REGION_MIN_FIBER:
                        regions.append(region_peak)

                pbar.update(1)

        self.regions = regions
        return regions

    #================================================================================#
    def resolve_overlaps(self):

        """
        Vérifie les chevauchements entre toutes les shapes.
        Rogne les shapes moins denses et met à jour les objets existants.
        """
        
        #------------------------------
        all_shapes_info = []

        for reg in self.regions:
            for shape in reg.shapes:

                all_shapes_info.append({"region": reg, "shape": shape})
        
        #------------------------------
        all_shapes_info.sort(key=lambda item: item["shape"].density, reverse=True)
        
        for reg in self.regions:
            reg.shapes  = []
            reg.area    = 0.0 
            
        global_mask = Polygon() 

        #------------------------------
        with tqdm(total=len(all_shapes_info), desc="Resolving overlaps   ", unit="shape") as pbar:
            
            for item in all_shapes_info:

                reg             = item["region"]
                current_shape   = item["shape"]
                poly            = current_shape.polygon
                
                #---------------
                if not global_mask.is_empty:
                    poly = poly.difference(global_mask).buffer(0) 
                
                #---------------
                if poly.is_empty:
                    pbar.update(1)
                    continue
                    
                #---------------
                valid_geoms = []

                if poly.geom_type == 'Polygon':
                    valid_geoms = [poly]

                elif poly.geom_type == 'MultiPolygon':
                    valid_geoms = [geom for geom in poly.geoms if geom.area >= stp.SHAPE_MIN_AREA]
                
                if not valid_geoms:
                    pbar.update(1)
                    continue

                #------------------------------
                first_geom      = valid_geoms[0]
                first_fibers    = [fib for fib in reg.fibers if first_geom.contains(Point(fib.position))]
                
                current_shape.update(first_geom, first_fibers)
                
                reg.shapes.append(current_shape)
                reg.area += current_shape.area
                global_mask = unary_union([global_mask, first_geom])
                
                #------------------------------
                if len(valid_geoms) > 1:

                    for extra_geom in valid_geoms[1:]:

                        extra_fibers = [fib for fib in reg.fibers if extra_geom.contains(Point(fib.position))]
                        
                        new_shape = Shape.Shape(extra_geom, extra_fibers, reg.mean_angle)
                        
                        reg.shapes.append(new_shape)
                        reg.area += new_shape.area
                        global_mask = unary_union([global_mask, extra_geom])
                
                pbar.update(1)

    #================================================================================#
    def compute_shapes(self) -> None:

        """
        compute the shapes attributes of each global regions in self.regions
        """
        
        #------------------------------
        with tqdm(total=len(self.regions), desc="Computing Regions boundaries  ", unit="reg") as pbar:

            for reg in self.regions:

                reg_boudaries  = reg.compute_boundaries(method=stp.SHAPE_METHOD) 
                reg.shapes     = Shape.set_shapes(reg_boudaries, reg.fibers, reg.mean_angle)
        
                for shape in reg.shapes :
                    reg.area += shape.polygon.area
                
                pbar.update(1)

        self.resolve_overlaps()

        return
    
    #================================================================================#
    def get_data(self, n_save=False, graph=False):

        """
        compute the data of each global regions in self.regions and save it in a csv file

        :params n_save: if true it saves the data in a csv file otherwise it return it (default : false)
        :params graph:  if true it display a graph of the data (default : false)
        """

        #------------------------------
        for reg in self.regions:

            reg_data = (reg.mean_angle, 
                        reg.nb_fibers, 
                        reg.mean_fibers_len,
                        reg.mean_fibers_width,
                        reg.area)
            
            self.regions_data.append(reg_data)

        #------------------------------
        with open(f"{self.data_path}regions_data_{stp.CONFIG}.csv", "w") as f:

            f.write("index,mean_angle, nb_fibers, mean_fibers_len, mean_fibers_width, area, centroid_x, centroid_y\n")
            for reg_data in self.regions_data:
                f.write(f"{int(reg_data[0])}, {reg_data[1]}, {reg_data[2]}, {reg_data[3]}, {reg_data[4]}\n")

        return(self.regions_data)
    
    #================================================================================#
    def get_roi(self, n_regions_path=None) -> list[np.ndarray]:

        """
        Return all the sample's regions as Numpy arrays formatted for OpenCV (N, 1, 2) in int32.
        The function load the .roi files from self.regions_path.

        :params n_regions_path: if not None, load the .roi files from this path instead of self.regions_path

        :return: list of Numpy arrays representing the regions
        """

        #------------------------------
        if n_regions_path is not None:
            roi_files = glob.glob(os.path.join(n_regions_path, "*.roi"))
        else:
            roi_files = glob.glob(os.path.join(self.regions_path, "*.roi"))

        roi_files = sorted(roi_files, key=tools.extract_number)

        if not roi_files:
            raise ValueError(f"get_roi() Sample.py : no .roi file found in {self.regions_path}")

        #------------------------------
        regions = []
        for roi_file in tqdm(roi_files, desc="Loading ROI           ", unit="reg"):
            
            #---------------
            if not os.path.exists(roi_file):
                raise ValueError(f"get_roi() Sample.py : {roi_file} does not exist")

            roi         = roifile.ImagejRoi.fromfile(roi_file)
            coords      = roi.coordinates()
            roi_array   = np.array(coords, dtype=np.int32).reshape((-1, 1, 2))

            regions.append(roi_array)

        self.regions = regions

        return regions

    #================================================================================#
    def save_config(self) -> None:

        """
        configuration file as an output summary of the params used for a particular run

        :params file name: sample.name + JJ_MM_AAAA_HH_MIN_S .json (exemple : hxtl_p25_pre_config_26_02_2026_13_51_51.json)
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

            "region_min_fiber"  : stp.REGION_MIN_FIBER,
            "shape_min_area"    : stp.SHAPE_MIN_AREA, 
            "hole_min_area"     : stp.HOLE_MIN_AREA, 

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
        with open(self.saving_config, "w", encoding="utf-8") as f:

            final_json = {
                "description": self.name + " config : " + self.params_config,
                "params": params,
                "colors": color_config
            }

            json.dump(final_json, f, indent=4)

    #================================================================================#
    def render(self, n_render : int = stp.RENDER_FIBER, render_splits : bool = False) -> np.ndarray:

        """
        render of all the regions in self.regions
        create a file for each region and return an image containing all th contours 

        :params n_render:       indicates the desired rendering type 
        :params render_splits:  if true do the render on each split of self.splits

        :return: image containing all the contours of the regions
        """
        
        #------------------------------
        if render_splits:

            for split in tqdm(self.splits, desc="Rendering            ", unit="img"):
                split.save(self.saving_config)
        
        #------------------------------
        else:

            if(stp.BACKGROUND == -1):
                all_regions_img = cv.cvtColor(self.img, cv.COLOR_GRAY2RGB)
            else:
                all_regions_img = np.ones_like(cv.cvtColor(self.img, cv.COLOR_GRAY2RGB)) * stp.BACKGROUND

            #---------------
            for reg in tqdm(self.regions, desc="Rendering regions     ", unit="img") :
                
                #----------
                if(stp.BACKGROUND == -1):
                    single_reg_img = cv.cvtColor(self.img, cv.COLOR_GRAY2RGB)
                else:
                    single_reg_img = np.ones_like(cv.cvtColor(self.img, cv.COLOR_GRAY2RGB)) * stp.BACKGROUND

                #----------
                all_regions_img = reg.render(all_regions_img, n_config_path=self.saving_config, render_type=n_render)   
                single_reg_img  = reg.render(single_reg_img, n_config_path=self.saving_config, render_type=n_render)   

                cv.imwrite(self.regions_path + self.name + "_region_" + str(int(reg.mean_angle)) + ".png", single_reg_img)

            return all_regions_img
        
    #================================================================================#
    def print(self, region=False):

        """
        print all the informations about a Sample 

        :params region: if true print the information about all the Regions in self.regions[]
        """
        
        #------------------------------
        if(region):
            print(f"\n#=================== SAMPLE {self.id} ====================#\n")
            print("mean_angle, nb_fibers, mean_fibers_len, mean_fibers_width, area, centroid_position\n")
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
    def save(self, n_regions_path=None) -> None:
        
        """
        save all the Region in self.regions[] in .roi format

        :params n_regions_path: if not None, save the .roi files in this path instead of self.regions_path
        """

        #------------------------------
        with tqdm(total=len(self.regions), desc="Export ROI               ", unit="reg") as pbar:
                                                
            for reg in self.regions:
                
                if reg.shapes == None:
                    pbar.update(1)
                    continue
                
                #---------------
                if n_regions_path is not None:
                    reg.save(n_regions_path=n_regions_path)
                else:
                    reg.save(n_regions_path=self.regions_path)
                
                pbar.update(1)

        #------------------------------
        print(f"Saving sample as pickle file")

        file_path = self.output_path + self.name + ".pkl"
        with open(file_path, 'wb') as file:
            pkl.dump(self, file)

        #---------------
        self.get_data(n_save=True)
        
    #================================================================================#
    @classmethod
    def load(cls, filepath : str = "./output/hxtl_p25_pre/hxtl_p25_pre.pkl") :

        """
        load a sample from a pickle file

        :params n_regions_path: if not None, save the .roi files in this path instead of self.regions_path
        """

        #------------------------------
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Le fichier {filepath} n'existe pas.")
            
        print(f"Loading sample from {filepath} ...", end="", flush=True)
        with open(filepath, 'rb') as file:
            loaded_sample = pkl.load(file)
        print(f"\r\rLoading sample from {filepath} ... Done")
        
        return loaded_sample
        
#============================================================================================================================#
#------------------------------------------------------ STATIC METHODS ------------------------------------------------------#
#============================================================================================================================#
def init(config_path=f"./config/config.json", n_fret = False) -> Sample:

    """
    initialisation of a Sample

    :params config_path:    path to the configuration file
    :params n_fret:         indicates whether the study is before or after fretting
    """

    #------------------------------
    stp.get_config(config_path=config_path)
    
    sample = Sample(stp.SAMPLE_INDEX, n_split=stp.NB_SPLIT)

    sample.params_config = config_path
    sample.set_path(n_fret=n_fret)

    #------------------------------
    if(os.path.exists(sample.output_path)):

        exception_files = [sample.data_path]
        tools.clear_folder(sample.output_path, except_files=exception_files)

    else:
        os.makedirs(sample.output_path, exist_ok=True)

    #------------------------------
    print("\n")

    sample.load_img()
    sample.split()

    sample.tresh_img(blur_method=stp.BLUR_METHOD,
                     thresh_method=stp.THRESH_METHOD)

    os.makedirs(sample.regions_path, exist_ok=True)
    os.makedirs(sample.data_path, exist_ok=True)

    return sample