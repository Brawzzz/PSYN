#----------------------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------- IMPORT ----------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------------#
import os 
import cv2 as cv
import numpy as np

from typing import TYPE_CHECKING 
if TYPE_CHECKING:
    import Region

import setup as stp


#----------------------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------- CLASS -----------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------------#
class Split:

    def __init__(self, n_id : int, n_origin : tuple[int, int], sample_path : str):
        
        self.id     = n_id
        self.prefix = stp.SPLIT_ + str(self.id)

        self.origin = n_origin

        self.regions : list['Region.Region'] = []

        self.render = stp.RENDER_FIBER

        self.dir_path       = sample_path + self.prefix + "/" 
        self.regions_dir    = self.dir_path + self.prefix + "_regions" + "/"

        (self.img_path, self.blur_path, self.thresh_path) = self.set_path() 
        
        os.makedirs(self.dir_path, exist_ok=True)
        os.makedirs(self.regions_dir, exist_ok=True)

    #================================================================================#
    def set_path(self):
        
        img_path       = self.dir_path + self.prefix + "_img" + stp.OUTPUT_EXTENSION 
        blur_path      = self.dir_path + self.prefix + "_blur" + stp.OUTPUT_EXTENSION
        thresh_path    = self.dir_path + self.prefix + "_thresh" + stp.OUTPUT_EXTENSION

        return(img_path, blur_path, thresh_path)
    
    #================================================================================#
    def set_render(self, n_render_type : dict):
        self.img_path = n_render_type
    
    #================================================================================#
    def print(self):

        print(f"#========== SPLIT {self.id} ==========#")

        print(f"nb regions        = {len(self.regions)}\n")

        print(f"render            = {self.render}\n")

        print(f"directory path    : {self.dir_path}\n")
        print(f"regions directory : {self.regions_dir}")

        print(f"image path        : {self.img_path}")
        print(f"thresh path       : {self.thresh_path}")
        
        print(f"#================================#\n")

    #================================================================================#
    def save(self, color_config_path : str):

        if(not os.path.exists(color_config_path)):
            raise ValueError(f"path do not exist : {color_config_path}")
        
        render = self.render["id"]
        all_img = cv.imread(self.img_path, cv.IMREAD_COLOR_BGR)

        for reg in self.regions:

            region_img = cv.imread(self.img_path, cv.IMREAD_COLOR_BGR)

            reg.render(region_img, 
                       color_config_path=color_config_path, 
                       render_type=render)
            
            reg.render(all_img, 
                       color_config_path=color_config_path,
                       render_type=render)

            region_render_path = os.path.join(self.regions_dir, self.prefix + "_" + reg.name + stp.OUTPUT_EXTENSION) 
            cv.imwrite(region_render_path, region_img)

        all_render_path = os.path.join(self.regions_dir, self.prefix + "_all" + stp.OUTPUT_EXTENSION)
        cv.imwrite(all_render_path, all_img)

#--------------------------------------------------------------------------------#
if __name__ == "__main__":

    id = 0
    sample_path = "./output/hxtl_p25_pre/"

    split = Split(n_id=id, sample_path=sample_path)

    split.print()