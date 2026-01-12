"""
Nom du Script : main.py
Description   : Image analysis for detecting fibers in samples of a composite material called HexTool.
                The script detects the fibres in a sample of this specific material
                and then extracts the regions where the fibres have the same orientation. 

Auteur        : 
Date          : 01/12/2025
Version       : 1.0.0
"""

__author__  = ""
__copyright__ = "Copyright 2025, "
__credits__ = ["", ""]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = ""
__email__   = ["", ""]
__status__  = "Development" 

#----------------------------------------------------------------------------------------------#
#------------------------------------------- IMPORT -------------------------------------------#
#----------------------------------------------------------------------------------------------#
import cv2 as cv

import Sample
import Region 
import Fiber

import tools
import setup as stp


#----------------------------------------------------------------------------------------------#
#-------------------------------------------- MAIN --------------------------------------------#
#----------------------------------------------------------------------------------------------#
tools.color_config()

sample = Sample.init(stp.SAMPLE_INDEX, n_split=stp.NB_SPLIT)
sample.print()

#---------------
row = 1
nb_split = row * sample.col

for split_index in range(nb_split):

    print(f"Process split {split_index+1}/{nb_split} ...", end="\r")

    thresh_img_path = sample.thresh_path + stp.THRESH_ + str(split_index) + stp.OUTPUT_EXTENSION   

    fibers          = Fiber.detect_fibers(thresh_img_path)
    sorted_fibers   = Fiber.sort_fibers(fibers)
    
    # split_img_path = sample.split_path + stp.SPLIT_ + str(split_index) + stp.OUTPUT_EXTENSION
    # split_img = cv.imread(split_img_path, cv.IMREAD_COLOR)

    print(f"Process split {split_index+1}/{nb_split} *..", end="\r")

    #---------------
    for i in range(len(sorted_fibers)):

        # img_region = cv.imread(split_img_path, cv.IMREAD_COLOR)

        reg_i = Region.Region(fibers = sorted_fibers[i], 
                              n_split_index = split_index, 
                              sample_regions_path = sample.regions_path)
        
        print(f"Process split {split_index+1}/{nb_split} **.", end="\r")

        # reg_i.save(split_img,img_region, drawing_method=stp.DRAW_SHAPE)

        sample.save_region(reg_i)

    # split_img_path = sample.regions_path  + stp.SPLIT_ + str(split_index) + "/" + stp.SPLIT_ + str(split_index) +"_all" + stp.OUTPUT_EXTENSION
    # cv.imwrite(split_img_path, split_img)

    print(f"Process split {split_index+1}/{nb_split} ***", end="\r")

# print("\n\nReconstruction ...", end="\r")
# sample.join(n_row=row)
# print("Reconstruction ... Done\n")   


    