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

from process import detect_fibers


#----------------------------------------------------------------------------------------------#
#-------------------------------------------- MAIN --------------------------------------------#
#----------------------------------------------------------------------------------------------#

tools.make_color_config()

sample = Sample.init(stp.SAMPLE_INDEX, n_split=stp.NB_SPLIT)
sample.print()

nb_split = 10
for split_index in range(nb_split):

    print(f"\nFinding fibers in split {split_index} ...", end="\r")
    fibers          = detect_fibers(sample, n_split_index=split_index)
    sorted_fibers   = Fiber.sort_fibers(fibers)
    print(f"Finding fibers in split {split_index} ... Done")

    # print(f"Fiber type found : {len(sorted_fibers)}\n")

    invalid_fibers = []

    split_index_path = sample.split_path + stp.SPLIT_ + str(split_index) + stp.OUTPUT_EXTENSION
    for i in range(len(sorted_fibers)):

        reg_i = Region.Region(fibers = sorted_fibers[i], 
                            n_split_index = split_index, 
                            sample_regions_path = sample.regions_path)
        
        reg_i.print()
        reg_i.save(split_index_path, drawing_method=stp.DRAW_SHAPE)

    


    