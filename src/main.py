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

import setup as stp

from process import detect_fibers


#----------------------------------------------------------------------------------------------#
#-------------------------------------------- MAIN --------------------------------------------#
#----------------------------------------------------------------------------------------------#
sample = Sample.Sample(stp.SAMPLE_INDEX)
sample.set_path(n_bf=True)

print("\nLoading image .........", end="\r")
sample.load_img()
print("Loading image ........... Done")

print("Spliting image .......... ", end="\r")
sample.split(nb_split=8)
print("Spliting image .......... Done")

print("Thresholding images ..... ", end="\r")
sample.tresh_img()
print("Thresholding images ..... Done\n")

sample.print()

split_index = 0

fibers          = detect_fibers(sample, n_split_index=split_index)
sorted_fibers   = Fiber.sort_fibers(fibers)

print(f"Fiber type found : {len(sorted_fibers)}\n")

split_index_path = sample.split_path + stp.SPLIT + str(split_index) + stp.OUTPUT_EXTENSION
for i in range(len(sorted_fibers)):

    reg_i = Region.Region(fibers = sorted_fibers[i], 
                        n_split_index = split_index, 
                        sample_regions_path = sample.regions_path)
    
    reg_i.print()
    reg_i.save(split_index_path)
