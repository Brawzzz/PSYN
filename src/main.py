"""
Script        : main.py
Description   : Image analysis for detecting fibers in samples of a composite material called HexTool.
                The script detects the fibres in a sample of this specific material
                and then extracts the regions where the fibres have the same orientation. 

Author        : 
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

#----------------------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------- IMPORT ----------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------------#
import os
import tools
import glob
import cv2 as cv
import numpy as np

from tqdm import tqdm

import setup as stp

import Sample
import Region

#----------------------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------- MAIN ------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------------#
sample = Sample.init(stp.SAMPLE_INDEX, n_split=stp.NB_SPLIT)

sample.process_regions()

sample.render()
sample.join()

sample.render_config()
sample.regions = sample.group_regions()

sample.print()

for i in range(len(sample.regions)):
    
    img = np.copy(cv.cvtColor(sample.img, cv.COLOR_GRAY2BGR))
    reg_0 = sample.regions[i]
    reg_0.render_shape(img)

    cv.imwrite(sample.output_path + "test_" + str(i) + "_.png", img)


print("\n")   

