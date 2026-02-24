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

#============================================================================================================================#
#---------------------------------------------------------- IMPORT ----------------------------------------------------------#
#============================================================================================================================#
import cv2 as cv
import numpy as np

import setup as stp

import Sample
import Region

#============================================================================================================================#
#---------------------------------------------------------- MAIN ------------------------------------------------------------#
#============================================================================================================================#
sample = Sample.init(stp.SAMPLE_INDEX, n_split=stp.NB_SPLIT, n_fret=False)
sample.process_regions()
sample.set_config()

# sample.render()
# sample.join()

sample.group_regions()
sample.global_shape()

sample.save()
sample.print()

img = np.copy(cv.cvtColor(sample.img, cv.COLOR_GRAY2BGR))

regions_img = sample.render(img, n_render=stp.RENDER_SHAPE["id"])
cv.imwrite(sample.output_path + sample.name + "_regions_all.png", regions_img)

print("\n")