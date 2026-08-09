"""
Script        : main.py
Description   : Image analysis for detecting fibers in samples of a composite material called HexTool.
                The script detects the fibres in a sample of this specific material
                and then extracts the regions where the fibres have the same orientation. 

Author        : WEIBEL Hugo, MA Oscar
Date          : 01/01/2025
Version       : 1.0.0
"""

__author__  = "WEIBEL Hugo, MA Oscar"
__copyright__ = "Copyright 2025, Hugo WEIBEL & Oscar MA"
__credits__ = ["M. FORTE DA CRUZ", "MME. LO FUEDO"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "WEIBEL Hugo, MA Oscar"
__email__   = ["hugo03.weibel@gmail.com", "oscarma230@email.com"]
__status__  = "Development" 

#============================================================================================================================#
#---------------------------------------------------------- IMPORT ----------------------------------------------------------#
#============================================================================================================================#
import time
import cv2 as cv
import numpy as np

import Sample
import setup as stp
import tools


#============================================================================================================================#
#---------------------------------------------------------- MAIN ------------------------------------------------------------#
#============================================================================================================================#
config = tools.arg_parse()

START = time.perf_counter()

sample = Sample.init(config_path=config, n_fret=False)
sample.process_splits()
sample.save_config()

sample.group_regions()
sample.compute_shapes()

sample.save()
sample.print(region=False)

regions_img = sample.render(n_render=stp.RENDER)
cv.imwrite(sample.output_path + sample.name + "_regions_all.png", regions_img)

END = time.perf_counter()

executio_time = END - START
print(f"\nExecution time : {executio_time:.4f} seconds")

#------------------------------
# sample_cpy  = sample.copy()
# regions     = sample_cpy.get_roi(n_regions_path="./data/sample_25/before_fretting/RoiSet_p25_pre/")

# img_col = cv.cvtColor(sample.img, cv.COLOR_GRAY2RGB)
# cv.drawContours(image=img_col, contours=regions, contourIdx=-1, color=(0, 255, 0), thickness=25)
# cv.imwrite("./output/2022/result_2022_regions.png", img_col)

print("\n")