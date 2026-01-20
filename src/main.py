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
import Sample

import tools
import setup as stp


#----------------------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------- MAIN ------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------------#
# tools.color_config()

sample = Sample.init(stp.SAMPLE_INDEX, n_split=stp.NB_SPLIT)
sample.print()

sample.process_regions()

sample.render()
sample.join()

print("\n")
    