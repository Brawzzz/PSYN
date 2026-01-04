import cv2 as cv


#----------------------------------------------------------------------------------------------#
#-------------------------------------------- PATH --------------------------------------------#
#----------------------------------------------------------------------------------------------#
SAMPLE_INDEX    = "25"

DATA_PATH       = "./data/"
OUTPUT_PATH     = "./output/"
CONFIG_PATH     = "./config/"

SPLIT_PATH      = "split_images/"
THRESH_PATH     = "thresh_images/"
REGION_PATH     = "regions_images/"
RECON_PATH      = "recon_images/"

FIBER_          = "_fiber"
SHAPE_          = "_shape"
FIBER_SHAPE_    = "_fiber_shape"

SPLIT_          = "split_"
THRESH_         = "thresh_"
REGION_         = "region_"

CONFIG_COLOR_PATH   = CONFIG_PATH + "color_config.json"

OUTPUT_EXTENSION    = ".png"

#----------------------------------------------------------------------------------------------#
#------------------------------------------ CONSTANT ------------------------------------------#
#----------------------------------------------------------------------------------------------#
RED             = (0, 0, 255)
GREEN           = (0, 255, 0)
BLUE            = (255, 0, 0)

#----------------------------------------------------------------------------------------------#
#------------------------------------------- PARAMS -------------------------------------------#
#----------------------------------------------------------------------------------------------#
NB_SPLIT        = 128
MIN_COL         = 12

MIN_ANGLE       = 0
MAX_ANGLE       = 180
DIFF_ANGLE      = 90
DELTA_ANGLE     = 8

CNTRS_LEN_MIN   = 10
FIBER_MIN_LEN   = 30
FIBER_MAX_WIDTH = 10

MIN_REGION_SIZE = 35

#--------------- BLUR ---------------#
KERNEL_SIZE     = (9, 9)
GAUSSIAN_BLUR   = 0
CLASSIC_BLUR    = 1

#-------------- THRESH --------------#
TH_MIN                      = 100
TH_MAX                      = 255
THRESH_TYPE                 = cv.THRESH_BINARY + cv.THRESH_OTSU
CLASSIC_THRESH_METHOD       = 0
ADAPTATIVE_THRESH_METHOD    = 1

#-------------- DBSCAN --------------#
DBSCAN_EPS          = 115
DBSCAN_MIN_SAMPLES  = 4

#-------------- A-SHAPE -------------#
ALPHA = 0.004

#-------------- RENDER --------------#
DRAW_FIBER          = 1
DRAW_SHAPE          = 2

FILL_SHAPE          = False
if FILL_SHAPE : 
    THICKNESS = -1 
else :  
    THICKNESS = 5