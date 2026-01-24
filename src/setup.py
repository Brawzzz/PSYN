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

MIN_ANGLE       = 0
MAX_ANGLE       = 180
MAX_ANGLE_DEV   = 45
DELTA_ANGLE     = 8

CNTRS_LEN_MIN   = 10

FIBER_LEN_MIN   = 2
FIBER_WIDTH_MAX = 30
FIBER_RATIO_MIN = 3

REGION_MIN_SIZE = 20

#----------------------------- BLUR -----------------------------#
GAUSSIAN_BLUR   = 1
BILATERAL_BLUR  = 2

KERNEL_SIZE     = (9, 9)

BILATERAL_D           = 9   
BILATERAL_SIGMA_COLOR = 50  
BILATERAL_SIGMA_SPACE = 25

#---------------------------- THRESH ----------------------------#
CLASSIC_THRESH      = 0
TH_MIN              = 100
TH_MAX              = 255
THRESH_TYPE         = cv.THRESH_BINARY + cv.THRESH_OTSU

ADAPTATIVE_THRESH   = 1
K_SIZE              = 9
MAX                 = 255
C                   = 3

#---------------------------- DBSCAN ----------------------------#
DBSCAN_EPS          = 100
DBSCAN_MIN_SAMPLES  = 5

#---------------------------- A-SHAPE ---------------------------#
ALPHA = 0.005

#---------------------------- RENDER ----------------------------#
DRAW_FIBER          = 1
DRAW_SHAPE          = 2

FILL_SHAPE = True
if FILL_SHAPE : 
    SHAPE_THICKNESS = -1 
else :  
    SHAPE_THICKNESS = 8

FIB_THICHNESS = 3

RENDER_FIBER = {

    "id"        : DRAW_FIBER,
    "suffix"    : "_fiber",
    "fib_thick" : FIB_THICHNESS
}

RENDER_SHAPE = {

    "id"          : DRAW_SHAPE,
    "suffix"      : "_shape",
    "shape_thick" : SHAPE_THICKNESS
}

RENDER_FIBER_SHAPE = {

    "id"            : DRAW_FIBER + DRAW_SHAPE,
    "suffix"        : "_fiber_shape",
    "fib_thick"     : FIB_THICHNESS,
    "shape_thick"   : SHAPE_THICKNESS
}