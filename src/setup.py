import json
import cv2 as cv


#================================================================================#
def get_config(config_path : str = "./config/config.json"):

    """
    set up all the necessary parameters from configuration file

    config_path : path to the wanted configuration file
    """
    
    #------------------------------
    global SAMPLE_INDEX, NB_SPLIT, BLUR_METHOD, THRESH_METHOD, SHAPE_METHOD
    global FIBRE_PERIMETER_MIN, FIBER_LEN_MIN, FIBER_WIDTH_MAX, FIBER_RATIO_MIN
    global REGION_MIN_FIBER, REGION_MIN_AREA, HOLE_MIN_AREA
    global DELTA_ANGLE, R_DILATE, R_ERODE, ALPHA, POINT_STEP
    global RENDER, FILL_SHAPE, SHAPE_THICKNESS, FIB_THICHNESS, BACKGROUND
    global THRESH_STR, BLUR_STR, SHAPE_STR, RENDER_STR

    with open(config_path, "r") as config_file:

        config = json.load(config_file)
        params = config.get("compute_params", [])
        render_params = config.get("render_params", [])

        #------------------------------
        SAMPLE_INDEX    = params["sample_index"]
        NB_SPLIT        = params["nb_split"]

        #---------------
        if(params["blur_method"] == "GAUSSIAN"):
            BLUR_METHOD = GAUSSIAN_BLUR
            BLUR_STR = "GAUSSIAN"

        elif(params["blur_method"] == "BILATERAL"):
            BLUR_METHOD = BILATERAL_BLUR
            BLUR_STR = "BILATERAL"

        else :
            BLUR_METHOD = BILATERAL_BLUR
            BLUR_STR = "BILATERAL"

        #---------------
        if(params["thresh_method"] == "CLASSIC"):
            THRESH_METHOD = CLASSIC_THRESH
            THRESH_STR = "CLASSIC"

        elif(params["thresh_method"] == "ADAPTATIVE"):
            THRESH_METHOD = ADAPTATIVE_THRESH
            THRESH_STR = "ADAPTATIVE"

        else : 
            THRESH_METHOD = CLASSIC_THRESH
            THRESH_STR = "CLASSIC"

        #---------------
        if(params["shape_method"] == "MORPH"):
            SHAPE_METHOD = MORPH
            SHAPE_STR = "MORPH"

        elif(params["shape_method"] == "A_SHAPE"):
            SHAPE_METHOD = A_SHAPE 
            SHAPE_STR = "A_SHAPE"

        else:
            SHAPE_METHOD = MORPH
            SHAPE_STR = "MORPH"
        
        #---------------
        FIBRE_PERIMETER_MIN = params["fiber_perimeter_min"]
        FIBER_LEN_MIN       = params["fiber_len_min"]
        FIBER_WIDTH_MAX     = params["fiber_width_max"]
        FIBER_RATIO_MIN     = params["fiber_ratio_min"]

        REGION_MIN_FIBER    = params["region_min_fiber"]
        REGION_MIN_AREA     = params["region_min_area"]
        HOLE_MIN_AREA       = params["hole_min_area"]

        #---------------
        DELTA_ANGLE         = params["delta_angle"]

        R_DILATE            = params["r_dilate"] 
        R_ERODE             = -params["r_erode"] 

        ALPHA               = params["alpha"] 
        POINT_STEP          = params["point_step"] 

        #------------------------------
        if(render_params["render"] == "FIBER"):
            RENDER = RENDER_FIBER["id"]
            RENDER_STR = "FIBER"

        elif(render_params["render"] == "SHAPE"):
            RENDER = RENDER_SHAPE["id"]
            RENDER_STR = "SHAPE"

        elif(render_params["render"] == "FIBER_SHAPE"):
            RENDER = RENDER_FIBER_SHAPE["id"]
            RENDER_STR = "FIBER_SHAPE"

        else : 
            RENDER = RENDER_SHAPE["id"]
            RENDER_STR = "SHAPE"

        #---------------
        if render_params["fill_shape"] : 
            SHAPE_THICKNESS = -1 
        else :  
            SHAPE_THICKNESS = render_params["shape_thickness"]

        FIB_THICHNESS = render_params["fiber_thickness"]

        #---------------    
        
        if(isinstance(render_params["background"], int)):
            BACKGROUND = render_params["background"]

        else:

            if(render_params["background"] in ("black" or "b")):
                BACKGROUND = 0
            
            elif(render_params["background"] in ("white" or "w")):
                BACKGROUND = 255
        
            elif(render_params["background"] in ("real" or "r")):
                BACKGROUND = -1
        
            else:
                BACKGROUND = 0

#============================================================================================================================#
#----------------------------------------------------------- PATH -----------------------------------------------------------#
#============================================================================================================================#
SAMPLE_INDEX     = "25"

DATA_PATH        = "./data/"
OUTPUT_PATH      = "./output/"
CONFIG_PATH      = "./config/"

SPLIT_PATH       = "split_images/"
THRESH_PATH      = "thresh_images/"
REGION_PATH      = "regions_images/"
RECON_PATH       = "recon_images/"

FIBER_           = "_fiber"
SHAPE_           = "_shape"
FIBER_SHAPE_     = "_fiber_shape"

SPLIT_           = "split_"
THRESH_          = "thresh_"
REGION_          = "region_"

OUTPUT_EXTENSION = ".png"

#============================================================================================================================#
#--------------------------------------------------------- CONSTANT ---------------------------------------------------------#
#============================================================================================================================#
THRESH_STR          = "CLASSIC"
BLUR_STR            = "BILATERAL"
SHAPE_STR           = "MORPH"
RENDER_STR          = "FIBER"

#------------------------------
MIN_ANGLE           = 0
MAX_ANGLE           = 180
MAX_ANGLE_DEV       = 45

GAUSSIAN_BLUR       = 0
BILATERAL_BLUR      = 1

CLASSIC_THRESH      = 0
ADAPTATIVE_THRESH   = 1

MORPH               = 0
A_SHAPE             = 1

DRAW_FIBER          = 1
DRAW_SHAPE          = 2

#============================================================================================================================#
#---------------------------------------------------------- PARAMS ----------------------------------------------------------#
#============================================================================================================================#
NB_SPLIT            = 64

FIBRE_PERIMETER_MIN = 10
FIBER_LEN_MIN       = 20
FIBER_WIDTH_MAX     = 20
FIBER_RATIO_MIN     = 2

REGION_MIN_FIBER    = 100
REGION_MIN_AREA     = 300000
HOLE_MIN_AREA       = 25000

DELTA_ANGLE         = 8.85

DBSCAN_EPS          = 100
DBSCAN_MIN_SAMPLES  = 5

SHAPE_METHOD        = MORPH

R_DIFF              = 40
R_DILATE            = 400
R_ERODE             = -(R_DILATE - R_DIFF)  

ALPHA               = 0.0065
POINT_STEP          = 200

KERNEL_SIZE         = (9, 9)

#------------------------ BLUR ------------------------#
BLUR_METHOD           = BILATERAL_BLUR

BILATERAL_D           = 9   
BILATERAL_SIGMA_COLOR = 50  
BILATERAL_SIGMA_SPACE = 25

#----------------------- THRESH -----------------------#
THRESH_METHOD       = CLASSIC_THRESH

TH_MIN              = 100
TH_MAX              = 255
THRESH_TYPE         = cv.THRESH_BINARY + cv.THRESH_OTSU
THRESH_TYPE_STR     = "cv.THRESH_BINARY + cv.THRESH_OTSU"

K_SIZE              = 9
MAX                 = 255
C                   = 3

#============================================================================================================================#
#---------------------------------------------------------- RENDER ----------------------------------------------------------#
#============================================================================================================================#
FILL_SHAPE = False
if FILL_SHAPE : 
    SHAPE_THICKNESS = -1 
else :  
    SHAPE_THICKNESS = 25

FIB_THICHNESS = 3
BACKGROUND    = 0

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

RENDER = RENDER_SHAPE