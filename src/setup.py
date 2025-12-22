#--------------------------------------------------------------------------------------------------------------------#
# ------------------------------------------------------- PATH ------------------------------------------------------#
#--------------------------------------------------------------------------------------------------------------------#
DATA_PATH = "./data/"
OUTPUT_PATH = "./output/"

CNTRS_OUTPUT_PATH = OUTPUT_PATH + "countours/"

# ------------------------------ SAMPLE ------------------------------%
SAMPLE_INDEX = "25"
SELECTED_SAMPLE = DATA_PATH + "sample_" + SAMPLE_INDEX + "/"

AFTER_FRET = SELECTED_SAMPLE + "after_fretting/"
BEFORE_FRET = SELECTED_SAMPLE + "before_fretting/"
BIN_MASKS = SELECTED_SAMPLE + "binary_masks/"

SUFFIXE = "_pre"
IMG_EXTENSION = ".bmp"

IMG_NAME = "hxtl_p" + SAMPLE_INDEX + SUFFIXE + IMG_EXTENSION

# ----------------------------- DATA BASE ----------------------------%
SPLIT_PATH = OUTPUT_PATH + "split_images/"
RECON_PATH = OUTPUT_PATH + "recon_images/"

#--------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------- CONSTANT -----------------------------------------------------#
#--------------------------------------------------------------------------------------------------------------------#
RED     = (0, 0, 255)
GREEN   = (0, 255, 0)
BLUE    = (255, 0, 0)

#---------- PARAMS   ----------#
AUTO_DETECTION = False

TH_MIN = 118
TH_MAX = 255

MIN_AREA = 500
DELTA = 10

CNTRS_LEN_MIN = 25 
