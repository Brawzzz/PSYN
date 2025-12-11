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

#--------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------- CONSTANT -----------------------------------------------------#
#--------------------------------------------------------------------------------------------------------------------#
RED     = (0, 0, 255)
GREEN   = (0, 255, 0)
BLUE    = (255, 0, 0)

#---------- PARAMS   ----------#
AUTO_DETECTION = False

TH_MIN = 0
TH_MAX = 255

ALPHA_X = 10
ALPHA_Y = 10

if(AUTO_DETECTION):
    THLD_CONTOURS_LEN = 195
else:
    THLD_CONTOURS_LEN = 500

X0 = 2200
Y0 = 300
W_W = 2000
W_H = 800

RECT_SIZE_HORIZONTALE = 150
RECT_SIZE_VERTICALE = 90

MIN_AREA = 500
DELTA = 10
