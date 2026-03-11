# --------------- PSYN --------------- #

PSYN stand for Projet de SYNthèse is an academic project which is based on image analysis for application on composites materials.
The studied material is called HexTOOL, it's made by Hexcel Composites​. 
This material exhibits certain local variations in the orientation of its fibres, so the aim of the project is to automate the delineation of regions of interest (ROI) contained in the different samples.

# -------------- Usage --------------- #

## --- Installation --- ##
We recommand to create a python virtual environment in order to use the script. ([text](https://docs.python.org/fr/3.9/library/venv.html))
The sample's directories need to named as sample_id (exemple : sample_25) and located in ./data/.
Moreover this project requires a lot of different library which are listed in requirement.txt

- python -m venv my_venv

- ./my_venv/Script/activate

- pip install -r requirements.txt

- python ./src/main.py

## --- Configuration file --- ##

In the ./config/ directory the config.json file is the input files which defines the main parameters :

— sample_index : designates the index of the studied sample.

— nb_split : number of image tiles (splits).

— delta_angle : angular tolerance interval for considering two angles as equal.

— blur_method : blurring method used (BILATERAL, GAUSSIAN).

— thresh_method : thresholding method used (CLASSIC, ADAPTIVE).

— fiber_perimeter_min : minimum perimeter.

— fiber_len_min : minimum length.

— fiber_width_max : maximum width.

— fiber_ratio_min : minimum aspect ratio.

— region_min_fiber : minimum number of fibers required to consider a region as valid.

— region_min_area : minimum area (in pixels) for a region to be valid.

— hole_min_area : minimum area (in pixels) for a "hole" to be considered valid.

— shape_method : boundary detection method used (MORPH, A_SHAPE).

— r_dilate : dilation radius for morphological closing.

— r_erode : erosion radius for morphological closing.

— alpha : defines the $\alpha$ parameter for the alpha shape algorithm.

— point_step : point step size used for the alpha shape algorithm.

— render : type of rendering output.

— fill_shape : toggle to fill regions (true/false).

— shape_thickness : line thickness for boundaries.

— fiber_thickness : line thickness for fiber contours.

— background : defines the rendering background (black, white, real).

# -------------- TO DO --------------- #

- Debug holes render
- fix .zip/.roi files nomenclature 
- Create a Frett Class to describe the tribological expirements

