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

— \texttt{sample_index} : designates the index of the studied sample.

— \texttt{nb_split} : number of image tiles (\emph{splits}).

— \texttt{delta_angle} : angular tolerance interval for considering two angles as equal.

— \texttt{blur_method} : blurring method used (\emph{BILATERAL}, \emph{GAUSSIAN}).

— \texttt{thresh_method} : thresholding method used (\emph{CLASSIC}, \emph{ADAPTIVE}).

— \texttt{fiber_perimeter_min} : minimum perimeter.

— \texttt{fiber_len_min} : minimum length.

— \texttt{fiber_width_max} : maximum width.

— \texttt{fiber_ratio_min} : minimum aspect ratio.

— \texttt{region_min_fiber} : minimum number of \emph{fibers} required to consider a \emph{region} as valid.

— \texttt{region_min_area} : minimum area (in pixels) for a \emph{region} to be valid.

— \texttt{hole_min_area} : minimum area (in pixels) for a "hole" to be considered valid.

— \texttt{shape_method} : boundary detection method used (\emph{MORPH}, \emph{A_SHAPE}).

— \texttt{r_dilate} : dilation radius for morphological closing.

— \texttt{r_erode} : erosion radius for morphological closing.

— \texttt{alpha} : defines the $\alpha$ parameter for the \emph{alpha shape} algorithm.

— \texttt{point_step} : point step size used for the \emph{alpha shape} algorithm.

— \texttt{render} : type of rendering output.

— \texttt{fill_shape} : toggle to fill \emph{regions} (\emph{true/false}).

— \texttt{shape_thickness} : line thickness for boundaries.

— \texttt{fiber_thickness} : line thickness for \emph{fiber} contours.

— \texttt{background} : defines the rendering background (\emph{black}, \emph{white}, \emph{real}).

# -------------- TO DO --------------- #

- Debug holes render
- fix .zip/.roi files nomenclature 
- Create a Frett Class to describe the tribological expirements

