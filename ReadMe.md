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

In the ./config/ directory the config.json file is the input files fo definig the main parameters :

- sample_index : the id of the sample we want to study

- nb_split : number of split

- blur_method : the wanted method for blur operation 
- thresh_method : the wanted method for thresh operation 

- shape_method : the wanted method for region delineation

- main fiber attributes :

    - fiber_perimeter_min
    - fiber_len_min
    - fiber_width_max 
    - fiber_ratio_min  

- render : the wanted type of render for region 

# -------------- TO DO --------------- #

- Debug holes render
- fix .zip/.roi files nomenclature 
- Create a Frett Class to describe the tribological expirements

