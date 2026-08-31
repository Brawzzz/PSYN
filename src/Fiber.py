#============================================================================================================================#
#---------------------------------------------------------- IMPORT ----------------------------------------------------------#
#============================================================================================================================#
import tqdm
import numpy as np
import cv2 as cv

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import Split

import tools
import setup as stp


#============================================================================================================================#
#---------------------------------------------------------- CLASS -----------------------------------------------------------#
#============================================================================================================================#
class Fiber:

    """
    The Fiber class describe the fibers contained in the materail HexTool
    It is defined mainly by : 

    Parameters
    ----------
    contour         : the contour of the corresponding fiber in the image
    oriented_box    : a bonding bow ofv the contours which is oriented
    angle           : the angle of the fiber (0-180°)  
    """

    #------------------------------
    def __init__(self, cnt : list):
        
        self.contour        = cnt
        self.perimeter      = len(self.contour)

        self.oriented_box   = self.compute_oriented_box()

        self.angle          = self.compute_angle()
        self.position       = self.compute_position()

        self.length         = self.compute_length()
        self.width          = self.compute_width()

        self.ratio = (self.length / self.width) if (self.width and self.width > 0) else 0.0
        
    #================================================================================#
    def compute_oriented_box(self):

        if(self.perimeter > 0):
            return(cv.minAreaRect(self.contour))

    #================================================================================#
    def compute_angle(self):
        
        if(self.oriented_box):
            return(fiber_angle(self.oriented_box))

    #================================================================================#
    def compute_position(self):

        if(self.oriented_box):
            return(self.oriented_box[0])
    
    #================================================================================#
    def compute_length(self):

        if(self.oriented_box):
            (w, h) = self.oriented_box[1]
            return(max(w, h))
    
    #================================================================================#
    def compute_width(self):

        if(self.oriented_box):
            (w, h) = self.oriented_box[1]
            return(min(w, h))
        
    #================================================================================#
    def valid_fiber(self) -> bool:

        """
        return True if the fiber is considerate as valid False otherwise
        """

        #------------------------------
        if(self.length >= stp.FIBER_LEN_MIN and self.width <= stp.FIBER_WIDTH_MAX and 
           self.ratio > stp.FIBER_RATIO_MIN and self.perimeter > stp.FIBRE_PERIMETER_MIN):
            return True
        
        return False
    
    #================================================================================#
    def map_fiber(self, split : 'Split.Split'):
        
        """
        allows to map fiber contour, pass the contour coordinates form the split basis to 
        the complete image basis

        Parameters
        ----------
        split : fiber's split
        """
        
        #------------------------------
        offset = np.array(split.origin, dtype=np.int32)
        mapped_cnt_array = self.contour + offset
        mapped_fiber = Fiber(mapped_cnt_array)

        return mapped_fiber

    #================================================================================#
    def render(self, img : np.ndarray, 
               n_config_path : str,
               reg_angle : float = None) -> None:

        """
        render the contours of the Fiber on img

        Parameters
        ----------
        img             : image on which to render the fiber
        n_config_path   : path of color configuration
        reg_angle       : angle of the fiber's region
        """

        #------------------------------
        if(tools.img_empty(img)):
            raise ValueError(f"render() Fiber.py line 63 : img is empty")
        
        if(reg_angle):
            color = tools.get_color(reg_angle, n_config_path=n_config_path)
        else:
            color = tools.angle_to_color(self.angle)

        cv.drawContours(image=img, contours=[self.contour], 
                        contourIdx=-1, color=color, 
                        thickness=stp.FIB_THICHNESS, lineType=cv.LINE_AA)

    #================================================================================#
    def print(self):
        
        """
        print all the data of a fiber
        """
        
        #------------------------------
        print(f"angle           = {self.angle}")
        print(f"position        = {self.position}")
        print(f"length          = {self.length}")
        print(f"oriented_box    = {self.oriented_box}")
        print(f"perimeter       = {self.perimeter}")
        print("\n")

#============================================================================================================================#
#----------------------------------------------------- STATICS METHODS ------------------------------------------------------#
#============================================================================================================================#
@staticmethod
def fiber_angle(rect) -> float:

    """
    return the orientation angle of a Fiber
    """
    
    (_, (w, h), angle) = rect
    
    if w > h:
        real_angle = angle 
    else:
        real_angle = angle + 90

    return real_angle % 180

#================================================================================#
@staticmethod
def select_fiber(n_fibers : list[Fiber]) -> list[Fiber]:

    """
    select ony the valid fiber in given list

    :params n_fibers: list of Fiber

    :return selected_fibers: list of fibers
    """
    
    #------------------------------
    selected_fibers = []
    for fib in n_fibers:
        
        if(fib.valid_fiber()):
            selected_fibers.append(fib)
    
    return selected_fibers

#================================================================================#
@staticmethod
def detect_fibers(thresh_img) -> list[Fiber]:

    """
    detect fibers in a threshold image

    :params thresh_img: path to the threshold image, or the mask array itself

    :return selected_fibers: list of fibers
    """

    #------------------------------
    if isinstance(thresh_img, str):
        thresh_img = cv.imread(thresh_img, cv.IMREAD_GRAYSCALE)

    if tools.img_empty(thresh_img):
        raise ValueError("detect_fibers() : empty threshold image")

    (contours, _) = cv.findContours(thresh_img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    fibers = [Fiber(cnt) for cnt in contours]

    return (select_fiber(fibers))

#================================================================================#
@staticmethod
def group_support(fibers : list[Fiber]) -> float:

    """
    Group's support : Total fibre length, in px.
    Measures the amount of material that is actually oriented, unaffected by fragmentation.
    """
    return sum(fib.length for fib in fibers)
    
#================================================================================#
@staticmethod
def group_fibers(n_fibers : list[Fiber]) -> list[list[Fiber]]:

    """
    create group of fiber with similar orientation

    :params n_fibers: list of Fiber
    """
    
    #------------------------------
    if not n_fibers:
        return []
    
    angles      = np.array([fib.angle for fib in n_fibers])
    peaks_index = tools.get_peaks(angles, min_peak_height=5, sigma_smoth=2.0)

    #------------------------------
    sorted_groups = {idx : [] for idx in peaks_index}

    with tqdm.tqdm(total=len(n_fibers), desc="Grouping fibers       ", unit="fib") as pbar:

        for fib in n_fibers:

            ang = fib.angle
            
            best_peak = -1
            min_dist = float('inf')

            for peak in peaks_index:
                
                d1 = abs(ang - peak)
                d2 = stp.MAX_ANGLE - d1
                d  = min(d1, d2)
                
                if d < min_dist:
                    min_dist = d
                    best_peak = peak
            
            if(min_dist <= stp.DELTA_ANGLE):
                sorted_groups[best_peak].append(fib)

            pbar.update(1)

    #------------------------------   
    result = []
    for peak in sorted_groups:

        group = sorted_groups[peak]

        # print(f"Group {peak} : {len(group)} fibers, support = {group_support(group)}")
        
        if group_support(group) > stp.REGION_MIN_SUPPORT:
            result.append(group)

    return result
