#----------------------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------- IMPORT ----------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------------#
import numpy as np
import cv2 as cv

import tools
import setup as stp

#----------------------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------- CLASS -----------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------------#
class Fiber:

    def __init__(self, cnt : list):
        
        self.contour        = cnt
        self.perimeter      = len(self.contour)

        self.oriented_box   = self.compute_oriented_box()

        self.angle          = self.compute_angle()
        self.position       = self.compute_position()

        self.length         = self.compute_length()
        self.width          = self.compute_width()

        self.ratio = (self.length / self.width) if (self.width and self.width > 0) else 0.0
        
    #--------------------------------------------------------------------------------#
    def compute_oriented_box(self):

        if(self.perimeter > 0):
            return(cv.minAreaRect(self.contour))

    #--------------------------------------------------------------------------------#
    def compute_angle(self):
        
        if(self.oriented_box):
            return(fiber_angle(self.oriented_box))

    #--------------------------------------------------------------------------------#
    def compute_position(self):

        if(self.oriented_box):
            return(self.oriented_box[0])
    
    #--------------------------------------------------------------------------------#
    def compute_length(self):

        if(self.oriented_box):
            (w, h) = self.oriented_box[1]
            return(max(w, h))
    
    #--------------------------------------------------------------------------------#
    def compute_width(self):

        if(self.oriented_box):
            (w, h) = self.oriented_box[1]
            return(min(w, h))
        
    #--------------------------------------------------------------------------------#
    def valid_fiber(self) -> bool:

        if(self.perimeter   > stp.CNTRS_LEN_MIN 
           and self.length  >= stp.FIBER_LEN_MIN 
           and self.width   <= stp.FIBER_WIDTH_MAX 
           and self.ratio   > stp.FIBER_RATIO_MIN):
            return True
        
        return False
    
    #--------------------------------------------------------------------------------#
    def render(self, img : np.ndarray, reg_angle : float = None) :

        if(tools.img_empty(img)):
            raise ValueError(f"render() Fiber.py line 63 : img is empty")
        
        if(reg_angle):
            color = tools.angle_to_color(reg_angle)
        else:
            color = tools.angle_to_color(self.angle)

        cv.drawContours(image=img, contours=[self.contour], 
                        contourIdx=-1, color=color, 
                        thickness=stp.FIB_THICHNESS, lineType=cv.LINE_AA)

    #--------------------------------------------------------------------------------#
    def print(self):

        print(f"angle           = {self.angle}")
        print(f"position        = {self.position}")
        print(f"length          = {self.length}")
        print(f"oriented_box    = {self.oriented_box}")
        print(f"perimeter       = {self.perimeter}")
        print("\n")

#----------------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------- STATICS METHODS ------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------------#
@staticmethod
def fiber_angle(rect):

    (_, (w, h), angle) = rect
    
    if w > h:
        real_angle = angle 
    else:
        real_angle = angle + 90

    return real_angle % 180

#--------------------------------------------------------------------------------#
@staticmethod
def select_fiber(fibers : list[Fiber]) -> list[Fiber]:

    selected_fibers = []
    for fib in fibers:
        
        if(fib.valid_fiber()):
            selected_fibers.append(fib)
    
    return selected_fibers

#--------------------------------------------------------------------------------#
@staticmethod
def detect_fibers(thresh_img_path : str) -> list[Fiber] : 

    thresh_img      = cv.imread(thresh_img_path, cv.IMREAD_GRAYSCALE) 
    (contours, _)   = cv.findContours(thresh_img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    fibers = []
    for cnt in contours:
        
        fib = Fiber(cnt)
        fibers.append(fib)

    fibers = select_fiber(fibers)

    return fibers

#--------------------------------------------------------------------------------#
@staticmethod
def group_fibers(fibers : list[Fiber]) -> list[list[Fiber]]:

    if not fibers:
        return []
    
    angles      = np.array([fib.angle for fib in fibers])
    peaks_index = tools.get_peaks(angles)

    #---------------
    sorted_groups = {idx : [] for idx in peaks_index}
    
    for fib in fibers:

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

    #---------------    
    result = []
    for peak in sorted_groups:
        group = sorted_groups[peak]

        if len(group) > stp.REGION_MIN_SIZE:
            result.append(group)

    return result
