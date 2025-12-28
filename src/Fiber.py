#----------------------------------------------------------------------------------------------#
#------------------------------------------- IMPORT -------------------------------------------#
#----------------------------------------------------------------------------------------------#
import numpy as np
import cv2 as cv
import math

import tools
import setup as stp


#----------------------------------------------------------------------------------------------#
#------------------------------------------- CLASS --------------------------------------------#
#----------------------------------------------------------------------------------------------#
class Fiber:

    def __init__(self, cnt : list):
        
        self.contour        = cnt
        self.oriented_box   = self.compute_oriented_box()
        self.angle          = self.compute_angle()
        self.position       = self.compute_position()
        self.length         = self.compute_length()
        self.width          = self.compute_width()
        
    #--------------------------------------------------------------------------------#
    def compute_oriented_box(self):

        if(len(self.contour) > 0):
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

        if(len(self.contour) > stp.CNTRS_LEN_MIN and 
           self.length >= stp.FIBER_MIN_LEN and 
           self.width <= stp.FIBER_MAX_WIDTH):
            return True
        
        return False
    
    #--------------------------------------------------------------------------------#
    def draw_fiber(self, img : np.ndarray, reg_angle : float = None) :

        if(tools.img_empty(img)):
            raise ValueError(f"draw_fibers() Fiber.py line 63 : img is empty")
        
        if(reg_angle):
            color = tools.angle_color(reg_angle)
        else:
            color = tools.angle_color(self.angle, stp.COLOR_PATH)

        cv.drawContours(image=img, contours=[self.contour], 
                        contourIdx=-1, color=color, 
                        thickness=1, lineType=cv.LINE_AA)

    #--------------------------------------------------------------------------------#
    def print(self):

        print(f"angle           = {self.angle}")
        print(f"position        = {self.position}")
        print(f"length          = {self.length}")
        print(f"oriented_box    = {self.oriented_box}")
        print(f"cnt.len         = {len(self.contour)}")
        print("\n")

#------------------------------------------------------------------------------------#
#-------------------------------------- STATIC --------------------------------------#
#------------------------------------------------------------------------------------#
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
def sort_fibers(fibers : list[Fiber]) -> list:

    if not fibers:
        return

    sorted_fibers = []
    fibers_cpy = fibers.copy()

    while(len(fibers_cpy) > 0) :

        ref_fib = fibers_cpy.pop(0)
        ref_ang = ref_fib.angle

        next_regions = []
        current_regions = [ref_fib] 
        for i in range(len(fibers_cpy)):
            
            current_fib = fibers_cpy[i]
            current_angle = current_fib.angle

            diff = abs(ref_ang - current_angle)

            if(diff > 90):
                diff = 180 - diff

            if(diff < stp.DELTA_ANGLE):
                current_regions.append(current_fib)
            else:
                next_regions.append(current_fib)

        if(len(current_regions) > stp.MIN_REGION_SIZE):
            sorted_fibers.append(current_regions) 
            
        fibers_cpy = next_regions    
            
    return sorted_fibers
    
#----------------------------------------------------------------------------------------------#
#------------------------------------------- MAIN ---------------------------------------------#
#----------------------------------------------------------------------------------------------#
if __name__ == "__main__":

    cnt = [0, 1.3, 15.3]
    fib = Fiber(cnt)
    fib.print()

