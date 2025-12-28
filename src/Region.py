#----------------------------------------------------------------------------------------------#
#------------------------------------------- IMPORT -------------------------------------------#
#----------------------------------------------------------------------------------------------#
import os
import numpy as np
import cv2 as cv
import alphashape as a_shape

import setup as stp
import Fiber


#----------------------------------------------------------------------------------------------#
#------------------------------------------- CLASS --------------------------------------------#
#----------------------------------------------------------------------------------------------#
class Region :

    def __init__(self, fibers : list[Fiber.Fiber], n_split_index : int, sample_regions_path : str):
        
        self.split_index    = n_split_index

        self.fibers         = fibers.copy()
        self.angle          = self.compute_angle()
        self.shape          = self.compute_shape()

        self.name           = stp.REGION + str(int(self.angle))
        self.region_path    = sample_regions_path + stp.SPLIT + str(self.split_index) + "/" + self.name + "/"

    #--------------------------------------------------------------------------------#
    def compute_angle(self):

        if not self.fibers :
            raise ValueError("compute_angle() in Region.py line 23 : self.fibers is empty")
        
        angles = [fib.angle for fib in self.fibers]
        return np.mean(angles)
    
    #--------------------------------------------------------------------------------#
    def compute_shape(self):

        a_shape = []
        return a_shape

    #--------------------------------------------------------------------------------#
    def draw_regions(self, img : np.ndarray, method : int = 0) :

        if(method == 0):
            for fib in self.fibers:
                fib.draw_fiber(img, reg_angle=self.angle)

        elif(method == 1):
            # shape
            return
        
        elif(method == 2):
            # fiber and shape
            return
        
        else:
            raise ValueError(f"draw_region() Region.py line : 43 : uknown method value | method = {method}")
    
    #--------------------------------------------------------------------------------#
    def process_region(self):

        return
    
    #--------------------------------------------------------------------------------#
    def print(self):

        print(f"angle       = {self.angle}")
        print(f"shape       = {self.shape}")
        print(f"fibers.len  = {len(self.fibers)}")
        print(f"region path : {self.region_path}")
        print("\n")

    #--------------------------------------------------------------------------------#
    def save(self, img_path : str):

        os.makedirs(self.region_path, exist_ok=True)

        if(os.path.exists(img_path)) :

            img = cv.imread(img_path, cv.IMREAD_COLOR_RGB)

            self.draw_regions(img, method = 0)
            file_name = self.region_path + self.name + stp.DRAW_FIBER + stp.OUTPUT_EXTENSION

            cv.imwrite(file_name, img)

        return

#------------------------------------------------------------------------------------#
#--------------------------------------- MAIN ---------------------------------------#
#------------------------------------------------------------------------------------#
if __name__ == "__main__":

    cnt = [0, 1.3, 15.3]

    fib_0 = Fiber.Fiber(cnt)
    fib_1 = Fiber.Fiber(cnt)
    fibers = [fib_0, fib_1]

    region = Region(fibers, stp.SAMPLE_INDEX, "./output/hxtl_p25_pre/regions_images/")

    region.print()

