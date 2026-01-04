#----------------------------------------------------------------------------------------------#
#------------------------------------------- IMPORT -------------------------------------------#
#----------------------------------------------------------------------------------------------#
import os
import cv2 as cv
import numpy as np
from sklearn.cluster import DBSCAN
import alphashape as a_shape

import Fiber
import setup as stp
import tools


#----------------------------------------------------------------------------------------------#
#------------------------------------------- CLASS --------------------------------------------#
#----------------------------------------------------------------------------------------------#
class Region :

    def __init__(self, fibers : list[Fiber.Fiber], n_split_index : int, sample_regions_path : str):
        
        self.split_index    = n_split_index

        self.fibers         = fibers.copy()
        self.angle          = self.compute_angle()
        self.shape          = self.compute_shape()

        self.name           = stp.REGION_ + str(int(self.angle))
        self.region_path    = sample_regions_path + stp.SPLIT_ + str(self.split_index) + "/" + self.name + "/"

    #--------------------------------------------------------------------------------#
    def compute_angle(self):

        if not self.fibers :
            raise ValueError("compute_angle() in Region.py line 23 : self.fibers is empty")
        
        angles = [fib.angle for fib in self.fibers]
        return np.mean(angles)

    #--------------------------------------------------------------------------------#
    def compute_shape(self):

        if(len(self.fibers) == 0):
            raise ValueError(f"compute_shape() Region.py line 41 : fibers is empty")

        #---------------
        all_contour_points = []

        for fib in self.fibers:
            pts = (fib.contour).reshape(-1, 2)
            all_contour_points.append(pts)

        if not all_contour_points:
            return None

        points = np.vstack(all_contour_points)

        #---------------
        try:
            shape_polygon = a_shape.alphashape(points, stp.ALPHA)
            return shape_polygon
        
        except Exception as e:
            print(f"Error while computing alphashape: {e}")
            return None

    #--------------------------------------------------------------------------------#
    def process_region(self):

        return
    
    #--------------------------------------------------------------------------------#
    def draw_regions(self, img : np.ndarray, drawing_method : int = stp.DRAW_FIBER) -> None:

        #---------------
        if(drawing_method == stp.DRAW_FIBER):

            for fib in self.fibers:
                fib.draw_fiber(img, reg_angle=self.angle)

            return

        #---------------
        elif(drawing_method == stp.DRAW_SHAPE):
            
            if self.shape.geom_type == 'Polygon':
                shape_list = [self.shape]
            elif self.shape.geom_type == 'MultiPolygon':
                shape_list = list(self.shape.geoms)
            else:
                return
            
            color = tools.angle_color(self.angle, stp.CONFIG_COLOR_PATH)

            shape_cnt = []
            
            for poly in shape_list:
                cnt = tools.shapely_to_opencv(poly)
                if cnt is not None:
                    shape_cnt.append(cnt)

            cv.drawContours(img, shape_cnt, -1, color, thickness=stp.THICKNESS)

        #---------------
        elif(drawing_method == stp.DRAW_FIBER + stp.DRAW_SHAPE):
            # fiber and shape
            return
        
        #---------------
        else:
            raise ValueError(f"draw_region() Region.py line : 43 : uknown method value | method = {drawing_method}")
    
    #--------------------------------------------------------------------------------#
    def print(self):

        print(f"#========== {self.name} ==========#")
        print(f"angle       = {self.angle}")
        print(f"shape       = {self.shape.geom_type}")
        print(f"fibers.len  = {len(self.fibers)}")
        print(f"region path : {self.region_path}")
        print(f"#================================#")
        print("\n")

    #--------------------------------------------------------------------------------#
    def save(self, img_path : str, drawing_method : int = stp.DRAW_FIBER):

        os.makedirs(self.region_path, exist_ok=True)

        if(os.path.exists(img_path)) :

            img = cv.imread(img_path, cv.IMREAD_COLOR_RGB)
            self.draw_regions(img, method=drawing_method)

            if(drawing_method == stp.DRAW_FIBER):
                file_name = self.region_path + self.name + stp.FIBER_ + stp.OUTPUT_EXTENSION
            elif(drawing_method == stp.DRAW_SHAPE):
                file_name = self.region_path + self.name + stp.SHAPE_ + stp.OUTPUT_EXTENSION
            elif(drawing_method == stp.DRAW_FIBER + stp.DRAW_SHAPE):
                file_name = self.region_path + self.name + stp.FIBER_SHAPE_ + stp.OUTPUT_EXTENSION

            cv.imwrite(file_name, img)

        return

    #--------------------------------------------------------------------------------#
    def clean_regions(self) -> list[Fiber.Fiber]:
        
        coords = np.array([fib.position for fib in self.fibers])
        dbscan = DBSCAN(eps=stp.DBSCAN_EPS, min_samples=stp.DBSCAN_MIN_SAMPLES, metric='euclidean').fit(coords)

        labels = dbscan.labels_
        valid_fibers = []
        invalid_fibers = []

        for fib, label in zip(self.fibers, labels):
            if label == -1: 
                invalid_fibers.append(fib)
            else:
                valid_fibers.append(fib)

        self.fibers = valid_fibers

        return (invalid_fibers)

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

