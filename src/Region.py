#----------------------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------- IMPORT ----------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------------#
import os
import cv2 as cv
import numpy as np
from sklearn.cluster import DBSCAN
import alphashape as a_shape

import Fiber
import setup as stp
import tools


#----------------------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------- CLASS -----------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------------#
class Region :

    def __init__(self, fibers : list[Fiber.Fiber], n_split_index : int):
        
        self.split_index    = n_split_index

        self.fibers         = self.set_fibers(fibers)

        self.mean_angle     = self.compute_angle()
        self.median_angle   = self.compute_angle(compute_mean=False)

        self.shape          = []

        self.name           = stp.REGION_ + str(int(self.mean_angle))
        
    #--------------------------------------------------------------------------------#
    def set_fibers(self, fibers : list[Fiber.Fiber]) -> None:
        
        coords = np.array([fib.position for fib in fibers])
        dbscan = DBSCAN(eps=stp.DBSCAN_EPS, min_samples=stp.DBSCAN_MIN_SAMPLES, metric='euclidean').fit(coords)

        valid_fibers = []
        labels = dbscan.labels_

        for fib, label in zip(fibers, labels):
            if label != -1: 
                valid_fibers.append(fib)

        return valid_fibers
    
    #--------------------------------------------------------------------------------#
    def compute_angle(self, compute_mean=True) -> float:

        if not self.fibers :
            Warning("compute_angle() in Region.py line 52 : self.fibers is empty")
            return -1
        
        angles = np.array([fib.angle for fib in self.fibers])

        #--------------- 
        if np.std(angles) > stp.MAX_ANGLE_DEV: 

            angles_shifted = np.where(angles > 90, angles - stp.MAX_ANGLE, angles)

            #--------------- 
            mean_val = np.mean(angles_shifted)
            if (mean_val < stp.MIN_ANGLE):
                mean_val += stp.MAX_ANGLE

            median_val = np.median(angles_shifted)
            if (median_val < stp.MIN_ANGLE):
                median_val += stp.MAX_ANGLE

            return mean_val if compute_mean else median_val
            
        #---------------
        else:
            return np.mean(angles) if compute_mean else np.median(angles)  
        
    #--------------------------------------------------------------------------------#
    def compute_shape(self):

        if(len(self.fibers) == 0):
            Warning("compute_angle() in Region.py line 52 : self.fibers is empty")
            return []

        #---------------
        STEP = 200
        all_contour_points = []

        for fib in self.fibers:
            pts = (fib.contour).reshape(-1, 2)
            all_contour_points.append(pts[::STEP])

        if not all_contour_points:
            return None

        points = np.vstack(all_contour_points)

        #---------------
        try:
            shape_polygon = a_shape.alphashape(points, stp.ALPHA)
            return shape_polygon

        except Exception as e:
            raise ValueError(f"Error while computing alphashape: {e}")
    
    #--------------------------------------------------------------------------------#
    def add_fiber(self, n_fib : Fiber.Fiber = None, n_fibers : list[Fiber.Fiber] = None):

        if n_fibers :
            for fib in n_fibers:
                (self.fibers).append(fib)

        elif n_fib:
            (self.fibers).append(n_fib)

        else :
            raise ValueError(f"n_fib and n_fibers are both None : n_fib = {n_fib}, n_fibers = {n_fibers} impossible to add fib")
        
        self.mean_angle   = self.compute_angle(compute_mean=True)
        self.median_angle = self.compute_angle()

    #----------------------------------------------------------------------------------------------------------------------------#
    #---------------------------------------------------------- REDER -----------------------------------------------------------#
    #----------------------------------------------------------------------------------------------------------------------------#
    def render_shape(self, img : np.ndarray, color_config_path : str = None) -> None:
                
        if self.shape.geom_type == 'Polygon':
            shape_list = [self.shape]
        elif self.shape.geom_type == 'MultiPolygon':
            shape_list = list(self.shape.geoms)
        else:
            return
        
        #---------------
        if(color_config_path != None):
            if(os.path.exists(color_config_path)):
                color = tools.get_color(self.mean_angle, config_path=color_config_path)
            else:
                raise ValueError(f"{color_config_path} do not exist")
        else:
            color = tools.angle_to_color(self.mean_angle)

        shape_cnt = []
        
        for poly in shape_list:
            cnt = tools.shapely_to_opencv(poly)
            if cnt is not None:
                shape_cnt.append(cnt)

        cv.drawContours(img, shape_cnt, -1, color, thickness=stp.SHAPE_THICKNESS)

    #--------------------------------------------------------------------------------#
    def render_fibers(self, img : np.ndarray, color_config_path : str) -> None:

        if(len(self.fibers) == 0):
            return
        
        for fib in self.fibers:
            fib.render(img, 
                       reg_angle=self.mean_angle, 
                       color_config_path=color_config_path)

    #--------------------------------------------------------------------------------#
    def render(self, img : np.ndarray, 
               color_config_path : str,
               render_type : int = stp.DRAW_FIBER) -> np.ndarray:

        #---------------
        if(render_type == stp.DRAW_FIBER):
            self.render_fibers(img, color_config_path)
            return img

        #---------------
        elif(render_type == stp.DRAW_SHAPE):
            self.render_shape(img, color_config_path)
            return img
        
        #---------------
        elif(render_type == stp.DRAW_FIBER + stp.DRAW_SHAPE):
            self.render_fibers(img, color_config_path)
            self.render_shape(img, color_config_path)
            return img
        
        #---------------
        else:
            raise ValueError(f"render() Region.py line : 136 : uknown method value | method = {render_type}")
    
    #--------------------------------------------------------------------------------#
    def print(self):

        print(f"#========== {self.name} ==========#")
        print(f"split index      = {self.split_index}")

        print(f"mean angle       = {self.mean_angle}")
        print(f"median angle     = {self.median_angle}")

        if(self.shape != []):
            print(f"shape            = {self.shape.geom_type}")

        print(f"fibers.len       = {len(self.fibers)}")
        print(f"#================================#")
        print("\n")

#----------------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------- STATICS METHODS ------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------------#
@staticmethod
def merge_regions(regions : list[Region]):

    all_fib = []
    for reg in regions:

        all_fib += reg.fibers

    region_merged = Region(all_fib, n_split_index=-1)
    
    return region_merged
