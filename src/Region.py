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

    def __init__(self, fibers : list[Fiber.Fiber], n_split_index : int, sample_regions_path : str):
        
        self.split_index    = n_split_index

        self.fibers         = self.set_fibers(fibers)

        self.mean_angle     = self.compute_angle()
        self.median_angle   = self.compute_angle(compute_mean=False)

        self.shape          = self.compute_shape()

        self.name           = stp.REGION_ + str(int(self.mean_angle))
        self.region_path    = sample_regions_path + stp.SPLIT_ + str(self.split_index) + "/" + self.name + "/"  

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
    def compute_angle(self, compute_mean=True):

        if not self.fibers :
            raise ValueError("compute_angle() in Region.py line 23 : self.fibers is empty")
        
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
            raise ValueError(f"Error while computing alphashape: {e}")
    
    #----------------------------------------------------------------------------------------------------------------------------#
    #---------------------------------------------------------- REDER -----------------------------------------------------------#
    #----------------------------------------------------------------------------------------------------------------------------#
    def render_shape(self, img : np.ndarray) -> None:
                
        if self.shape.geom_type == 'Polygon':
            shape_list = [self.shape]
        elif self.shape.geom_type == 'MultiPolygon':
            shape_list = list(self.shape.geoms)
        else:
            return
        
        #---------------
        color = tools.angle_color(self.mean_angle, config_path=stp.CONFIG_COLOR_PATH)

        shape_cnt = []
        
        for poly in shape_list:
            cnt = tools.shapely_to_opencv(poly)
            if cnt is not None:
                shape_cnt.append(cnt)

        cv.drawContours(img, shape_cnt, -1, color, thickness=stp.SHAPE_THICKNESS)

    #--------------------------------------------------------------------------------#
    def render_fibers(self, img : np.ndarray) -> None:

        for fib in self.fibers:
            fib.render(img, reg_angle=self.mean_angle)

    #--------------------------------------------------------------------------------#
    def draw(self, img : np.ndarray, render_type : int = stp.DRAW_FIBER) -> None:

        #---------------
        if(render_type == stp.DRAW_FIBER):
            self.render_fibers(img)
            return

        #---------------
        elif(render_type == stp.DRAW_SHAPE):
            self.render_shape(img)
            return
        
        #---------------
        elif(render_type == stp.DRAW_FIBER + stp.DRAW_SHAPE):
            self.render_fibers(img)
            self.render_shape(img)
            return
        
        #---------------
        else:
            raise ValueError(f"draw() Region.py line : 136 : uknown method value | method = {render_type}")
    
    #--------------------------------------------------------------------------------#
    def render(self, 
               all_img : np.ndarray, 
               region_img : np.ndarray,
               render_type : int = stp.DRAW_FIBER):

        if(tools.img_empty(all_img) or tools.img_empty(region_img)):
            raise ValueError(f"render() Region.py line 157 : img is empty img = {all_img.shape} or {region_img.shape}")
        
        os.makedirs(self.region_path, exist_ok=True)

        self.draw(region_img, render_type=render_type)
        self.draw(all_img, render_type=render_type)

        if(render_type == stp.DRAW_FIBER):
            region_img_path = self.region_path + self.name + stp.FIBER_ + stp.OUTPUT_EXTENSION

        elif(render_type == stp.DRAW_SHAPE):
            region_img_path = self.region_path + self.name + stp.SHAPE_ + stp.OUTPUT_EXTENSION

        elif(render_type == stp.DRAW_FIBER + stp.DRAW_SHAPE):
            region_img_path = self.region_path + self.name + stp.FIBER_SHAPE_ + stp.OUTPUT_EXTENSION

        cv.imwrite(region_img_path, region_img)
    
    #--------------------------------------------------------------------------------#
    def print(self):

        print(f"#========== {self.name} ==========#")
        print(f"split index      = {self.split_index}")

        print(f"mean angle       = {self.mean_angle}")
        print(f"median angle     = {self.median_angle}")

        print(f"shape            = {self.shape.geom_type}")
        print(f"fibers.len       = {len(self.fibers)}")

        print(f"region path      : {self.region_path}")
        print(f"#================================#")
        print("\n")
