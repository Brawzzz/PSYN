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
        
        angles = np.array([fib.angle for fib in self.fibers])

        # 1. Gestion de la circularité (Si on a un mix proche de 0 et 180)
        # On décale temporairement les angles > 90 vers les négatifs pour le calcul
        # Ex: 179 devient -1. Ainsi la médiane de [1, -1] est 0.

        # On détecte si la région traverse la frontière 0/180
        # (Astuce simple: si l'écart type est énorme, c'est qu'on est sur la frontière)
        if np.std(angles) > 45: 
            angles_shifted = np.where(angles > 90, angles - 180, angles)
            median_val = np.median(angles_shifted)
            
            # On remet le résultat entre [0, 180]
            if median_val < 0:
                median_val += 180
            return median_val
            
        # 2. Cas standard (pas de frontière 0/180)
        else:
            return np.median(angles)
        # return np.mean(angles)

    #--------------------------------------------------------------------------------#
    def clean_fibers(self) -> None:
        
        coords = np.array([fib.position for fib in self.fibers])
        dbscan = DBSCAN(eps=stp.DBSCAN_EPS, min_samples=stp.DBSCAN_MIN_SAMPLES, metric='euclidean').fit(coords)

        valid_fibers = []
        labels = dbscan.labels_

        for fib, label in zip(self.fibers, labels):
            if label != -1: 
                valid_fibers.append(fib)

        self.fibers = valid_fibers

        return
    
    #--------------------------------------------------------------------------------#
    def compute_shape(self):

        if(len(self.fibers) == 0):
            raise ValueError(f"compute_shape() Region.py line 41 : fibers is empty")

        self.clean_fibers()

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

    #--------------------------------------------------------------------------------#
    def process_region(self):

        return
    
    #--------------------------------------------------------------------------------#
    def draw_shape(self, img : np.ndarray) -> None:
                
        if self.shape.geom_type == 'Polygon':
            shape_list = [self.shape]
        elif self.shape.geom_type == 'MultiPolygon':
            shape_list = list(self.shape.geoms)
        else:
            return
        
        #---------------
        color = tools.angle_color(self.angle, config_path=stp.CONFIG_COLOR_PATH)

        shape_cnt = []
        
        for poly in shape_list:
            cnt = tools.shapely_to_opencv(poly)
            if cnt is not None:
                shape_cnt.append(cnt)

        cv.drawContours(img, shape_cnt, -1, color, thickness=stp.SHAPE_THICKNESS)

    #--------------------------------------------------------------------------------#
    def draw_regions(self, img : np.ndarray, drawing_method : int = stp.DRAW_FIBER) -> None:

        #---------------
        if(drawing_method == stp.DRAW_FIBER):

            for fib in self.fibers:
                fib.draw_fiber(img, reg_angle=self.angle)

            return

        #---------------
        elif(drawing_method == stp.DRAW_SHAPE):

            self.draw_shape(img)
            return
        
        #---------------
        elif(drawing_method == stp.DRAW_FIBER + stp.DRAW_SHAPE):

            for fib in self.fibers:
                fib.draw_fiber(img, reg_angle=self.angle)

            self.draw_shape(img)

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
    def save(self, 
             split_img : np.ndarray, 
             region_img : np.ndarray,
             drawing_method : int = stp.DRAW_FIBER):

        if(tools.img_empty(split_img)):
            raise ValueError(f"save() Region.py line 157 : imgis empty img = {split_img}")
        
        os.makedirs(self.region_path, exist_ok=True)

        self.draw_regions(region_img, drawing_method=drawing_method)
        self.draw_regions(split_img, drawing_method=drawing_method)

        if(drawing_method == stp.DRAW_FIBER):
            img_path = self.region_path + self.name + stp.FIBER_ + stp.OUTPUT_EXTENSION
        elif(drawing_method == stp.DRAW_SHAPE):
            img_path = self.region_path + self.name + stp.SHAPE_ + stp.OUTPUT_EXTENSION
        elif(drawing_method == stp.DRAW_FIBER + stp.DRAW_SHAPE):
            img_path = self.region_path + self.name + stp.FIBER_SHAPE_ + stp.OUTPUT_EXTENSION

        cv.imwrite(img_path, region_img)

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

