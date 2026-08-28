#============================================================================================================================#
#---------------------------------------------------------- IMPORT ----------------------------------------------------------#
#============================================================================================================================#
import os
import roifile

import cv2 as cv
import numpy as np
import alphashape as a_shape

from shapely.geometry import Polygon
from shapely.geometry import MultiPolygon
from shapely.ops import unary_union

from sklearn.cluster import DBSCAN

import Fiber
import tools
import setup as stp


#============================================================================================================================#
#---------------------------------------------------------- CLASS -----------------------------------------------------------#
#============================================================================================================================#
class Region :

    def __init__(self, n_fibers : list[Fiber.Fiber], n_split_index : int):
        
        self.split_index        = n_split_index

        self.fibers             = self.set_fibers(n_fibers)
        self.nb_fibers          = len(self.fibers)

        self.mean_fibers_len    = self.compute_mean_fibers_len()
        self.mean_fibers_width  = self.compute_mean_fibers_width()
            
        self.mean_angle         = self.compute_angle()
        self.median_angle       = self.compute_angle(compute_mean=False)

        self.shapes             = []
        self.area               = 0.0
        self.shape_file         = ""

        self.name               = stp.REGION_ + str(int(self.mean_angle))

    #================================================================================#
    def set_fibers(self, n_fibers : list[Fiber.Fiber]) -> None:
        
        """
        fill self.fibers with n_fibers params
        """
        
        #------------------------------
        coords = np.array([fib.position for fib in n_fibers])
        dbscan = DBSCAN(eps=stp.DBSCAN_EPS, min_samples=stp.DBSCAN_MIN_SAMPLES, metric='euclidean').fit(coords)

        valid_fibers = []
        labels = dbscan.labels_

        for fib, label in zip(n_fibers, labels):
            if label != -1: 
                valid_fibers.append(fib)

        return valid_fibers
    
    #================================================================================#
    def compute_angle(self, compute_mean=True) -> float:

        """
        compute the global angle of a Region.

        :params compute_mean: if true return the mean angle else return the median angle
        """
        
        #------------------------------
        if not self.fibers :
            Warning("compute_angle() in Region.py line 52 : self.fibers is empty")
            return -1
        
        angles = np.array([fib.angle for fib in self.fibers])

        #------------------------------
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
            
        #------------------------------
        else:
            return np.mean(angles) if compute_mean else np.median(angles)  

    #================================================================================#
    def run_moph(self):

        """
        function to run the morphological algorithm on region's fiber list
        
        :return boudaries: MultiPolygon object defining the region's boundaries
        """

        #------------------------------
        polygons = []
        for fib in self.fibers:

            pts = fib.contour.reshape(-1, 2) 

            if len(pts) >= 3:

                poly =  Polygon(pts)

                if not poly.is_valid:
                    poly = poly.buffer(0) 

                if not poly.is_empty:
                    polygons.append(poly)

        #---------------
        if not polygons:
            return MultiPolygon([])

        dilated_polys   = [p.buffer(stp.R_DILATE).buffer(0) for p in polygons]
        merged_shape    = unary_union(dilated_polys)
        final_shape     = merged_shape.buffer(stp.R_ERODE).buffer(0)
        
        #---------------
        if(final_shape.geom_type == "MultiPolygon"):
            
            valid_geoms = []
            geoms = list(final_shape.geoms)

            #---------------
            for poly in geoms:

                if poly.area >= stp.SHAPE_MIN_AREA:
                    valid_geoms.append(poly)     

            #---------------
            if not valid_geoms:
                return MultiPolygon([]) 
            
            elif len(valid_geoms) == 1:
                return valid_geoms[0] 
            
            else:
                return MultiPolygon(valid_geoms)

        #--------------- 
        elif(final_shape.geom_type == "Polygon"):
            
            if final_shape.area >= stp.SHAPE_MIN_AREA:  
                return final_shape

            return MultiPolygon([])

    #================================================================================#
    def run_alphashape(self):

        """
        function to run the alphashape algorithm on region's fiber list

        :return boudaries: MultiPolygon object defining the region's boundaries
        """

        #------------------------------
        all_contour_points = []
        
        for fib in self.fibers:
            pts = (fib.contour).reshape(-1, 2)
            all_contour_points.append(pts[::stp.POINT_STEP])

        if not all_contour_points:
            return MultiPolygon([])

        points = np.vstack(all_contour_points)

        #---------------
        try:

            shape_polygon = a_shape.alphashape(points, stp.ALPHA)

            #---------------
            if shape_polygon.geom_type == 'MultiPolygon':
                
                valid_polys = [poly for poly in shape_polygon.geoms if poly.area > stp.SHAPE_MIN_AREA]
                
                if not valid_polys:
                    return  MultiPolygon([])
                elif len(valid_polys) == 1:
                    return valid_polys[0]  
                else:
                    return MultiPolygon(valid_polys)

            #---------------
            elif shape_polygon.geom_type == 'Polygon':
                
                if shape_polygon.area > stp.SHAPE_MIN_AREA:
                    return shape_polygon
                else:
                    return  MultiPolygon([])

            #---------------
            else:
                return  MultiPolygon([])

        except Exception as e:
            raise ValueError(f"Error while running alphashape algorithm : {e}")

    #================================================================================#
    def compute_boundaries(self, method : int = 0) -> MultiPolygon :

        """
        compute the general shape of the Region

        :params method: defines the wanted method to compute the shape (defaults : morph)

        :return boundaries: MultiPolygon object
        """
        
        #------------------------------
        if(len(self.fibers) == 0):
            raise ValueError("compute_boundaries() in Region.py line 52 : self.fibers is empty")

        #------------------------------
        if(method == stp.MORPH):
            return(self.run_moph())
            
        elif(method == stp.A_SHAPE) :
            return(self.run_alphashape())

        
    #================================================================================#
    def compute_mean_fibers_len(self):

        """
        compute the mean length of the fibers in the Region
        """

        #------------------------------
        mean = 0
        for fib in self.fibers:
            mean += fib.length 

        if(self.nb_fibers != 0):
            self.mean_fibers_len = mean / self.nb_fibers
        else:
            self.mean_fibers_len = -1.0

        return self.mean_fibers_len
    
    #================================================================================#
    def compute_mean_fibers_width(self):

        """
        compute the mean width of the fibers in the Region
        """

        #------------------------------
        mean = 0
        for fib in self.fibers:
            mean += fib.width

        if(self.nb_fibers != 0):
            self.mean_fibers_width = mean / self.nb_fibers
        else:
            self.mean_fibers_width = -1.0

        return self.mean_fibers_width
    
    #================================================================================#
    def add_fiber(self, n_fibers : list[Fiber.Fiber] | Fiber.Fiber = None):

        """
        allows to manualy add Fibers in a regions

        :params n_fibers: list of Fiber or a simple Fiber
        """
        
        #------------------------------
        if n_fibers is None:
            return
        
        if not isinstance(n_fibers, list):
            fibers = [n_fibers]

        (self.fibers).extend(fibers)

        self.mean_angle   = self.compute_angle(compute_mean=True)
        self.median_angle = self.compute_angle()
        
    #================================================================================#
    def render(self, 
               img : np.ndarray, 
               n_config_path : str,
               render_type : int = stp.DRAW_FIBER) -> np.ndarray:
        
        """
        global rendering of a region

        :params img:             image on wich we draw the shape
        :params n_config_path:   path to the configuration file
        :params render_type:     wanted type of render 

        :return img: image with the bondaries of the region
        """

        #------------------------------
        if not self.shapes:
            return img
        
        for shape in self.shapes:
            shape.render(img=img, render_type=render_type, n_config_path=n_config_path)

        return img
    
    #================================================================================#
    def print(self):

        """
        print all the data of the region
        """

        #------------------------------
        print(f"#========== {self.name} ==========#")
        print(f"split index      = {self.split_index}")

        print(f"mean angle       = {self.mean_angle}")
        print(f"median angle     = {self.median_angle}")

        if(self.shapes != [] or self.shapes != None):
            print(f"shape            = {self.shapes.geom_type}")

        print(f"fibers.len       = {len(self.fibers)}")
        print(f"#================================#")
        print("\n")

    #================================================================================#
    def save(self, n_regions_path : str):

        """
        allows to save the Region in .roi format compatible with ImageJ

        :params n_regions_path:  path to the directory in wich the Region wille be saved 
        """
        
        #------------------------------
        if not self.shapes:
            raise ValueError(f"self.dhapes is empty or None : {self.shapes}")
        
        #------------------------------
        ext_count = 0
        for shape in self.shapes:
            
            poly            = shape.polygon
            points_ext      = list(poly.exterior.coords)
            
            roi_ext         = roifile.ImagejRoi.frompoints(points_ext)
            roi_ext.name    = f"{self.name}_{ext_count}_ext"
            roi_ext.roitype = roifile.ROI_TYPE.POLYGON
            
            filename_ext    = f"{self.name}_cnt_{ext_count}_ext.roi"
            roi_ext.tofile(os.path.join(n_regions_path, filename_ext))

            #---------------
            int_count = 0
            for hole in poly.interiors:

                hole_area = Polygon(hole).area                

                if hole_area > stp.HOLE_MIN_AREA:

                    points_hole = list(hole.coords)
                    roi_hole    = roifile.ImagejRoi.frompoints(points_hole)
                    
                    roi_hole.name       = f"{self.name}_{ext_count}_hole_{int_count}"
                    roi_hole.roitype    = roifile.ROI_TYPE.POLYGON
                    
                    filename_hole = f"{self.name}_cnt_{ext_count}_hole_{int_count}.roi"
                    roi_hole.tofile(os.path.join(n_regions_path, filename_hole))

                    int_count += 1
                
            ext_count += 1

#============================================================================================================================#
#----------------------------------------------------- STATICS METHODS ------------------------------------------------------#
#============================================================================================================================#
@staticmethod
def merge_regions(regions : list[Region]):

    """
    allows to merge regions

    :params regions:    list of regions to merge
    """

    #------------------------------
    all_fib = []
    for reg in regions:

        all_fib += reg.fibers

    region_merged = Region(all_fib, n_split_index=-1)
    
    return region_merged
