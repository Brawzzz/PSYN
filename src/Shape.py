#============================================================================================================================#
#---------------------------------------------------------- IMPORT ----------------------------------------------------------#
#============================================================================================================================#
import os
import roifile

import cv2 as cv
import numpy as np

from tqdm import tqdm
from shapely.geometry import Point, Polygon, MultiPolygon

import Fiber

import tools
import setup as stp 


#============================================================================================================================#
#---------------------------------------------------------- CLASS -----------------------------------------------------------#
#============================================================================================================================#
class Shape:

    #================================================================================#
    def __init__(self, n_poly : Polygon, n_fibers : list[Fiber.Fiber], n_angle : float):

        self.fibers         = n_fibers
        self.nb_fiber       = len(self.fibers)
        self.angle          = n_angle
        
        self.polygon        = n_poly

        self.area           = self.polygon.area
        self.perimeter      = self.polygon.length
        self.centroid       = self.polygon.centroid
        self.density        = self.nb_fiber / self.area if self.area > 0 else 0

        self.render_file    = ""
        self.saving_file    = ""

    #================================================================================#
    def print(self):

        """
        print all the information of a shape
        """

        #------------------------------
        print(f"centroid    : {self.polygon.centroid}")
        print(f"area        : {self.polygon.area}")

    #================================================================================#
    def update(self, n_poly : Polygon, n_fibers : list[Fiber.Fiber]):

        """
        update a shape's informations

        :params n_poly:     new polygon of the updated shape
        :params n_fibers:   new list of fiber of the updated shape 
        """

        #------------------------------
        self.polygon    = n_poly
        self.fibers     = n_fibers
        self.nb_fiber   = len(self.fibers)
        
        self.area       = self.polygon.area
        self.perimeter  = self.polygon.length
        self.centroid   = self.polygon.centroid
        
        self.density = self.nb_fiber / self.area if self.area > 0 else 0.0

    #================================================================================#
    def render_polygon(self, img : np.ndarray, n_config_path : str = None) -> None:
            
        """
        rendering polygon's boundaries on an img

        :params img:            image on wich we draw the the polygon's boundaries
        :params n_config_path:  path to the configuration file
        """

        #------------------------------
        if(self.polygon == None):
            raise ValueError(f"render_polygon() Shape.py line 62 : self.polygon is None")
        
        #------------------------------
        if(n_config_path != None):

            if(os.path.exists(n_config_path)):
                color = tools.get_color(self.angle, n_config_path=n_config_path)
            else:
                raise ValueError(f"{n_config_path} do not exist")
            
        else:
            color = tools.angle_to_color(self.angle)

        #------------------------------
        poly_cnt = tools.shapely_to_opencv(self.polygon)
        cv.drawContours(img, poly_cnt, -1, color, thickness=stp.SHAPE_THICKNESS, lineType=cv.LINE_AA)
        
        #------------------------------
        # if(not os.path.exists(self.shape_file)):

        #     shape_cnt = []
        #     for poly in shape_list:
        #         shape_cnt.extend(tools.shapely_to_opencv(poly))

        #     cv.drawContours(img, shape_cnt, -1, color, thickness=stp.SHAPE_THICKNESS, lineType=cv.LINE_AA)

        #------------------------------
        # else:

        #     roi     = roifile.ImagejRoi.fromfile(self.shape_file)
        #     points  = roi.coordinates()
        #     points  = np.array(points, dtype=np.int32)

        #     cnt = points.reshape((-1, 1, 2))

        #     cv.drawContours(img, [cnt], -1, color, thickness=stp.SHAPE_THICKNESS, lineType=cv.LINE_AA)

    #================================================================================#
    def render_fibers(self, img : np.ndarray, n_config_path : str) -> None:

        """
        render all the fiber of the Region contained in self.fibers

        :params img:            image on wich we draw the shape
        :params n_config_path:  path to the configuration file
        """
        
        #------------------------------
        if(self.nb_fiber == 0):
            return
        
        for fib in self.fibers:
            fib.render(img, 
                        reg_angle=self.angle, 
                        n_config_path=n_config_path)
                
    #================================================================================#
    def render(self, img : np.ndarray, render_type : int, n_config_path : str = None) -> None:
        
        """
        rendering of the shapes forming the Region

        :params img:            image on wich we draw the shape
        :params n_config_path:  path to the configuration file
        """

        #------------------------------
        if(self.polygon == None):
            raise ValueError(f"render() Shape.py line 67 : self.polygon is empty")

        #------------------------------
        if(render_type == stp.DRAW_FIBER):
            self.render_fibers(img, n_config_path)
            return img

        #------------------------------
        elif(render_type == stp.DRAW_SHAPE):
            self.render_polygon(img, n_config_path)
            return img
        
        #------------------------------
        elif(render_type == stp.DRAW_FIBER + stp.DRAW_SHAPE):
            self.render_fibers(img, n_config_path)
            self.render_polygon(img, n_config_path)
            return img
        
        #------------------------------
        else:
            raise ValueError(f"render() Region.py line : 136 : uknown method value | method = {render_type}")

        
#================================================================================#
@staticmethod
def set_shapes(borders : MultiPolygon, n_fibers : Fiber.Fiber, n_angle : float) -> list[Shape]:

    """
    split the borders compute by Regions method into a list of Shapes

    :params borders:    shapely Polygon or multipolygon from Regions computation
    :params n_fibers:   group of Fiber contained in the Regions 

    :return shapes: list of shape
    """

    #------------------------------
    if borders.geom_type == "Polygon":
        poly_list = [borders]
    elif borders.geom_type == "MultiPolygon":
        poly_list = list(borders.geoms)
    else:
        return []

    #------------------------------
    shapes = []

    with tqdm(total=len(poly_list), desc=f"Split region {int(n_angle)} in shapes  ", unit="polygons") as pbar:

        for poly in poly_list:

            shape_fibers    = [fib for fib in n_fibers if poly.contains(Point(fib.position))]
            shape_i         = Shape(poly, shape_fibers, n_angle)

            shapes.append(shape_i)

            pbar.update(1)

    return shapes


