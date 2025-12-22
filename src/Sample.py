import os
import cv2 as cv
import numpy as np
import setup as stp
import utilis


class Sample :

    def __init__(self, n_id):

        self.id             = n_id
        self.img_path       = None 
        self.img            = None 
        self.name           = None 
        self.before_fret    = None 

        self.regions        = []
        self.regions_plot   = None
        self.result_path    = ""

        self.split_list     = []
        self.row            = 0
        self.col            = 0
        self.split_path     = ""
    
    #----------------
    def set_img_path(self, n_bf=True):

        self.before_fret = n_bf

        if(self.before_fret):
            self.name = "hxtl_p" + self.id + "_pre" + stp.IMG_EXTENSION
            self.img_path = stp.DATA_PATH + "sample_" + str(self.id) + "/before_fretting/" + self.name
        else:
           self.name = "hxtl_p" + self.id + "_post" + stp.IMG_EXTENSION
           self.img_path = stp.DATA_PATH + "sample_" + str(self.id) + "/after_fretting/" + self.name 
        
        self.split_path = stp.SPLIT_PATH + os.path.splitext(self.name)[0] + "/"
    
    #----------------
    def load_img(self):

        if(self.img_path == None):
            print('\n********** EROR **********')
            print(f"Impossible to load img : img_path = {self.img_path}")
            print('\n**************************')

        elif(not os.path.exists(self.img_path)):
            print('\n********** EROR **********')
            print(f"Impossible to load img : img_path do not exist : img_path = {self.img_path}")
            print('\n**************************')

        else :
            self.img = cv.imread(self.img_path, cv.IMREAD_GRAYSCALE)

    #----------------
    def compute_row_col(self, n):

        if(n == 2):
            self.row = 2
            self.col = 1
        
        self.row = 2
        self.col = int(n / self.row)

        while(self.col % 2 == 0 and self.col > 4):

            self.row *= 2
            self.col = int(self.col / 2)        

    #----------------
    def split(self, nb_split=2):
        
        if(nb_split % 2 != 0):

            print("***** ERROR *****")
            print(f"nb_split must be even : nb_split = {nb_split}")
            print("***************************")

            return
        
        if(utilis.img_empty(self.img)):

            print("***** ERROR *****")
            print(f"impossible to split : self.img is empty ")
            print("***************************")
        
            return
        
        (img_h, img_w) = self.img.shape[:2]
        self.compute_row_col(nb_split) 
        
        h_step = img_h // self.row
        w_step = img_w // self.col

        for y in range(self.row):
            for x in range(self.col):
                
                y_start = y * h_step
                y_end = (y + 1) * h_step

                x_start = x * w_step
                x_end = (x + 1) * w_step

                if (y == self.row - 1):
                    y_end = img_h

                if (x == self.col - 1):
                    x_end = img_w

                img_xy = self.img[y_start:y_end, x_start:x_end]
                self.split_list.append(img_xy)

    #----------------
    def join(self):

        sub_img = []
        for y in range(self.row):
                
            start_index = y * self.col
            end_index = start_index + self.col

            row_img = self.split_list[start_index:end_index]
            strip_i = np.hstack(row_img)
            sub_img.append(strip_i) 

        img = np.vstack(sub_img)

        cv.imwrite(stp.RECON_PATH + os.path.splitext(self.name)[0] + "_recon.jpg", img)

        return img
    
    #----------------
    def print(self):

        print(f"#================ SAMPLE {self.id} ================#")
        print(f"name : {self.name}")
        print(f"path : {self.img_path}")
        print(f"before fret : {self.before_fret}")

        print(f"\n(row, col) : ({self.row}, {self.col})")

        if(self.split_list == None):
            print(f"split list : {self.split_list}")
        else:
            print(f"split list len : {len(self.split_list)}")
            print(f"split list : \n")
            for im in self.split_list:
                print(im.size)

        print("\n\n")

    #----------------
    def save(self):

        if(not utilis.img_empty(self.regions_plot)):

            os.makedirs(self.result_path, exist_ok=True)
            cv.imwrite(self.result_path, self.regions_plot)

        if(len(self.split_list) > 0):

            for i in range(len(self.split_list)):

                os.makedirs(self.split_path, exist_ok=True)
                file_name = self.split_path + "sub_" + str(i) + ".jpg"

                cv.imwrite(file_name, self.split_list[i])