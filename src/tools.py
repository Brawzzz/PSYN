#============================================================================================================================#
#---------------------------------------------------------- IMPORT ----------------------------------------------------------#
#============================================================================================================================#
import os
import shutil
import json
import re
import numpy as np
import cv2 as cv

import setup as stp


#============================================================================================================================#
#-------------------------------------------------------- FUNCTION ----------------------------------------------------------#
#============================================================================================================================#
def angle_to_color(angle : float):

    """
    compute a unique color associated to angle
    """

    #------------------------------
    hue = int(angle % 180)
    hsv = np.uint8([[[hue, 255, 255]]])
    bgr_px = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)[0][0]

    return (int(bgr_px[0]), int(bgr_px[1]), int(bgr_px[2]))

#================================================================================#
def get_color(angle : float, n_config_path : str):
    
    """
    retun a color associated to angle from a colors configuration file

    n_config_path : path to the colors configuration file
    """
    #------------------------------
    if(os.path.exists(n_config_path)):
        
        with open(n_config_path, "r") as f:
            config = json.load(f)
        
        colors = config.get("colors", [])

        min_dist = float('inf')
        best_color = [-1, -1, -1]

        for item in colors:

            d1 = abs(item["angle"] - angle)
            d2 = stp.MAX_ANGLE - d1
            d  = min(d1, d2)

            if d < min_dist:
                min_dist = d
                best_color = item["color"]

        if(best_color == [-1, -1, -1]):
            return [0, 0, 0]
        else:
            return best_color
    
    else:
        print(f"{n_config_path} : No such file or directory")
        return
    
#================================================================================#
def clear_folder(folder_path):

    """
    clear completely a directory 

    folder_path :  path of the folder to clear
    """

    #------------------------------
    for item in os.listdir(folder_path):
        
        item_path = os.path.join(folder_path, item)

        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.remove(item_path)
            
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
                
        except Exception as e:
            print(f"Impossible dto remove {item_path} : {e}")

#================================================================================#
def shapely_to_opencv(polygon):

    """
    convert a Shapely object to an OpenCV object 
    """

    #------------------------------
    if polygon.is_empty:
        return None

    (x, y) = polygon.exterior.coords.xy

    points = np.array([ [int(xi), int(yi)] for xi, yi in zip(x, y) ], dtype=np.int32)
    points = points.reshape((-1, 1, 2))

    return points

#================================================================================#
def get_peaks(angles : list[float], 
              min_peak_height = 5, 
              sigma_smoth = 2.0) -> list[float]:
    
    """
    return a list of the peak angles in a list of angles

    angles          : list[float] 
    min_peak_height : minimun occurces to be considerate as a peak
    sigma_smoth     : smooth factor to compute the histogramme
    """

    #------------------------------
    if len(angles) == 0:
        return []
    
    if(max(angles) > stp.MAX_ANGLE or min(angles) < stp.MIN_ANGLE):
        Warning(f"max(angles) > {stp.MAX_ANGLE} or min(angles) < {stp.MIN_ANGLE}")
        return angles
    
    #------------------------------
    BIN_SIZE        = 1       
    MARGIN_CIRCULAR = stp.DELTA_ANGLE 

    #------------------------------
    angles_extended = list(angles)

    for ang in angles:
        if ang < MARGIN_CIRCULAR : 
            angles_extended.append(ang + stp.MAX_ANGLE)
        elif ang > stp.MAX_ANGLE - MARGIN_CIRCULAR : 
            angles_extended.append(ang - stp.MAX_ANGLE)
    
    #------------------------------
    bins        = np.arange(-MARGIN_CIRCULAR, stp.MAX_ANGLE + MARGIN_CIRCULAR + BIN_SIZE, BIN_SIZE)
    (hist, _)   = np.histogram(angles_extended, bins=bins)
    hist        = hist.astype(np.float32).reshape(1, -1)

    k_size      = int(2 * np.ceil(3 * sigma_smoth) + 1)
    hist_smooth = cv.GaussianBlur(hist, (k_size, 1), sigma_smoth)[0]
    
    #------------------------------
    start_idx = int(MARGIN_CIRCULAR / BIN_SIZE)
    end_idx   = start_idx + int(stp.MAX_ANGLE / BIN_SIZE)

    hist_real   = hist_smooth[start_idx : end_idx]    
    peaks_index = []

    for i in range(1, len(hist_real) - 1):

        if((hist_real[i-1] < hist_real[i]) and (hist_real[i] > hist_real[i+1])):
            if(hist_real[i] > min_peak_height):
                peaks_index.append(i)
    
    if len(hist_real) > 0:
        
        if((hist_real[0] > hist_real[1] and hist_real[0] > min_peak_height) or 
           (hist_real[-1] > hist_real[-2] and hist_real[-1] > min_peak_height)) :
            
            if(0 not in peaks_index and 179 not in peaks_index):
                peaks_index.append(0)

    if not peaks_index:
        return []

    #------------------------------
    peaks_index.sort()
    
    merged_peaks = []
    if peaks_index:

        current_peak = peaks_index[0]

        for i in range(1, len(peaks_index)):

            next_peak = peaks_index[i]
            
            d1 = abs(current_peak - next_peak)
            d2 = stp.MAX_ANGLE - d1
            d  = min(d1, d2)
            
            if d < stp.DELTA_ANGLE:
                if hist_real[next_peak] > hist_real[current_peak]:
                    current_peak = next_peak
            else:
                merged_peaks.append(current_peak)
                current_peak = next_peak

        #---------------
        if len(merged_peaks) > 0:
            first = merged_peaks[0]

            d1 = abs(first - current_peak)
            d2 = stp.MAX_ANGLE - d1
            d  = min(d1, d2)

            if d <= stp.DELTA_ANGLE:
                if hist_real[current_peak] > hist_real[first]:
                    merged_peaks[0] = current_peak
            else:
                merged_peaks.append(current_peak)
                
        else:
            merged_peaks.append(current_peak)

        return  merged_peaks

#================================================================================#
def img_empty(img):

    """
    return True if img is empty or do not exists
    """
    
    #------------------------------
    if img is None:
        return True
    
    if img.size == 0:
        return True
        
    return False

#================================================================================#
def extract_number(path):

    match = re.search(r'_(\d+)', path)
    if match:
        return int(match.group(1)) 
    return 0

#================================================================================#
def is_pow_2(n):
    return((n and (n-1)) == 0)

#================================================================================#
def f_pass(x):
    pass
