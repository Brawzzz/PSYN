#----------------------------------------------------------------------------------------------#
#------------------------------------------- IMPORT -------------------------------------------#
#----------------------------------------------------------------------------------------------#
import os
import shutil
import json
import colorsys
import numpy as np
import cv2 as cv

import setup as stp


#----------------------------------------------------------------------------------------------#
#------------------------------------------ FUNCTION ------------------------------------------#
#----------------------------------------------------------------------------------------------#
def f_pass(x):
    pass

#--------------------------------------------------------------------------------#
def is_pow_2(n):
    return((n and (n-1)) == 0)

#--------------------------------------------------------------------------------#
def generate_colors(n):

    colors = []

    for i in range(n):

        hue = i / n
        saturation = 1.0
        lightness = 1.0
        
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, lightness)

        colors.append((int(r*255), int(g*255), int(b*255)))
        
    return colors

#--------------------------------------------------------------------------------#
def angle_color(angle : float, config_path : str = None):

    if(config_path):

        if(os.path.exists(config_path)):
            
            with open(config_path, "r") as f:
                config = json.load(f)
            
            angle = angle % 180
            ranges = config.get("ranges", [])

            for item in ranges:
                if item["min"] <= angle <= item["max"]:
                    return tuple(item["color"])

            return tuple(config.get("default_color", [255, 0, 0]))
        
        else:
            print(f"{config_path} : No such file or directory")
            return
        
    else:

        normalize_angle = (angle % 360)
        hue = int(normalize_angle / 2)
        hsv = np.uint8([[[hue, 255, 255]]])

        bgr = cv.cvtColor(hsv, cv.COLOR_HSV2RGB)[0][0]

        return (int(bgr[0]), int(bgr[1]), int(bgr[1]))

#--------------------------------------------------------------------------------#
def img_empty(img):

    if img is None:
        return True
    
    if img.size == 0:
        return True
        
    return False

#--------------------------------------------------------------------------------#
def interactive_th(img_blur, f_pass=f_pass):
    
    cv.namedWindow('Thresh settings')
    cv.createTrackbar('Thresh', 'Thresh settings', 100, 255, f_pass)

    while True:

        thresh_val = cv.getTrackbarPos('Thresh', 'Thresh settings')
        (ret, img_bw )= cv.threshold(img_blur, thresh_val, stp.TH_MAX, cv.THRESH_TOZERO)
        cv.imshow('Thresh settings', img_bw)
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cv.destroyAllWindows()
    print(f"Selecterd value for thresh : {thresh_val}")

    return(thresh_val, img_bw)

#--------------------------------------------------------------------------------#
def make_color_config():

    if(180 % int(stp.DELTA_ANGLE) != 0):
        return

    step = int(stp.DELTA_ANGLE)
    nb_color = int(180 / step)
    
    config_list = []
    colors = generate_colors(nb_color)

    index = 0
    for i in range(0, 180, step):

        config = {
            "min": i,
            "max": i + step,
            "color": colors[index],
        }

        config_list.append(config)
        index += 1

    with open("./config/color_config.json", "w", encoding="utf-8") as f:
        
        final_json = {
            "description": "Color Config",
            "step": step,
            "ranges": config_list
        }
        
        json.dump(final_json, f, indent=4)

#--------------------------------------------------------------------------------#
def clear_folder(folder_path, extension=stp.OUTPUT_EXTENSION):

    for item in os.listdir(folder_path):
        
        item_path = os.path.join(folder_path, item)

        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.remove(item_path)
            
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
                
        except Exception as e:
            print(f"Impossible dto remove {item_path} : {e}")

#--------------------------------------------------------------------------------#
def verifiy_folder(folder_path : str, folder_len : int) -> bool:

    if(os.path.exists(folder_path)):
            
        if(len(os.listdir(folder_path)) == folder_len):
            return False
        else:
            clear_folder(folder_path)
            return True

    else:
        os.makedirs(folder_path, exist_ok=True)
        return True


#--------------------------------------------------------------------------------#
def shapely_to_opencv(polygon):

    if polygon.is_empty:
        return None

    (x, y) = polygon.exterior.coords.xy

    points = np.array([ [int(xi), int(yi)] for xi, yi in zip(x, y) ], dtype=np.int32)
    points = points.reshape((-1, 1, 2))

    return points
    
#----------------------------------------------------------------------------------------------#
#------------------------------------------- MAIN ---------------------------------------------#
#----------------------------------------------------------------------------------------------#
if __name__ == "__main__":
    make_color_config()

