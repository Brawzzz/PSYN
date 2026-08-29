import os
import numpy as np
import pandas as pd
import cv2 as cv

from tqdm import tqdm

import Sample

import setup as stp


#================================================================================#
def compare_data(file1_path, file2_path, output_path):

    #------------------------------
    df1 = pd.read_csv(file1_path, skipinitialspace=True)
    df2 = pd.read_csv(file2_path, skipinitialspace=True)
    
    print(df1)
    print()
    print(df2)

    #------------------------------
    rows_report = []
    for idx1, row1 in df1.iterrows():   

        print(idx1)

        # distances = np.sqrt((df2["centroid_x"] - row1["centroid_x"])**2 + 
        #                     (df2["centroid_y"] - row1["centroid_y"])**2)
        
        # idx2_proche  = distances.idxmin()
        # distance_min = distances.min()
        
        # if distance_min > 100: 
        #     continue
            
        # row2 = df2.loc[idx2_proche]
        
        # diff_data = {
        #     "region_id": idx1,
        #     "mean_angle_ref": row1["mean_angle"],
        #     "diff_nb_fibers": row2["nb_fibers"] - row1["nb_fibers"],
        #     "diff_area_percent": ((row2["area"] - row1["area"]) / row1["area"]) * 100,
        #     "centroid_shift_px": distance_min
        # }
        # rows_report.append(diff_data)
    
    #------------------------------
    report_df = pd.DataFrame(rows_report)
    report_df.to_csv(output_path, index=False)

#================================================================================#
def build_dataset(n_sample : Sample.Sample, 
                  patch_size : int = 512, 
                  export_dir : str = "./data/dataset/UNet"):

    """
    build a dataset by spliting the sample image and the global mask of teh regions

    :params global_img:     Sample img to be split   
    :params global_mask:    Global mask of the Sample's regions  
    :params patch_size:     size of a split 
    :params export_dir:     export direcrtory to save the images
    """

    #------------------------------
    img_dir     = os.path.join(export_dir, "images")
    mask_dir    = os.path.join(export_dir, "masks")

    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    #------------------------------
    (h, w)      = n_sample.img.shape[:2]
    global_mask = np.full((h, w), -100.0, dtype=np.float32)

    for reg in n_sample.regions:

        angle_value = float(reg.mean_angle) 

        #---------------
        for shape in reg.shapes:
            for fib in shape.fibers:

                cv.drawContours(global_mask, [fib.contour], contourIdx=-1, color=angle_value, thickness=cv.FILLED)

    #------------------------------
    (h, w)      = n_sample.img.shape[:2]
    patch_id    = 0

    y_steps         = len(range(0, h, patch_size))
    x_steps         = len(range(0, w, patch_size))
    total_patches   = y_steps * x_steps

    with tqdm(total=total_patches, desc="Building dataset ", unit="reg") as pbar:
    
        for y in range(0, h, patch_size):
            for x in range(0, w, patch_size):
                
                img_patch   = n_sample.img[y:y+patch_size, x:x+patch_size]
                mask_patch  = global_mask[y:y+patch_size, x:x+patch_size]

                #---------------
                if img_patch.shape[:2] != (patch_size, patch_size):
                    pbar.update(1)
                    continue
                    
                #---------------
                cv.imwrite(os.path.join(img_dir, f"patch_{patch_id}.png"), img_patch)
                np.save(os.path.join(mask_dir, f"patch_{patch_id}.npy"), mask_patch)
                
                patch_id += 1

                pbar.update(1)

#================================================================================#
if __name__ == "__main__":

    # config_file_0_1 = f"./output/hxtl_p25_pre/data/regions_data_config_0.1.csv"
    # config_file_0_2 = f"./output/hxtl_p25_pre/data/regions_data_config_0.2.csv"
    # output_path = f"./output/hxtl_p25_pre/data/regions_data_diff_config_0.1_0.2.csv"

    # compare_data(config_file_0_1, config_file_0_2, output_path)

    sample = Sample.Sample(stp.SAMPLE_INDEX, n_split=stp.NB_SPLIT)
    sample.set_path()
    sample.regions_path = "./data/sample_25/before_fretting/RoiSet_p25_pre/"
    sample.img_path = "./data/sample_25/before_fretting/hxtl_p25_pre.bmp"
    sample.load_img()

    regions = sample.get_roi()

    img_col = cv.cvtColor(sample.img, cv.COLOR_GRAY2RGB)
    cv.drawContours(image=img_col, contours=regions, contourIdx=-1, color=(0, 255, 0), thickness=25)
    cv.imwrite("./output/result_2022_regions.png", img_col)
