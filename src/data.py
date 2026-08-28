import numpy as np
import pandas as pd
import cv2 as cv

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
