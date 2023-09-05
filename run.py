from ARFS_inputs import MainPreprocess

import glob
import os
import multiprocessing

# Inputs below
DEM_folder = r"C:\Users\water\Desktop\WORK\AutoRoute\CODE_AutoRoute\Permanent_Variables\DEMs_for_Entire_World"
buffered_dems = r"D:\DEMs_buff"

LU_folder = r"C:\Users\water\Desktop\WORK\AutoRoute\CODE_AutoRoute\Permanent_Variables\LandCover"
buffered_lus = r"D:\LUs_buff"

parquets = r"C:\Users\water\Desktop\WORK\AutoRoute\CODE_AutoRoute\Permanent_Variables\Parquets"
extent_table = r"C:\Users\water\AutoRoute\extents.parquet"
buffered_strms = r"D:\Strms_buf"

num_processes = 8

mannings_dict = {0:0,
                 1:1,
                 20:2,
                 30:3,
                 40:4,
                 50:5,
                 60:6,
                 70:7,
                 80:8,
                 90:9,
                 100:10,
                 111:11,
                 112:12,
                 113:13,
                 114:14,
                 115:15,
                 116:16,
                 200:20,
                 121:21,
                 122:22,
                 123:23,
                 124:24,
                 125:25,
                 126:26}


def split_list_into_sublists(input_list, chunk_size):
    for i in range(0, len(input_list), chunk_size):
        yield input_list[i:i + chunk_size]

if __name__ =='__main__':
    DEMs = glob.glob(os.path.join(DEM_folder, '*','*.tif'))
    sublists = list(split_list_into_sublists(DEMs, 8))
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.starmap(MainPreprocess, [(file, DEM_folder, buffered_dems, LU_folder, buffered_lus, buffered_strms, parquets, extent_table, mannings_dict) for file in DEMs])

