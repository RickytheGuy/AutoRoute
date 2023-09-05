from src.floodspreader_lrr43.main import *

import geopandas as gpd

import os
import glob
import sys
import logging
import re

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout
)

stream_parquets = r'C:\Users\water\Desktop\WORK\AutoRoute\CODE_AutoRoute\Permanent_Variables\Parquets\*.parquet'
stream_gpks = r'C:\Users\water\Desktop\WORK\AutoRoute\CODE_AutoRoute\Permanent_Variables\StreamLines_World\*.gpkg'
out_dir = r"D:\ARFS"
DEMs_World = r"C:\Users\water\Desktop\WORK\AutoRoute\CODE_AutoRoute\Permanent_Variables\DEMs_for_Entire_World"
LandCover_World = r"C:\Users\water\Desktop\WORK\AutoRoute\CODE_AutoRoute\Permanent_Variables\LandCover"
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


vpu_folder_list = sorted(glob.glob(stream_parquets))
gpkg_vpu_folder_list = sorted(glob.glob(stream_gpks))

for vpu_parquet, gpkg in zip(vpu_folder_list, gpkg_vpu_folder_list):
    try:
        vpu_number = re.findall(r'\d+', os.path.basename(vpu_parquet))[0]
        vpu_folder = os.path.join(out_dir, vpu_number)
        os.makedirs(vpu_folder, exist_ok=True)
        logging.info(f'Working on {vpu_number}')

        # dem = os.path.join(vpu_folder, f'dem_{vpu_number}.tif')
        # if not os.path.exists(dem):
        #     extent = gpd.read_parquet(vpu_parquet).total_bounds
        #     DEM_Parser(DEMs_World, dem, extent=extent)

        # lu = os.path.join(vpu_folder, f'lu_{vpu_number}.tif')
        # if not os.path.exists(lu) and os.path.exists(dem):
        #     LU_Parser(LandCover_World, dem, lu, mannings_dict)

        strm = os.path.join(vpu_folder, f'strm_{vpu_number}.tif')
        # if not os.path.exists(strm) and os.path.exists(dem):
        #     PrepareStream(dem, gpkg, strm, field='TDXHydroLinkNo')

        rapid = os.path.join(vpu_folder, f'rapid_{vpu_number}.txt')
        if not os.path.exists(rapid) and os.path.exists(strm):
            MakeRAPIDFile("the csv", strm, rapid, field='TDHydroLinkNo')

        logging.info(f"Finished {vpu_number}")
    except Exception as e:
        logging.error(f"{vpu_number} failed!!!!!")
        logging.error(e)