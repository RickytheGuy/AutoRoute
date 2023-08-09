import os
import sys
import numpy as np

try:
    import gdal
    import ogr
    import osr
    from gdalconst import GA_ReadOnly
except:
	from osgeo import gdal, ogr, osr
	from osgeo.gdalconst import GA_ReadOnly
        
# Add the project_root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.AutoRoute import *

# some variables to use
ds = gdal.Open(r"tests\DEMs\DEM_merged.tif", GA_ReadOnly)
merged_dem_arr = ds.GetRasterBand(1).ReadAsArray(0)
ds = gdal.Open(r"tests\DEMs\DEM_extent.tif", GA_ReadOnly)
extent_dem_arr = ds.GetRasterBand(1).ReadAsArray(0)

lr_dems = [r"tests\DEMs\DEM_left.tif", r"tests\DEMs\DEM_right.tif"]
out_dem = r"tests\test.tif"

try:
    # Test 1: Merge multiple DEMs into a single DEM
    PrepareDEM(lr_dems,out_dem)
    ds = gdal.Open(out_dem, GA_ReadOnly)
    out_arr = ds.GetRasterBand(1).ReadAsArray(0)
    ds = None

    assert merged_dem_arr.shape == out_arr.shape, "Test 1: Merge multiple DEMs into a single DEM failed: Mismatched shapes"
    assert (merged_dem_arr == out_arr).all(), "Test 1.1: Merge multiple DEMs into a single DEM failed: Values do not match"

    # Test 2: Dem clip to extent
    PrepareDEM(lr_dems,out_dem,extent=(-70.76,18.89,-70.7,18.96))
    ds = gdal.Open(out_dem, GA_ReadOnly)
    out_arr = ds.GetRasterBand(1).ReadAsArray(0)
    ds = None

    assert extent_dem_arr.shape == out_arr.shape, "Test 2: Dem clip to extent failed: Mismatched shapes"
    assert (extent_dem_arr == out_arr).all(), "Test 2.1: Dem clip to extent failed: Values do not match"

    print("Finished tests succesfully")
except Exception as e:
     print(e)
