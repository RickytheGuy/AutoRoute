import os
import re
from distutils.version import LooseVersion

import numpy as np
import geopandas as gpd
import pandas as pd
from osgeo import gdal, ogr
from osgeo.gdalconst import GA_ReadOnly

def MainPreprocess(file: str, 
                   DEM_folder,
                   buffered_dems: str, 
                   LU_folder: str, 
                   buffered_lus: str, 
                   buffered_strms: str,
                   parquets: str,
                   extent_table: str, 
                   mannings_dict: str):
    """
    Create the DEM, rasterized streamfile, and reclassified landcover for use in AutoRoute.

    Parameters:
    -----------
    file: str,
        Path to a DEM (the orgiginal from FABDEM)
    DEM_folder: str,
        Path to the folder containing subfolders for the DEMs of the entire world
    buffered_dems: str
        Path to the output folder of the buffered DEMs
    LU_folder: str
        Path to the folder containing the LU rasters
    buffered_lus: str
        Path to the output folder of the buffered and reclassified LUs
    buffered_strms: str
        Path to the output folder of the buffered and rasterized streams
    parquets: str
        Path to the folder that contains all the vpus as parquets
    extent_table: str
        Path to the parquet file that contains the extent of every stream in the world
    manning_dict: str
        A dictionary to use to reclassify the LU

    """
    pattern = r"(\w\d{2})(\w\d{3}).+"
    are_there_streams_here = True

    file_extent = [int(x[1:]) if x[0] in 'NE' else -int(x[1:]) for x in re.findall(pattern, os.path.basename(file))[0]]
    file_extent.reverse()
    file_extent += [file_extent[0] + 1, file_extent[1] + 1]
    minx = max(file_extent[0] - 0.1, -180)
    miny = max(file_extent[1] - 0.1, -90)
    maxx = min(file_extent[2] + 0.1, 180)
    maxy = min(file_extent[3] + 0.1, 90)

    out_dem = os.path.join(buffered_dems, os.path.basename(file).split('.')[0] + '_buffered.tif')
    out_strm = os.path.join(buffered_strms, os.path.basename(file).split('.')[0]  + '_strm.tif')
    out_parquet = os.path.join(buffered_strms, os.path.basename(file).split('.')[0]  + '.parquet')
    if not os.path.exists(out_strm) or not os.path.exists(out_parquet):
        are_there_streams_here = StreamLine_Parser(out_dem, DEM_folder, parquets, extent_table, (minx,miny,maxx,maxy), out_strm, out_parquet,'TDXHydroLinkNo')

    if are_there_streams_here == False:
        return 

    out_lu = os.path.join(buffered_lus, os.path.basename(file).split('.')[0]  + '_LU.tif')
    if not os.path.exists(out_lu):
        LU_Parser(LU_folder,out_dem,out_lu,mannings_dict)


def PrepareLU(dem_file_path: str, land_file_path: str or list, out_path: str, normalize: dict = None):
    """
    Make a Land Cover raster that is the same size and resolution as the DEM file from one or more landcover files.

    Parameters
    ----------
    dem_file_path : string
        Path to DEM file
    land_file_path : string
        Path to land file or path to a folder containg land cover files
    out_path : string, list
        Path, including name and file extension, of the output, or a list of such
    normalize : dict, optional
       If a dictionary is passed in the raster is reclassified according to the input dictionary

        .. versionadded:: 0.1.0

    Returns
    -------
    PrepareLU : None, dict
        Either returns None or a dictionary of normalized LU values

    Notes
    -----

    Examples
    --------

    """

    (minx, miny, maxx, maxy, _, _, ncols, nrows, _, projection) = _Get_Raster_Details(dem_file_path)

    if isinstance(land_file_path, (list, tuple)):
        tiff_files = land_file_path
    elif isinstance(land_file_path, str):
        if land_file_path.endswith(".tif"):
            tiff_files = [land_file_path]
        else:
            tiff_files = [os.path.join(land_file_path, file) for file in os.listdir(land_file_path) if file.endswith('.tif')]
    else:
        raise ValueError(f"I dont't know what to do with land file path of type {type(land_file_path)}")

    # Check if the projection is EPSG:4326 (WGS84)
    # if '["EPSG","4326"]' not in projection:
    #     # Reproject all GeoTIFFs to EPSG:4326
    #     projection = osr.SpatialReference()
    #     projection.ImportFromEPSG(4326)

    # Create an in-memory VRT (Virtual Dataset) for merging the GeoTIFFs
    vrt_options = gdal.BuildVRTOptions(resampleAlg='nearest')
    vrt_dataset = gdal.BuildVRT('', [tiff_file for tiff_file in tiff_files], options=vrt_options)

    # options = gdal.TranslateOptions(
    #     outputType=gdal.GDT_Byte,
    #     width=ncols,
    #     height=nrows,
    #     outputBounds=(minx, miny, maxx, maxy),
    #     outputSRS=projection,
    #     resampleAlg=gdal.GRA_NearestNeighbour,
    #     format='VRT'
    # )

    # final_dataset = gdal.Translate('', vrt_dataset, options=options)
    options = gdal.WarpOptions(outputType=gdal.GDT_Byte, width=ncols, height=nrows, outputBounds=(minx, miny, maxx, maxy), 
                                dstSRS=projection, resampleAlg=gdal.GRA_NearestNeighbour, format='VRT', multithread=True,
                                creationOptions = ["COMPRESS=DEFLATE", "PREDICTOR=2","BIGTIFF=YES"])
    final_dataset = gdal.Warp('', vrt_dataset, options=options) 

    hDriver = gdal.GetDriverByName("GTiff")
    out_file = hDriver.Create(out_path, xsize=ncols, ysize=nrows, bands=1, eType=gdal.GDT_Byte)
    out_file.SetGeoTransform(final_dataset.GetGeoTransform())
    out_file.SetProjection(projection)


    block_size = 30000
    # Process the array in blocks
    for y in range(0, nrows, block_size):
        for x in range(0, ncols, block_size):
            actual_block_size_x = min(block_size, ncols - x)
            actual_block_size_y = min(block_size, nrows - y)

            # Read the block of data
            block = final_dataset.ReadAsArray(x, y, actual_block_size_x, actual_block_size_y)

            # Perform reclassification operation on the block
            if normalize is not None:
                out_file.GetRasterBand(1).WriteArray(np.vectorize(normalize.get)(block), x, y)
            else:
                out_file.GetRasterBand(1).WriteArray(block, x, y)

    out_file = None
    final_dataset = None

def LU_Parser(folder: str, dem: str, output: str, reclassify: dict = None,  
              pattern: str = r"(\w\d{3})(\w\d{2}).+\.tif"):
    minx1, miny1, maxx1, maxy1, _, _, ncols, nrows, _, projection = _Get_Raster_Details(dem)
    if miny1 > maxy1:
        maxy1, miny1 = miny1, maxy1
    if minx1 > maxx1:
        maxx1, minx1 = minx1, maxx1

    files = []

    for file in os.listdir(folder):
        if 'E020' in file:
            x=1
        file_extent = [int(x[1:]) if x[0] in 'NE' else -int(x[1:]) for x in re.findall(pattern, file)[0]]
        file_extent[1] -= 20
        file_extent += [file_extent[0]+20, file_extent[1]+20]
        minx2, miny2, maxx2, maxy2 = file_extent
        if minx1 <= maxx2 and maxx1 >= minx2 and miny1 <= maxy2 and maxy1 >= miny2:
            files.append(os.path.join(folder,file))

    if len(files) == 0:
        raise ValueError("No files found that are in the extent. Check the extent, filenames, and the patterns!")
    PrepareLU(dem,files,output,reclassify)

def StreamLine_Parser(dem: str, 
                      DEM_folder: str, 
                      stream_parquet_folder: str, 
                      extent_table: str, 
                      extent: tuple or list,
                      output_strm: str, 
                      out_parquet: str, 
                      field: str = 'TDXHydroLinkNo'):
    # Load the DEM extent
    minx, miny, maxx, maxy = extent

    extents_df = pd.read_parquet(extent_table)
    filtered_gdf = extents_df[
        (minx <= extents_df.maxx) &
        (maxx >= extents_df.minx) &
        (miny <= extents_df.maxy) &
        (maxy >= extents_df.miny)
    ]

    vpus = filtered_gdf.VPUCode.unique()
    resulting_dfs = []

    for file in os.listdir(stream_parquet_folder):
        if file.endswith('.parquet') and int(file[4:7]) in vpus:
            resulting_dfs.append(
                filtered_gdf.merge(gpd.read_parquet(os.path.join(stream_parquet_folder, file)), 
                                    on=field, 
                                    how='inner'
                                    )
                )

    # If empty, there are no streams here
    if resulting_dfs == []:
        return False
    
    if not os.path.exists(dem):
        DEM_Parser(DEM_folder, dem, extent=(minx, miny, maxx, maxy))

    minx, miny, maxx, maxy, _, _, ncols, nrows, geoTransform, projection = _Get_Raster_Details(dem)

    gdf = gpd.GeoDataFrame(pd.concat(resulting_dfs), crs = 'EPSG:4326')
    gdf.to_parquet(out_parquet)

    # Check GDAL version
    gdal_version = LooseVersion(gdal.__version__)

    # Define the required version
    parque_version = LooseVersion("3.5")

    # In case gdal won't open a parquet or older version is installed, we have two options (Riley may not like but it definetly helps me out)
    if gdal_version >= parque_version:
        gdf = None
        LayerName = os.path.basename(output_strm).split('.')[-2]
        options = gdal.RasterizeOptions(noData=0, outputType=gdal.GDT_UInt32, attribute=field, width=ncols, height=nrows, 
                                outputBounds=(minx, miny, maxx, maxy), layers=LayerName,
                                outputSRS=projection,
                                creationOptions = ["COMPRESS=DEFLATE", "PREDICTOR=2","BIGTIFF=YES"])
    
        gdal.Rasterize(output_strm, out_parquet, options=options)
    
        return True

    driver = ogr.GetDriverByName("Memory")
    ds = driver.CreateDataSource('')
    field_defn = ogr.FieldDefn(field, ogr.OFTInteger)
    dest_srs = ogr.osr.SpatialReference()
    dest_srs.ImportFromEPSG(4326)
    source_layer = ds.CreateLayer(field, dest_srs, ogr.wkbMultiLineString)
    source_layer.CreateField(field_defn)

    for _, row in gdf.iterrows():
        feature = ogr.Feature(source_layer.GetLayerDefn())
        feature.SetGeometry(ogr.CreateGeometryFromWkb(row.geometry.wkb))
        feature.SetField(field, int(row[field]))
        source_layer.CreateFeature(feature)
        feature = None


    # Create the destination data source
    target_ds = gdal.GetDriverByName('GTiff').Create(output_strm, ncols, nrows, 1, gdal.GDT_UInt32)
    target_ds.SetGeoTransform(geoTransform)
    target_ds.SetProjection(projection)

    # Rasterize
    options = ["COMPRESS=DEFLATE", "PREDICTOR=2","ATTRIBUTE=" + field]
    gdal.RasterizeLayer(target_ds, [1], source_layer, options=options)
    return True

def PrepareDEM(dem, output: str, extent = None, clip = None):
    """
    Prepare a DEM for use in AutoRoute / FloodSpreader programs. This function will:
        1) merge all DEMs into one file (if applicable)
        2) Clip DEM to an extent using the nearest cells
        3) Clip DEM to a region using the nearest cells

    Parameters
    ----------
    dem : any
        dem is either a path to a dem, a list of paths, or a directory containg .tif files. 
    extent: any, optional
        If specified, a list or tuple in the format of (minx, miny, maxx, maxy) is expected. 
        The DEM is clipped as close as possible to the extent
    clip : any, optional
        If specified, a file (.shp, .gpkg, .parquet) is expected. 
        The DEM is clipped as close as possible to the extent of the feature
    """

    # check the inputs
    if isinstance(dem, str):
        assert os.path.exists(dem), "Dem file does not exist"
    if extent is not None and clip is not None:
        raise ValueError("Must pass either extent or clip, not both")
    if isinstance(extent, (tuple, list)):
        if len(extent) != 4 or extent[0] >= extent[2] or extent[1] >= extent[3]:
            raise ValueError(f"Invlaid extent {extent}")

    # if dem is one file, then load it into memory
    if isinstance(dem, (list, tuple)):
        tiff_files = dem
    elif dem.endswith(".tif"):
        tiff_files = [dem]
    else:
        tiff_files = [os.path.join(dem, file) for file in os.listdir(dem) if file.endswith('.tif')]
    minx, miny, maxx, maxy, dx, dy, ncols, nrows, geoTransform, projection = _Get_Raster_Details(tiff_files[0])

    # Check if the projection is EPSG:4326 (WGS84)
    # if 'EPSG:4326' not in projection:
    #     # Reproject all GeoTIFFs to EPSG:4326
    #     projection = osr.SpatialReference()
    #     projection.ImportFromEPSG(4326)

    # Create an in-memory VRT (Virtual Dataset) for merging the GeoTIFFs
    vrt_options = gdal.BuildVRTOptions(resampleAlg='bilinear')
    vrt_dataset = gdal.BuildVRT('', [tiff_file for tiff_file in tiff_files], options=vrt_options)
    vrt_geo = vrt_dataset.GetGeoTransform()
    dataset_extent = [vrt_geo[0],  # xmin
                      vrt_geo[3] + vrt_dataset.RasterYSize * vrt_geo[5],  # ymin
                      vrt_geo[0] + vrt_dataset.RasterXSize * vrt_geo[1],  # xmax
                      vrt_geo[3]  # ymax
                      ]
    
    warp=True
    if extent is not None:
        # assert (extent[0] > dataset_extent[0] and extent[1] > dataset_extent[1] and
        # extent[2] < dataset_extent[2] and extent[3] < dataset_extent[3]), f"You have specified an extent that is not entirely within the dataset!!\n{dataset_extent}"

        # This is done so that we don't haver large arrays with lots of no value data. Extent is trimmed to the size of the dataset
        extent = list(extent)
        # i = 0
        # for e1, e2 in zip(extent, dataset_extent):
        #     if i < 2 and e1 < e2:
        #         extent[i] = e2
        #     elif e1 > e2:
        #         extent[i] = e2
        #     i += 1

        warp_options = gdal.WarpOptions(
            format='GTiff',
            dstSRS=projection,
            dstNodata=-9999,
            outputBounds=extent,
            outputType=gdal.GDT_Float32,  
            multithread=True, 
            creationOptions = ["COMPRESS=DEFLATE", "PREDICTOR=3","BIGTIFF=YES"]
            )          
    elif clip is not None:
        shp = ogr.Open(clip)
        layer = shp.GetLayer()
        #shp_extent = layer.GetExtent()

        # assert (shp_extent[0] > dataset_extent[0] and shp_extent[1] > dataset_extent[1] and
        # shp_extent[2] < dataset_extent[2] and shp_extent[3] < dataset_extent[3]), "You are trying to clip something that is not entirely within the dataset!!"

        name = layer.GetName()
        shp = None
        warp_options = gdal.WarpOptions(
            format='GTiff', 
            dstSRS=projection, 
            dstNodata=-9999,
            utlineDSName=clip, 
            cutlineLayer=name, 
            cropToCutline=True, 
            outputType=gdal.GDT_Float32, 
            multithread=True, 
            creationOptions = ["COMPRESS=DEFLATE", "PREDICTOR=3","BIGTIFF=YES"])
    else:
        warp_options = gdal.WarpOptions(
            format='GTiff', 
            dstSRS=projection, 
            dstNodata=-9999, 
            outputType=gdal.GDT_Float32, 
            multithread=True, 
            creationOptions = ["COMPRESS=DEFLATE", "PREDICTOR=3","BIGTIFF=YES"])

    try:
        gdal.Warp(output, vrt_dataset, options=warp_options)
    except RuntimeError as e:
        try:
            disk, required = re.findall(r"(\d+)", str(e))
            raise MemoryError(f"Need {_sizeof_fmt(int(required))}; {_sizeof_fmt(int(disk))} of space on this machine")
        except:
            print(e)

    # Clean up the VRT dataset
    vrt_dataset = None

def DEM_Parser(folder: str, output: str, extent: tuple or list = None, clip: str = None,
               pattern: str = r"(\w\d{2})(\w\d{3})-(\w\d{2})(\w\d{3}).+", 
               file_pattern = r"(\w\d{2})(\w\d{3}).+"):
    """
    """
    # some checks
    if clip is None and extent is None:
        raise NotImplementedError
    if extent is not None and len(extent) != 4:
        raise ValueError(f'Invalid extent: {extent}')
    else:
        minx1, miny1, maxx1, maxy1 = extent
    if clip is not None:
        if not os.path.exists(clip):
            raise ValueError(f'Clip does not exist: {clip}')
        shp = ogr.Open(clip)
        layer = shp.GetLayer()
        extent = layer.GetExtent()
        minx1, maxx1, miny1, maxy1 = extent

    sub_folders = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isdir(os.path.join(folder,f))]
    folders_to_inspect = []
    for folder in sub_folders:
        folder_extent = [int(x[1:]) if x[0] in 'NE' else -int(x[1:]) for x in re.findall(pattern, folder)[0]]
        miny2, minx2, maxy2, maxx2 = folder_extent
        if (minx1 <= maxx2 and maxx1 >= minx2 and miny1 <= maxy2 and maxy1 >= miny2):
            folders_to_inspect.append(folder)
    
    files = []
    for folder in folders_to_inspect:
        for file in os.listdir(folder):
            if not file.endswith('.tif'):
                continue
            file_extent = [int(x[1:]) if x[0] in 'NE' else -int(x[1:]) for x in re.findall(file_pattern, file)[0]]
            file_extent.reverse()
            file_extent += [file_extent[0] + 1, file_extent[1] + 1]
            minx2, miny2, maxx2, maxy2 = file_extent
            if (minx1 <= maxx2 and maxx1 >= minx2 and miny1 <= maxy2 and maxy1 >= miny2):
                files.append(os.path.join(folder,file))

    if len(files) == 0:
        raise ValueError("No files found that are in the extent. Check the extent, filenames, and the patterns!")
    elif clip == None:
        PrepareDEM(files, output, extent)
    else:
        PrepareDEM(files, output, clip=clip)
        return output

def _sizeof_fmt(num:int) -> str:
    """
    Take in an int number of bytes, outputs a string that is human readable
    """
    for unit in ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB"):
        if abs(num) < 1024.0:
            return f"{num:3.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} YB"

def _Get_Raster_Details(raster_file: str):
    """
    Get important information from a raster file
    """
    data = gdal.Open(raster_file, GA_ReadOnly)
    try:
        projection = data.GetProjection()
    except RuntimeError as e:
        print(e)
        print("Consider adding this to the top of your python script, with your paths:")
        print('os.environ[\'PROJ_LIB\'] = "C:\\Users\\USERNAME\\.conda\\envs\\gdal\\Library\\share\\proj \
            os.environ[\'GDAL_DATA\'] = "C:\\Users\\USERNAME\\.conda\\envs\\gdal\\Library\\share"')
        
    geoTransform = data.GetGeoTransform()
    ncols = int(data.RasterXSize)
    nrows = int(data.RasterYSize)
    minx = geoTransform[0]
    dx = geoTransform[1]
    maxy = geoTransform[3]
    dy = geoTransform[5]
    maxx = minx + dx * ncols
    miny = maxy + dy * nrows

    if maxx == minx or maxy == miny:
        raise Exception(f"{raster_file} is {maxx-minx}x{maxy-miny}, which is not supported.")

    return minx, miny, maxx, maxy, dx, dy, ncols, nrows, geoTransform, projection



