import os
import subprocess
import re
import numpy as np
import pandas as pd
import concurrent.futures
from datetime import datetime
import time
import geopandas as gpd
from shapely.geometry import box, MultiLineString

try:
    import gdal
    import ogr
    import osr
    from gdalconst import GA_ReadOnly
except:
	from osgeo import gdal, ogr, osr
	from osgeo.gdalconst import GA_ReadOnly

gdal.UseExceptions()
# os.environ['PROJ_LIB'] = "C:\\Users\\lrr43\\.conda\\envs\\gdal\\Library\\share\\proj"
# os.environ['GDAL_DATA'] = "C:\\Users\\lrr43\\.conda\\envs\\gdal\\Library\\share"

def Preprocess(dem_folder: str, lu_folder: str, streams: str, flowfile_to_sim: str, temp_folder: str, 
               extent: tuple or list = None, clip: str = None, reach_id: str = 'LINKNO', 
               flows_to_use: str or list = None, reclassify_dict: dict = None, overwrite: bool = False):
    """
    Main function that will preprocess everything
    """

    out_dem = os.path.join(temp_folder,'dem.tif')
    os.makedirs(out_dem, exist_ok=True)
    out_lu = os.path.join(temp_folder, 'lu.tif')
    os.makedirs(out_lu, exist_ok=True)
    out_strm = os.path.join(temp_folder, 'strm.tif')
    os.makedirs(out_strm, exist_ok=True)
    out_rapid = os.path.join(temp_folder, 'rapid.txt')
    os.makedirs(out_rapid, exist_ok=True)

    DEM_Parser(dem_folder, out_dem, extent, clip)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        # Submit the functions to the executor
        future1 = executor.submit(LU_Parser, lu_folder, out_dem, out_lu, reclassify_dict)
        future2 = executor.submit(PrepareStream, out_dem, streams, out_strm, reach_id)
        
        concurrent.futures.wait([future2], return_when=concurrent.futures.FIRST_COMPLETED)

        MakeRAPIDFile(flowfile_to_sim, out_strm,out_rapid,flows_to_use,reach_id)

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
    _checkExistence([dem_file_path])

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
    if '["EPSG","4326"]' not in projection:
        # Reproject all GeoTIFFs to EPSG:4326
        projection = osr.SpatialReference()
        projection.ImportFromEPSG(4326)

    # Create an in-memory VRT (Virtual Dataset) for merging the GeoTIFFs
    vrt_options = gdal.BuildVRTOptions(resampleAlg='nearest')
    vrt_dataset = gdal.BuildVRT('', [tiff_file for tiff_file in tiff_files], options=vrt_options)
    options = gdal.WarpOptions(outputType=gdal.GDT_UInt16, width=ncols, height=nrows, outputBounds=(minx, miny, maxx, maxy), 
                                dstSRS=projection, resampleAlg=gdal.GRA_NearestNeighbour, format='VRT')
    final_dataset = gdal.Warp('', vrt_dataset, options=options) 

    hDriver = gdal.GetDriverByName("GTiff")
    out_file = hDriver.Create(out_path, xsize=ncols, ysize=nrows, bands=1, eType=gdal.GDT_UInt16)
    out_file.SetGeoTransform(final_dataset.GetGeoTransform())
    out_file.SetProjection(projection)

    print("Writing out array... ", end='')
    block_size = 8192
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

    print('finished')
    out_file = None
    final_dataset = None

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
    
    if extent is not None:
        # assert (extent[0] > dataset_extent[0] and extent[1] > dataset_extent[1] and
        # extent[2] < dataset_extent[2] and extent[3] < dataset_extent[3]), f"You have specified an extent that is not entirely within the dataset!!\n{dataset_extent}"

        # This is done so that we don't haver large arrays with lots of no value data. Extent is trimmed to the size of the dataset
        extent = list(extent)
        i = 0
        for e1, e2 in zip(extent, dataset_extent):
            if i < 2 and e1 < e2:
                extent[i] = e2
            elif e1 > e2:
                extent[i] = e2
            i += 1

        warp_options = gdal.WarpOptions(format='GTiff', dstSRS=projection, dstNodata=0, outputBounds=extent)
    elif clip is not None:
        shp = ogr.Open(clip)
        layer = shp.GetLayer()
        shp_extent = layer.GetExtent()

        # assert (shp_extent[0] > dataset_extent[0] and shp_extent[1] > dataset_extent[1] and
        # shp_extent[2] < dataset_extent[2] and shp_extent[3] < dataset_extent[3]), "You are trying to clip something that is not entirely within the dataset!!"

        name = layer.GetName()
        shp = None
        warp_options = gdal.WarpOptions(format='GTiff', dstSRS=projection, dstNodata=0, cutlineDSName=clip, cutlineLayer=name, cropToCutline=True)
    else:
        warp_options = gdal.WarpOptions(format='GTiff', dstSRS=projection, dstNodata=0)

    print(f'Writing {output}... ', end='')
    try:
        gdal.Warp(output, vrt_dataset, options=warp_options)
    except RuntimeError as e:
        try:
            disk, required = re.findall(r"(\d+)", str(e))
            raise MemoryError(f"Need {_sizeof_fmt(int(required))}; {_sizeof_fmt(int(disk))} of space on this machine")
        except:
            print(e)

    print('finished')

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
    if clip is not None:
        if not os.path.exists(clip):
            raise ValueError(f'Clip does not exist: {clip}')
        shp = ogr.Open(clip)
        layer = shp.GetLayer()
        extent = layer.GetExtent()

    sub_folders = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isdir(os.path.join(folder,f))]
    folders_to_inspect = []
    for folder in sub_folders:
        folder_extent = [int(x[1:]) if x[0] in 'NE' else -int(x[1:]) for x in re.findall(pattern, folder)[0]]
        minx1, miny1, maxx1, maxy1 = extent
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
            minx1, miny1, maxx1, maxy1 = extent
            minx2, miny2, maxx2, maxy2 = file_extent
            if (minx1 <= maxx2 and maxx1 >= minx2 and miny1 <= maxy2 and maxy1 >= miny2):
                files.append(os.path.join(folder,file))

    if len(files) == 0:
        raise ValueError("No files found that are in the extent. Check the extent, filenames, and the patterns!")
    elif len(files) == 1:
        return files[0]
    elif clip == None:
        print(files)
        PrepareDEM(files, output, extent)
    else:
        PrepareDEM(files, output, clip=clip)
        return output

def LU_Parser(folder: str, dem: str, output: str, reclassify: dict = None,  
              pattern: str = r"(\w\d{3})(\w\d{2}).+\.tif"):
    minx1, miny1, maxx1, maxy1, _, _, ncols, nrows, _, projection = _Get_Raster_Details(dem)
    files = []

    for file in os.listdir(folder):
        file_extent = [int(x[1:]) if x[0] in 'NE' else -int(x[1:]) for x in re.findall(pattern, file)[0]]
        file_extent[1] -= 20
        file_extent += [file_extent[0]+20, file_extent[1]+20]
        minx2, miny2, maxx2, maxy2 = file_extent
        if (minx1 <= maxx2 and maxx1 >= minx2 and miny1 <= maxy2 and maxy1 >= miny2):
            files.append(os.path.join(folder,file))

    if len(files) == 0:
        raise ValueError("No files found that are in the extent. Check the extent, filenames, and the patterns!")
    PrepareLU(dem,files,output, reclassify)
    
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

def _readDatasetToArray(tif: str) -> np.ndarray:
    Dataset = gdal.Open(tif, GA_ReadOnly)
    Band = Dataset.GetRasterBand(1)
    return np.array(Band.ReadAsArray(0))

def PrepareStream(dem_file_path: str, shapefile: str, out_path: str, field: str ='COMID') -> None:
    """
    Make a stream raster, with each cell having a value that represents the COMID of the stream in that location. 

    Returns nothing, but creates the STRM file.

    Parameters
    ----------
    dem_file_path : string, os.path object
        Path to DEM file, or a folder containing a DEM file
    shapefile : string, os.path object
        Path to stream shapefile
    out_path : string, os.path object
        Path, including name and file extension, of the output
    field : string, optional
        The label of the field containg the unique stream identifiers, usually COMID or HydroID

        .. versionadded:: 0.1.0

    Returns
    -------
    PrepareStream : None
        Only creates a .tiff file

    Notes
    -----
    We automatically assume that the field is 'COMID' unless otherwise stated.

    Examples
    --------

    """

    _checkExistence([dem_file_path, shapefile])

    (minx, miny, maxx, maxy, _, _, ncols, nrows, _, _) = _Get_Raster_Details(dem_file_path)
    LayerName = os.path.basename(shapefile).split('.')[-2]

    options = gdal.RasterizeOptions(noData=0, outputType=gdal.GDT_UInt32, attribute=field, width=ncols, height=nrows, 
                                    outputBounds=(minx, miny, maxx, maxy), layers=LayerName)
    gdal.Rasterize(out_path, shapefile, options=options)

def _checkExistence(paths_to_check: list) -> None:
    for path in paths_to_check:
        if not os.path.exists(path):
            raise IOError(f'{path} does not exist')

def _returnBasename(path: str) -> str:
    # For cross compatibility across different OS.
    # See: https://stackoverflow.com/questions/8384737/extract-file-name-from-path-no-matter-what-the-os-path-format
    head, tail = os.path.split(path)
    return tail or os.path.basename(head)

def _sizeof_fmt(num:int) -> str:
    """
    Take in an int number of bytes, outputs a string that is human readable
    """
    for unit in ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB"):
        if abs(num) < 1024.0:
            return f"{num:3.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} YB"

def MakeRAPIDFile(mainfile: str, stream_raster: str, out_path: str, flows: str or list = None, field: str='LINKNO') -> None:
    """
    Make a RAPID flow file from a shapefile or from a csv of COMID-max flow pairs, as seen in the Notes section.

    Returns nothing, but creates the RAPID flow file

    Parameters
    ----------
    mainfile : string, os.path object
        Either a stream shapefile or a csv of flow pairs. It's type will be automatically detected and proccessed accordingly.
    stream_raster : string, os.path object
        Stream raster to read from. If folder, then it will search for the first .tif
    out_path : string, os.path object
        Out path to stream shapefile
    flows : str, list
        List of flows or one flow as a string to be considered (present in the mainfile). If none and the input file is a csv-like input,
        then the flows are assumed to be the other columns
    field : string, optional
        The label of the field containg the unique stream identifiers, usually COMID or HydroID

        .. versionadded:: 0.1.0

    Returns
    -------
    MakeRAPIDFile : None
        Only creates a RAPID flow file

    Notes
    -----
    We automatically assume that the field is 'LINKNO' unless otherwise stated.

    An example of an input csv, with two lines of information after the header, is as follows:

    COMID,MaxFlow,50_year
    914360,134.56,67.23
    914370,40.25,15.66

    Examples
    --------
    The following three inputs yield the same .txt file:

    >>> MakeRAPIDFile(maxFlow.txt,'testSTRM.tif','testFLOW.txt', ['100_year'])
    >>> MakeRAPIDFile(maxFlow.csv,'testSTRM.tif','testFLOW.txt', ['100_year'])
    >>> MakeRAPIDFile(river.shp,'testSTRM.tif','testFLOW.txt', ['100_year'])

    The text file:

    ROW COL COMID 100_year\n
    8 143 13071247 10\n
    9 141 13071247 10n\n
    ...

    """
    _checkExistence([mainfile, stream_raster])  
    if isinstance(flows, str):
        flows = [flows]

    if os.path.isdir(stream_raster):
        files = os.listdir(stream_raster)
        files = [f for f in files if f.endswith('.tif')]
        if files == []:
            raise ValueError(f"Couldn't find any .tif files in {stream_raster}")
        if len(files) > 1:
            print(f"WARNING: multiple files found, using {files[0]}")
        stream_raster = os.path.join(stream_raster, files[0])

    print(mainfile)

    # If a shapefile/gpkg
    if mainfile.endswith(('.shp', '.gpkg')) and os.path.isfile(mainfile):
        if flows == None:
            raise ValueError('Must specify "flows" when using .shp or .gpkg inputs')
        
        if mainfile.endswith('.shp'):
            driver = ogr.GetDriverByName("ESRI Shapefile")
        else:
            driver = ogr.GetDriverByName("GPKG")

        dataSource = driver.Open(mainfile, 0)
        layer = dataSource.GetLayer()
        layerDefinition = layer.GetLayerDefn()
        layer_list = []

        for i in range(layerDefinition.GetFieldCount()):
            layer_name = layerDefinition.GetFieldDefn(i).GetName()
            layer_list.append(layer_name)

        # Make sure flows and field are in the shapefile
        check = all(item in layer_list for item in flows)
        if (check is False):
            raise ValueError("One of the flows fields is not in the shapefile")
        if (not field in layer_list):
            raise ValueError("The field %s is not in the shapefile" % field)
        
        # Make a dataframe to store the flows with the ids
        ids = []
        flow_list = []
        for feature in layer:
            ids.append(feature.GetField(field))
            flow_list.append([feature.GetField(flow) for flow in flows])

        flow_list = np.array(flow_list)
        df = pd.DataFrame({field:ids})
        for i, flow in enumerate(flows):
            df[flow] = flow_list[:,i]
            df[flow] = df[flow].astype(float)
        
        dataSource = None

    # If a .txt or .csv
    elif mainfile.endswith(('.csv', '.txt')) and os.path.isfile(mainfile): 
        df = (
            pd.read_csv(mainfile, sep=',')
            .drop_duplicates(field)
            .dropna()
        )
        
        if flows != None:
            check = all(item in list(df) for item in flows)
            if (check is False):
                raise ValueError(f"One of the flows fields is not in the file\nflows entered: {flows}, columns found: {list(df)}")
            df = df[[field] + flows]
    else:
        raise IOError(f"The mainfile has the extension of {os.path.splitext(mainfile)[-1]}, which is not recognized. Please use a .shp, .txt, or .csv file")
            
    #Now look through the Raster to find the appropriate Information and print to the FlowFile
    data = gdal.Open(stream_raster, GA_ReadOnly)
    band = data.GetRasterBand(1)
    data_array = band.ReadAsArray()

    indices = np.where(data_array > 0)
    matches = df[df[field].isin(data_array[indices])].shape[0]

    if matches != df.shape[0]:
        print(f"WARNING: {matches} ids out of {df.shape[0]} from your input file are present in the stream raster...")
    
    (
        pd.DataFrame({'ROW': indices[0], 'COL': indices[1], field: data_array[indices]})
        .merge(df, on=field, how='left')
        .fillna(0)
        .to_csv(out_path, sep=" ", index=False)
    )

    data = None

def AutoRoute(EXE: str, DEM: str, stream: str, RAPID: str, LandUseSameRes: str, out_path: str, Mannings: str, RAPIDflow: str, 
              Spatial_Units: str = 'deg', X_distance: int = 2000, Q: float = 1.024, Prev_D: int = 1, Man_n = None, 
              LandUseDiffRes: bool = False, Dir_dist: int = 1, Slope_dist: int = 11, Weight_angles:int = 0, 
              Dist_past_WE: int = 0, Stream_lower_limit: float = 0,Stream_upper_limit: float = 1000000000000000, 
              Degree_manip: float = 6.5, Degree_interval: float = 2.1, LowSpotRange: int = 2, LowSpotRangeM: float = 0, 
              LowSpotBox: bool = False, LowSpotBoxSize: int = 1, LowSpotFindFlat: bool = False, FindFlatCutoff: float = float('inf'), 
              Database: bool = True, DatabaseIts: int = 15, RAPIDid: str = 'reach_id', RAPIDbaseflow: str = '', 
              Rapid_da_flow: str = "", Bathy_mthd: int = 0, Bathy_max_depth: float = 0.2, Bathy_y_Shallow: float = 0.2, 
              Bathy_side_slope: float = 2.0,
              SubtractBaseflow: bool = False, RowColFromRAPID: bool = True, Bathy: bool = False, BathyAlpha: float = None,
              StartRow: int = 0, EndRow=None, AdjustFlow: float = 1, Uniform: bool = False, Proportional: bool = False, 
              Exponential: bool = False, ExPrecip: bool = False, Log10: bool = False, FlowAlpha: float = 1, FlowBeta: float = 0.1, 
              FlowGamma: float = 1, PrecipConst: float = 0.2, PrecipExp: float = 0.2, SQFT: bool = False, ACRE: bool = False, 
              SQMI: bool = False, CMS: bool = False, M2: bool = None, UseShortTW: bool = False, StrmMask: bool = None, 
              StrmMaskID: str = '', BathyOut: str = '', FloodPoints: str = '', DepthPoints: str = '', BathyPoints: str = '', 
              PointsJust3: bool = False, XSection: str = '', Metafile: str = datetime.now().strftime("%Y-%d-%m_%H-%M-%S") + ".txt", 
              DONTRUN: bool = False, MIFN: str = 'tempMIF.txt', Silent: bool = True):
    """
    Make a VDT file with AutoRoute executable. Most parameters are the suggested default values.

    Returns nothing, but creates the VDT file

    Parameters
    ----------
    EXE : string
        Path to the AutoRoute Executable
    DEM : string
        Digital elevation model. The raster can be projected or geographic, but the elevation values must be in meters.
    stream_raster : string
        Stream raster made in a preprocessing step. Tells AutoRoute what stream raster to use.
    RAPID : string
        Input text file that specifies for each stream cell the row, col, HydroID (or similar COMID, etc.), and flow rates.  In the file you can specify several flow rates for each stream cell.  The Flow_File is tab- or space-separated and is typically created in a preprocessing step.  An example Flow_File is shown below:
    LandUseSameRes : string
        Land Use raster that has a spatial resolution and spatial extent the EXACT SAME as the DEM and Stream Raster files.  The values of the land use raster must be integer.
    Out_path : string
        If Database is True, Output file that for each stream cell it prints the stream identifier (COMID, HydroID, etc.), row, col, elevation, baseflow, and for several flow rates the flow, velocity, top-width, and water surface elevation associated.  This output file is used as the database file when using FloodSpreader. Otherwise, Output file that for each stream cell it prints the stream identifier (COMID, HydroID, etc.), row, col, flow, velocity, depth, top-width, elevation, and water surface elevation.  This output file is used as an input for FloodSpreader.
    Mannings : string
        .txt file table that corresponds a land use type to a Mannings n value. Each cell within the model is assigned a Mannings n value. The description should have NO SPACES OR TABS
    RAPIDflow : string
        Specifies the flow rates that AutoRoute uses.  The text (String) must be present as a column heading in the Flow_RAPIDFile file.
    Spatial_Units : string
        Spatial units of the x- and y- coordinates in the input rasters. The only options are “km”, “m”, and “deg.” If “deg” is specified then Latitude must also be defined in AutoRoute versions prior to AutoRoute Version 2.0, but this is not supported in this python package
    X_distance : int, float
        This is the distance (meters) from the stream cell that will be sampled from the DEM to create a cross-section profile. A value of 100 would indicate that 200 meters would be sampled (100 meters on each side of the stream cell). Higher values are needed for wide rivers, but with these larger values also comes computational burden. It is really up the modeler to balance the need for sampling wide cross sections with the computational cost.
    Q : int, float
        AutoRoute uses an iterative method to calculate the flow depth at each cross-section. The flow depth is iteratively increased until the calculated flow rate (Qcalc) is similar to the flow rate assigned to the stream cell (Qassigned). The cross-section is not used if Qcalc > Q_Limit * Qassigned Default value for Q_Limit is 1.1.  This means that if the Qcalc is 10% higher than Qassigned the cross-section is thrown out.  Some detail is provided in Follum et al. (2020), but in that paper Q_Limit is set to 1.01, which would throw out more cross-sections.
    Prev_D : int
        The value is either 0 or 1. If this card is specified with a value of 1 and a cross-section is going to be thrown out because Qcalc > Q_Limit * Qassigned, , the previous depth value (which didn’t violate the Q_Limit criteria) is used for the cross-section.  However, Qcalc could be less (significantly less sometimes) than Qassigned. This option should almost always be used and therefore the default value is now set to 1.
    Man_n : NoneType, int, float
        Specifies the Mannings n value for the entire model. This is overwritten if a Land Use raster is defined and a Mannings n table is utilized (see LU_Manning_n and LU_Raster or LU_Raster_SameRes)
    LandUseDiffRes : bool
        Indicates that the land Use raster has a spatial resolution and spatial extent greater than the DEM or Stream raster files. This option was included because land use rasters are typically 30m resolution and DEM rasters are typically 10m resolution.  It also allows the modeler to use a single land use raster when simulating a large area that is made-up of several DEM rasters.
    Dir_dist : int
        This is the number of cells upstream and downstream the model will search to calculate the stream direction for each stream cell. The cross-sections are sampled from angles perpendicular to the stream direction. Based on sinuosity of the rivers, this is really up to the modeler to determine the appropriate value. Default value is 1, but is often set to 10 depending on the model application.
    Slope_dist : int
        This is the number of cells upstream and downstream the model will search to calculate the stream slope for each stream cell. This is really up to the modeler to determine the appropriate value.
    Weight_angles : int, float
        For calculating slope and stream direction it was thought at one time the stream cells farther upstream/downstream from the stream cell of interest should carry a different weight than those that are closer. A value greater than 0 would increase the weight of cells farther from the stream cell of interest. Default value is 0, which would apply no weight based on distance from the cell. This has not been shown to be all that useful and should likely not be used.
    Dist_past_WE : int
        When printing out cross-section ordinates this value indicates how many ordinates past the water edge you want printed out with the cross-section. This was mainly used to provide ingress/egress information for mobility modeling. Default value is 0. When used it was typically set to 4.
    Stream_lower_limit : int, float
        Any stream cell with a stream raster value below this specified value is omitted (no cross-section sampled nor flood depth calculated) This is useful in omitting certain stream reaches. Also useful when simulating flows when the input stream raster is a flow accumulation raster (only cells that have a value greater than Str_Limit_Val are simulated). Default value is 0.0
    Stream_upper_limit : int, float
        Any stream cell with a stream raster value above this specified value is omitted (no cross-section sampled nor flood depth calculated) This is useful in omitting certain stream reaches. Also useful when simulating flows when the input stream raster is a flow accumulation raster (only cells that have a value less than UP_Str_Limit_Val are simulated). Default value is near infinity
    Degree_manip : float
        Degree Manip and Degree_Interval are both inputs that are followed by a floating point that represents an angle in degrees.  Both inputs are required for model to sample additional cross-sections at each stream cell. Degree Manip in conjunction with Degree_Interval are used for two reasons: 1.) Attempt to more accurately capture a cross-section perpendicular to the stream direction and 2.) Fill in the flood inundation map from AutoRoute (this is no longer necessary with FloodSpreader). As described in Follum et al. (2007), AutoRoute samples an initial cross-section.  Around sharp bends of the stream, a single perpendicular cross section may not adequately capture the floodplain geometry. Therefore, additional cross sections are developed by incrementally pivoting the cross-sectional orientation relative to the stream direction.  As shown in the figure below, Degree_Manip is the maximum angle (in degrees) that the cross-sections will pivot to either side of the original cross-section.  Degree_Interval (in degrees) specifies the incremental pivots about the original cross-section.
    Degree_interval : float
        See Degree_manip.
    LowSpotRange : int
        Tells AutoRoute to search for a low-spot in the cross-section profile.  The value associated with the parameter is the number of ordinates laterally along the cross-section searched to find the lowest point within the cross-section.  This parameter is often needed because the specified stream cell location is not always the lowest point within the cross-section.  The user needs to be aware as to not make this value too large because it is possible a low spot outside of the river channel could be found if the search distance is too large. If several cells have the lowest elevation analyzed, the cell closest to the modeled stream line is utilized. The default value is 2
    LowSpotRangeM : int, float
        Tells AutoRoute to search for a low-spot in the cross-section profile.  The value associated with the parameter is the distance laterally along the cross-section searched to find the lowest point within the cross-section.  This parameter is often needed because the specified stream cell location is not always the lowest point within the cross-section.  The user needs to be aware as to not make this value too large because it is possible a low spot outside of the river channel could be found if the search distance is too large. If several cells have the lowest elevation analyzed, the cell closest to the modeled stream line is utilized. The default value is 0.0.  If both Low_Spot_Range and Low_Spot_Dist_m are specified, the model will use Low_Spot_Dist_m.
    LowSpotBox : bool
        When “Low_Spot_Range_Box” is specified, the AutoRoute model samples a cross-section perpendicular to the stream direction.  The model then proceeds to move along each ordinate analyzing the ordinate elevation as well as the elevations of surrounding cells.  The number of ordinates analyzed is still defined by the “Low_Spot_Range” or “Low_Spot_Dist_m” input cards. Several elevation cells at each ordinate are analyzed instead of the interpolated ordinate elevations within the cross-section.  The interpolated ordinate elevations can mask the low-spots of the river, therefore the elevation cells of several cells surrounding the ordinate are analyzed.  The number of cells analyzed at each ordinate is defined using the “Low_Spot_Range_Box_Size” input card followed by an integer.  The default number is 1.  When using the “Low_Spot_Range_Box” input card the stream cell is moved to a lower-elevation cell once, and then various cross-sections are sampled from the new stream location.  This was done because the purpose of the multiple cross-sections was to better align a perpendicular cross-section from a known stream location, not to find the stream location itself.  The updated method better serves this purpose. If several cells have the lowest elevation analyzed, the cell closest to the modeled stream line is utilized.
    LowSpotBoxSize : int
        The number of cells analyzed at each ordinate when using “Low_Spot_Range_Box” is defined using the “Low_Spot_Range_Box_Size” input card followed by an integer.  The default number is 1, which would search the closest elevation cell to the ordinate and all the immediately surrounding cells in a square pattern.  If “Low_Spot_Range_Box_Size” is set to 1, 9 elevation cells are analyzed.  If “Low_Spot_Range_Box_Size” is set to 2, 25 elevation cells are analyzed.  If “Low_Spot_Range_Box_Size” is set to 3, 49 elevation cells are analyzed.
    LowSpotFindFlat : bool
        In many elevation datasets the elevations of large rivers are “burned-in”, giving them uniform elevations. The rivers may not have the lowest elevation within the cross-section profile, but they can be identified because they have a constant elevation.  The “Low_Spot_Find_Flat” input card was added to AutoRoute to help identify the centerline of larger rivers.  This feature only works when a stream line has a flow rate greater than a user-specified flow rate (“Low_Spot_Range_FlowCutoff”) which allows only larger rivers to implement the “Low_Spot_Find_Flat” method.  In the “Low_Spot_Find_Flat” method a cross-section is sampled and the model analyzes the 20 closest ordinates to the modeled centerline.  If several of the ordinates have the same elevation it is assumed this area of “flat” cells is a river and the centerline is moved to the center of these flat cells
    FindFlatCutoff : float
        The FindFlatCutoff input card with associated float value define the flow rate at which the “Low_Spot_Find_Flat” method can be used.  Typically, only larger rivers should utilize this option.  Therefore, the default value is near infinity.
    Database : bool
        See Out_path
    DatabaseIts : int
        When using the “Print_VDT_Database” option the AutoRoute model is simulating several flow rates and recording the information in an output VDT database file.  This input card allows the user to define how many flow iterations are simulated.  The default value is 15.
    RAPIDid : string
        Specifies the stream identifier that AutoRoute uses.  The text must be present as a column heading in the Flow_RAPIDFile file.  
    RAPIDbaseflow : string
        Specifies the base flow rates that AutoRoute uses.  The text (String) must be present as a column heading in the Flow_RAPIDFile file.   
    Rapid_da_flow: string, optional
        o	Highly recommended when using left bank- or right bank quadratic bathymetric profile.
        o	This card helps AutoRoute know the left bank and the right bank of a stream.  This allows for constant bathymetric profiles, especially when using a left bank quadratic or right bank quadratic bathymetric profile.
        o	The text (String) must be present as a column heading in the Flow_RAPIDFile file.
    Bathy_mthd: int, optional
        o	0 indicates the default parabolic bathymetric shape described in Follum et al. (2020).
        o	1 indicates Left Bank Quadratic
        o	2 indicates Right Bank Quadratic
        o	3 indicates Double Quadratic
        o	4 indicates Trapezoidal
        o	5 indicates Triangle
    Bathy_max_depth: float, optional
        o	The value associated with Bathymetry_XMaxDepth tells the model what percentage of the top width from the bank that the lowest elevation of the bathymetry profile begins (see figure below).
        o	Applies to trapezoidal, left bank quadratic, right bank quadratic, and double quadratic bathymetric profiles.
    Bathy_y_Shallow: float, optional
        o	The shallow portion of the bathymetric profile is calculated as the maximum depth times the value associated with Bathymetry_YShallow (see figure below).
        o	Applies to left bank quadratic, right bank quadratic, and double quadratic bathymetric profiles.
    Bathy_side_slope
        o	The value associated with Bathymetry_SideSlope is the side slope of the initial estimate of the trapezoidal bathymetric profile. However, the method AutoRoute uses to calculate the bathymetric profile cannot preserve the side slopes, therefore this option should not be used to control the final shape of the bathymetric profile.  The use of “Bathymetry_XMaxDepth” is a better way to control the final shape of the trapezoidal bathymetric profile.

    SubtractBaseflow : bool, optional
        Tells the AutoRoute model to account for baseflow by subtracting the baseflow rate from the target flow rate.  These flow rates are defined using RAPID_BaseFlow_Param and RAPID_Flow_Param. RAPID_Flow_ID, RAPID_Flow_Param, and RAPID_BaseFlow_Param must all be defined. The other option to account for bathymetry is to use the Bathymetry input card.  When both RAPID_Subtract_BaseFlow and Bathymetry are specified in the input file the model will default to using RAPID_Subtract_BaseFlow.     
    RowColFromRAPID : bool, optional
        The row and column for each stream cell is defined within the Flow_RAPIDFile file by the Row and Col headings.  If this card is not present then the model will use the Stream_File to define the row and column of each stream cell.
    Bathy : bool, optional
        Tells AutoRoute to estimate channel bathymetry using a power function.  The method is described in Follum et al. (2020)
    BathyAlpha : float, optional
        Allows the user to manually specify ẟ in the bathymetry power function (see description under Bathymetry).  Default ẟ value is 0.001.
    StartRow : int, optional
        This card specifies the row where the AutoRoute model will start sampling cross-sections and simulating flood events.  This option was implemented to save computational time by only simulating a smaller zone.  The default value is 0.
    EndRow : NoneType, int, optional
        This card specifies the row where the AutoRoute model will stop sampling cross-sections and simulating flood events.  This option was implemented to save computational time by only simulating a smaller zone.  The default value is set to the total number of rows.
    AdjustFlow : int, float, optional
        The flow rate that is inputted or calculated is simply multiplied by the specified ADJUST_FLOW_BY_FRACTION value.  If the Calculated flow value for a given cell is 50.2 cms and  ADJUST_FLOW_BY_FRACTION is set to 0.75, then the model will try to simulate a flow rate of 37.65 cms.
    Uniform : bool, optional
        Flag where flow (Q, cms) is set constant: Q = Flow_Alpha
    Proportional : bool, optional
        Flag where flow (Q, cms) is set proportional to the stream raster value (Sx,y): Q = Flow_Alpha *  Sx,y
    Exponential : bool, optional
        Flag typically used for flow regression equations.  Flow (Q, cms) is calculated using an exponential function based on drainage area (DA): Q = Flow_Alpha * DA^Flow_Beta       Flow_Alpha and Flow_Beta are user-defined parameters.  DA is calculated based on the stream raster cell values and the unit conversion cards 
    ExPrecip : bool, optional
        Flag typically used for flow regression equations.  Flow (Q, cms) is calculated using an exponential function based on drainage area (DA) and precipitation: Q = Flow_Alpha * DA^Flow_Beta + Flow_Prec_Const^Flow_Prec_Exp   Flow_Alpha, Flow_Beta, Flow_Prec_Const, and Flow_Prec_Exp are all user-defined parameters.  DA is calculated based on the stream raster cell values and the unit conversion card
    Log10 : bool, optional
        Flag typically used for flow regression equations.  Flow (Q, cms) is calculated using a logarithmic function based on drainage area (DA): Q = 10 ^ ( Flow_Gamma + Flow_Alpha * DA^Flow_Beta )     Flow_Alpha, Flow_Beta, and Flow_Gamma are user-defined parameters.  DA is calculated based on the stream raster cell values and the unit conversion cards
    FlowAlpha : int, float, optional
        User-defined parameter often utilized in flow regression equations
    FlowBeta : int, float, optional
        User-defined parameter often utilized in flow regression equations
    FlowGamma : int, float, optional
        User-defined parameter often utilized in flow regression equations
    PrecipConst : int, float, optional
        User-defined parameter often utilized in flow regression equations
    PrecipExp : int, float, optional
        User-defined parameter often utilized in flow regression equations
    SQFT : bool, optional
        Multiplies each stream cell value within the Stream_File by 10763910.4, which converts the cell value from km2 to ft2. AutoRoute initially calculates Drainage Area as km2 based on STR_IS_M2 and the Stream_Cell values, which are assumed to be flow accumulation values. This input card is typically used when using flow regression equations.  All of the Unit Conversion cards can be used at the same time and build upon each other.
    ACRE : bool, optional
        Multiplies each stream cell value within the Stream_File by 247.1, which converts the cell value from km2 to acre. AutoRoute initially calculates Drainage Area as km2 based on STR_IS_M2 and the Stream_Cell values, which are assumed to be flow accumulation values. This input card is typically used when using flow regression equations.  All of the Unit Conversion cards can be used at the same time and build upon each other.
    SQMI : bool, optional
        Multiplies each stream cell value within the Stream_File by 0.38610216, which converts the cell value from km2 to square miles. AutoRoute initially calculates Drainage Area as km2 based on STR_IS_M2 and the Stream_Cell values, which are assumed to be flow accumulation values.
    CMS : bool, optional
        Multiplies each flow rate by 0.0283168, which converts the cell value from cubic feet per second to cubic meters per second. This is typically used when using flow regression equations.
    M2 : NoneType, int, float, optional
        Defines resolution (in meters, not square meters) of the cellsize for purposes of flow calculations.  If this is not defined, the default is to average the x- and y-cellsize resolution.
    UseShortTW : bool, optional
        Tells AutoRoute to output only the cross-section with the shortest top-width.  Only an option when using Degree_Manip and Degree_Interval where several cross-sections are sampled for each stream cell.
    StrmMaskID : string, optional
        Outputs a raster of the Stream Identifier number or Flow Rate for each stream cell.  Mainly used to check values.  Used in Conjunction with Print_Strm_Mask.
    StrmMask : any, optional
        Only used in conjunction with Print_Strm_Mask_ID.  Value indicates the following for the output raster: 1) Print Flow Values for each stream cell (does not currently work) and 2) Print identifier for each stream cell.  This identifier is set equal to row*ncols+col.  Basically, this is the number of cell if you read top-to-bottom, left-to-right.
    BathyOut : string, optional
        The full path to an output raster file of the AutoRoute-generated bathymetry.  All cells that were not included in cross-sections have a value of 0.  The main purpose of this file is as an input into FloodSpreader to create a more complete bathymetric map.
    FloodPoints : string, optional
        (path) Prints the X (e.g. Longitude) and Y (e.g. Latitude) coordinates of each ordinate in the cross-section to a comma-separated text file. This text file can then be input into ArcGIS.  The format is XYZ.  The waters-edge cells are always 3.  If Pnts_Just_3 is set within the input file than all the ordinates are set to 3, otherwise the ordinates at the waters-edge (ends of the cross-section) are set to a value of 2.
    DepthPoints : string, optional
        (path) Prints the Longitude, Latitude, and the flow depth of each ordinate in the cross-section to a comma-separated text file.
    BathyPoints : string, optional
        (path) Prints the Longitude, Latitude, and the bathymetry elevation of each ordinate in the cross-section to a comma-separated text file.
    PointsJust3 : bool, optional
        When outputting a flood-points text file using Print_Flood_Pnts, if Pnts_Just_3 is set within the input file than all the ordinates are set to 3, otherwise the ordinates at the waters-edge (ends of the cross-section) are set to a value of 2.
    XSection : string, optional
        (path) Prints the cross-section sampled at each stream cell in a two-line format.  The first line indicates the stream locations, stream identifications, flow, velocity, flow area, wetted perimeter, top-width, flow depth, cell elevation, slope, and coordinates of the ends of the cross-section.  The second line gives the ordinates of the cross-section.
    Metafile : any, optional
        A metafile is written for every AutoRoute run.  The information includes the AutoRoute version used, the location and size of the rasters being used, the inputs that were chosen, simulation time, etc.  If no filename is given, the program defaults to using the datetime the function was run.  This input card lets you specify the name of the Meta File.
    DONTRUN : bool, optional
        If set to True, the program will not run AutoRoute and instead only create the Input File. It is recommended to change the MIF paramter to a desired location
    MIFN : string, optional
        Location and name of temporary input file that AutoRoute will use, unless DONTRUN is True, in which case it will not be deleted.
    Silent : bool, optional
        If True, nothing will appear in the console when running AutoRoute. Setting to False may be useful if you want to see the live output and process of AutoRoute

        .. versionadded:: 0.1.0

    Returns
    -------
    AutoRoute : None
        Creates some user-specified file output.

    Notes
    -----

    Examples
    --------

    ...

    """
    # Because we don't yet have access to AutoRoute source code, we must write a temporary text file, use it is an input to autoroute, and then delete it, unless the user wishes to keep it.
    # Check the validity of each input
    _checkExistence([DEM, stream, RAPID, LandUseSameRes, Mannings])

    if Spatial_Units != 'deg' and Spatial_Units != 'km' and Spatial_Units != 'm':
        raise ValueError('Spatial units must be either "deg", "km", or "m"')

    for intorfloat in [X_distance, Q, Degree_interval, Weight_angles,Stream_lower_limit, Stream_upper_limit, LowSpotRangeM, AdjustFlow, FlowAlpha, FlowBeta, FlowGamma, PrecipConst, PrecipExp]:
        if not isinstance(intorfloat, int) and not isinstance(intorfloat, float):
            raise ValueError(f'The value {intorfloat} is invalid. Please reenter the value as an integer or float.')

    for isint in [Prev_D, Dir_dist, Slope_dist, Dist_past_WE, LowSpotRange, LowSpotBoxSize, DatabaseIts, StartRow]:
        if not isinstance(isint, int):
            raise ValueError(f'The value {isint} is invalid. Please reenter the value as an integer.')

    for isfloat in[Degree_manip, FindFlatCutoff]:
        if not isinstance(isfloat, float):
            raise ValueError(f'The value {isfloat} is invalid. Please reenter the value as a float.')
            
    for isbool in [LandUseDiffRes, LowSpotBox, LowSpotFindFlat, Database, SubtractBaseflow, RowColFromRAPID, Bathy, Uniform, Proportional, Exponential, ExPrecip, Log10, SQFT, ACRE, SQMI, CMS, UseShortTW, PointsJust3, DONTRUN]:
        if not isinstance(isbool, bool):
            raise ValueError(f'The value {isbool} is invalid. Please reenter the value as a boolean.')

    for issmall in [X_distance, Dir_dist, Slope_dist]:
        if issmall <= 0:
            raise ValueError(f'A value passed was {issmall}, but it should be greater than zero')

    if Prev_D != 0 and Prev_D!= 1:
        raise ValueError(f"Prev_D must be either 0 or 1, not {Prev_D}")

    with open(MIFN,'w') as MIF:
        # Required inputs and outputs
        _AR_Write(MIF, 'DEM_File', DEM)
        _AR_Write(MIF, 'Stream_File', stream)
        if LandUseDiffRes == True:
            _AR_Write(MIF, 'LU_Raster', LandUseSameRes)
        else:
            _AR_Write(MIF, 'LU_Raster_SameRes', LandUseSameRes)
        _AR_Write(MIF, 'LU_Manning_n', Mannings)
        _AR_Write(MIF, 'Flow_RAPIDFile', RAPID)
        if RowColFromRAPID:
            _AR_Write(MIF, 'RowCol_From_RAPIDFile', '')
        _AR_Write(MIF, 'RAPID_Flow_ID', RAPIDid)
        _AR_Write(MIF, 'RAPID_Flow_Param', RAPIDflow)
        if SubtractBaseflow and RAPIDbaseflow!= '':
            _AR_Write(MIF, 'RAPID_Subtract_BaseFlow', '')
            _AR_Write(MIF, 'RAPID_BaseFlow_Param', RAPIDbaseflow)
            _AR_Write(MIF, 'RAPID_Flow_ID', RAPIDid)
        _AR_Write(MIF, 'Spatial_Units', Spatial_Units)
        if Database:
            _AR_Write(MIF, 'Print_VDT_Database', out_path)
            _AR_Write(MIF, 'Print_VDT_Database_NumIterations', DatabaseIts)
        else: 
            _AR_Write(MIF, 'Print_VDT', out_path)
        _AR_Write(MIF, 'Meta_File', Metafile)

        # Optional inputs
        _AR_Write(MIF, 'X_Section_Dist', X_distance)
        _AR_Write(MIF, 'Degree_Manip', Degree_manip)
        _AR_Write(MIF, 'Degree_Interval', Degree_interval)
        _AR_Write(MIF, 'Low_Spot_Range', LowSpotRange)
        _AR_Write(MIF, 'Q_Limit', Q)
        _AR_Write(MIF, 'Gen_Dir_Dist', Dir_dist)
        _AR_Write(MIF, 'Gen_Slope_Dist', Slope_dist)
        if Bathy:
            _AR_Write(MIF, "Bathymetry", "")
            _AR_Write(MIF, 'RAPID_BaseFlow_Param', RAPIDbaseflow)
            _AR_Write(MIF, "Bathymetry_Method", Bathy_mthd)
            _AR_Write(MIF, "Bathymetry_XMaxDepth", Bathy_max_depth)
            _AR_Write(MIF, "Bathymetry_YShallow", Bathy_y_Shallow)
            _AR_Write(MIF, "Bathymetry_SideSlope", Bathy_side_slope)
            if BathyAlpha is not None:
                _AR_Write(MIF, "Bathymetry_Alpha",BathyAlpha)
            if BathyOut:
                _AR_Write(MIF, "BATHY_Out_File", BathyOut)
            if Rapid_da_flow:
                _AR_Write(MIF, "RAPID_DA_or_Flow_Param", Rapid_da_flow)

        if Prev_D == 0:
            _AR_Write(MIF, 'Use_Prev_D_4_XS', 0)
        if Man_n != None:
            _AR_Write(MIF, 'Man_n', Man_n)
        _AR_Write(MIF, 'Weight_Angles', Weight_angles)
        if Dist_past_WE != 0:
            _AR_Write(MIF, 'Dist_X_PastWE', Dist_past_WE)
        if Stream_lower_limit != 0:
            _AR_Write(MIF, 'Str_Limit_Val', Stream_lower_limit)
        if Stream_upper_limit < 1000000000000000:
            _AR_Write(MIF, 'UP_Str_Limit_Val', Stream_upper_limit)
        if LowSpotRangeM!= 0:
            _AR_Write(MIF, 'Low_Spot_Dist_m', LowSpotRangeM)
        if LowSpotBox:
            _AR_Write(MIF, 'Low_Spot_Range_Box', '')
            _AR_Write(MIF, 'Low_Spot_Range_Box_Size', LowSpotBoxSize)
        if LowSpotFindFlat:
            _AR_Write(MIF, 'Low_Spot_Find_Flat', '')
            _AR_Write(MIF, 'Low_Spot_Range_FlowCutoff', FindFlatCutoff)
        if StartRow:
            _AR_Write(MIF, 'Layer_Row_Start',StartRow)
        if EndRow:
            _AR_Write(MIF, 'Layer_Row_End', EndRow)
        if AdjustFlow != 1:
            _AR_Write(MIF, 'ADJUST_FLOW_BY_FRACTION', AdjustFlow)
        if Uniform:
            _AR_Write(MIF, 'UNIFORM_FLOW', '')
            _AR_Write(MIF, 'Flow_Alpha', FlowAlpha)
        if Proportional:
            _AR_Write(MIF, 'PROPORTIONAL_FLOW', '')
            _AR_Write(MIF, 'Flow_Alpha', FlowAlpha)
        if Exponential:
            _AR_Write(MIF, 'EXPONENTIAL_FLOW', '')
            _AR_Write(MIF, 'Flow_Alpha', FlowAlpha)
            _AR_Write(MIF, 'Flow_Beta', FlowBeta)
        if ExPrecip:
            _AR_Write(MIF, 'EXPON_PRECIP_FLOW', '')
            _AR_Write(MIF, 'Flow_Alpha', FlowAlpha)
            _AR_Write(MIF, 'Flow_Beta', FlowBeta)
            _AR_Write(MIF, 'Flow_Prec_Const', PrecipConst)
            _AR_Write(MIF, 'Flow_Prec_Exp', PrecipExp)
        if Log10:
            _AR_Write(MIF, 'Flow_Alpha', FlowAlpha)
            _AR_Write(MIF, 'Flow_Gamma', FlowGamma)
            _AR_Write(MIF, 'Flow_Beta', FlowBeta)
        if SQFT:
            _AR_Write(MIF, 'CONVERT_DA_TO_SQFT', '')
        if ACRE:
            _AR_Write(MIF, 'CONVERT_DA_TO_ACRE', '')
        if SQMI:
            _AR_Write(MIF, 'CONVERT_DA_TO_SQMI', '')
        if CMS:
            _AR_Write(MIF, 'CONVERT_Q_CFS_TO_CMS', '')
        if M2:
            _AR_Write(MIF, 'STR_IS_M2', M2)

        # Optional outputs
        if UseShortTW:
            _AR_Write(MIF, 'Print_Rasters_With_ShortTW', UseShortTW)
        if StrmMaskID:
            _AR_Write(MIF, 'Print_Strm_Mask_ID', StrmMaskID)
            _AR_Write(MIF, 'Print_Strm_Mask', StrmMask)
        if FloodPoints:
            _AR_Write(MIF, 'Print_Flood_Pnts', FloodPoints)
            if PointsJust3:
                _AR_Write(MIF, 'Pnts_Just_3', '')
        if DepthPoints:
            _AR_Write(MIF, 'Print_Depth_Pnts', DepthPoints)
        if BathyPoints:
            _AR_Write(MIF, 'Print_Bathy_Pnts', BathyPoints)
        if XSection:
            _AR_Write(MIF, 'Print_XSections', '')
            _AR_Write(MIF, 'XSections_FileName', XSection)

    if not DONTRUN:
        if Silent:
            c = subprocess.call([EXE, MIFN], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        else:
            c = subprocess.call([EXE, MIFN])

def _AR_Write(MIF, Card, Argument) -> None:
    MIF.write(f"{Card}\t{Argument}\n")

def FloodSpreader(EXE: str, DEM: str, VDT: str, OutName: str, Database: bool = True, COMIDFile: str = '', AdjustFlow: float = 1, 
                  ARBathy: str = '', FSBathy: str = '', Linear_Interp_Bathy: bool = False, Inverse_Bathy: float = None,
                  TWDF: float = 1.5, AdjstFldDpthMthd: int = 1, WSEDist: int = 10, 
                  WSEStDev: float = 0.25, WSERmv3: bool = False, SpecifiedDpth: float = 10, UseARTW: bool = False, 
                  OutDEP: bool = False, OutFLD: bool = False, OutVEL: bool = False, OutWSE: bool = False, DONTRUN: bool = False, 
                  MIFN: str = 'tempMIF.txt', Silent: bool = True) -> None:
    """
    Create a floodmap, using a DEM, a VDT file, and usually a COMID file.

    Returns nothing, but can create a floodmap (.tif)

    Parameters
    ----------
    EXE : string
        Path to the FloodSpreader Executable
    DEM : string
        Path to DEM file. The raster can be projected or geographic, but the elevation values must be in meters.
    VDT : string
        Path to a certain file, see Database.
    OutName : string
        Path, name of output including file extension
    Database : bool, optional
        By default this is True. If True, COMIDFile must also be specified. If false, uses the Print_VDT file, which  is the main output from AutoRoute that is used as an input for FloodSpreader.  For each stream cell the file has the stream identifier (COMID, HydroID, etc.), row, col, flow, velocity, depth, top-width, elevation, and water surface elevation.  The model uses these values to create a flood map.
        Otherwise, uses the Print_VDT_Database file, which is used as the database file when using FloodSpreader.  The file is created using AutoRoute.  For each stream cell the file has the stream identifier (COMID, HydroID, etc.), row, col, elevation, baseflow, and for several flow rates the flow, velocity, top-width, and water surface elevation associated.  These values are all precomputed.  The flows that you want to simulate are given to the model using the Comid_Flow_File input card and associated file.  Comid_Flow_File is also required if using Print_VDT_Database.   
    COMIDFile : string optional
        This card and associated file are used in conjunction with the Print_VDT_Database card.  The Comid_Flow_File simply lists the stream identifiers (i.e. COMID, HydroID) with their associated flow in cms.
    AdjustFlow : int, float, optional
        When assigning flow rates in the model using “Comid_Flow_File”, the flow rate assigned to each stream cell using the COMID_Flow_File is multiplied by the specified floating point.  This is a simple way to change the flow values used within FloodSpreader without having to create a new COMID_Flow_File.  Default value is 1.0, which is no change to the defined flow rate.
    ARBathy : string, optional
        The full path to the AutoRoute-generated bathymetry.  All cells that were not included in cross-sections have a value of 0.  The main purpose of this file is as an input into FloodSpreader to create a more complete bathymetric map.
    FSBathy : string, optional
        The full path to the FloodSpreader-generated bathymetry.  The bathymetry is burned into the original DEM data.
    Linear_Interp_Bathy: bool, optional
        FloodSpreader has the option to use a simple linear interpolation to fill bathymetry estimates between cross-sections.  
    Inverse_Bathy: float, optional
        Default is 1.0. To simulate a continuous bathymetric surface an inverse-distance weighting approach is used.
    TWDF: int, float, optional
        As discussed in Follum et al., (2020), higher TopWidthDistanceFactor (α) values increase the impact that each stream cell has on surrounding cells when generating a flood inundation map using FloodSpreader. The paper tested several α values and found higher α values resulted in increased accuracy but also increased the computational burden. The paper found α=1.5 provided good coverage of the river floodplain while remaining computationally efficient.  However, those tests were performed at sites in the U.S. using floating-point elevation datasets.  The α value has a default value of 1.5.
    AdjstFldDpthMthd : int, optional
        An int betweeen 0 and 5. 0 means no methods to adjust flood depths are used; this is the default. The rest are different methods described below:
            1: Flood Bad Cells. If the Top Width or Depth value associated with a cell is less than 1e-16 and the Flood_BadCells flag is specified, then the Depth and Top Width used in the cell is set to the average for the stream identifier (basically the stream reach).  This option was included to omit outliers but has NOT proven to be very useful.
            2: Use AR Depths. Uses the flow depths values originally read-in from the VDT File (Print_VDT).
            3: Smooth WSE. Input card specified within the main input file telling FloodSpreader to omit outliers using a spatial averaging method.
            4: Use AR Depths StDev. In an effort to remove outliers, this flag tells FloodSpreader to set the depth and topwidth values of a stream cell to the average for the stream identifier if the value is outside of one standard deviation.  This option has never worked well. 
            5: Specify Depth. Optional input card that tells FloodSpreader to set a uniform depth for all stream cells.
    WSEDist : int, optional
        Used in conjunction with the AdjstFldDpthMthd=3 input card. An optional input card that specifies the proximity in which water surface elevation (WSE) values are analyzed.  The default value is 10, meaning that WSE values in a box 10 cells above, below, to the left, and to the right will be analyzed.  In total, stream cells within a 441 cell box (stream cell being analyzed is omitted) are analyzed to determine the mean and standard deviation of the WSE.
    WSEStDev : int, float, optional
        Used in conjunction with the AdjstFldDpthMthd=3 input card. An optional input card that specifies the threshold in which a WSE value is adjusted.  The default value is 0.25, meaning that if the WSE of the stream cell being analyzed is outside of 0.25 standard deviations from the mean it is adjusted.  For example, if the mean WSE of surrounding cells is 156.5 m and the SD for that stream reach is 3.2 m:; If the WSE of the stream cell being analyzed is 157.2 m, it will remain 157.2 m; If the WSE of the stream cell being analyzed is 159.5 m, it will be adjusted to 157.3 m (156.5m + 0.25 * 3.2m); If the WSE of the stream cell being analyzed is 154.2 m, it will be adjusted to 155.7 m (156.5m - 0.25 * 3.2m)
    WSERmv3 : bool, optional
        Used in conjunction with the AdjstFldDpthMthd=3 input card. An optional input card that removes the highest 3 flow elevations in the search radius (when using “FloodSpreader_SmoothWSE_SearchDist”).  This was a test to remove outliers.  This option did not work well for a testcase in Croatia, and should not be used until it has shown to be beneficial.
    SpecifiedDpth : int, float, optional
        Used in conjunction with the AdjstFldDpthMthd=5 input card. See AdjstFldDpthMthd.
    UseARTW : bool, optional
        Uses the top width values originally read-in from the VDT File (Print_VDT). Default is to use the average top width value for the stream identifier.
    OutDEP : bool, optional
        Output depth (m) map from FloodSpreader.  Cells that are not considered flooded have a value of 0.
    OutFLD : bool, optional
        Output flood map from FloodSpreader.  Cells that are not considered flooded have a value of 0.
    OutVEL : bool, optional
       Output flow velocity (m/s) map from FloodSpreader. Cells that are not considered flooded have a value of 0.
    OutWSE : bool, optional
        Output water surface depth (m) map from FloodSpreader.  Cells that are not considered flooded have the value from the DEM (DEM_File)
    DONTRUN : bool, optional
        If set to True, the program will not run AutoRoute and instead only create the Input File. It is recommended to change the MIF paramter to a desired location
    MIFN : string, optional
        Location and name of temporary input file that AutoRoute will use, unless DONTRUN is True, in which case it will not be deleted.
    Silent : bool, optional
        If True, nothing will appear in the console when running AutoRoute. Setting to False may be useful if you want to see the live output and process of AutoRoute
       
        .. versionadded:: 0.1.0


    Returns
    -------
    FloodSpreader : None
        Only creates a .tiff file

    Notes
    -----
    First three lines of an example COMID file:

    COMID,MaxFlow
    959684,32.12
    959690,40.32
    ...

    Examples
    --------

    """
    _checkExistence([EXE, DEM, VDT])
    if Database:
        _checkExistence([COMIDFile])

    for isbool in [Database, UseARTW, OutDEP, OutFLD, OutVEL, OutWSE, DONTRUN, Silent]:
        if not isinstance(isbool, bool):
            raise ValueError(f'The value {isbool} is invalid. Please reenter the value as a boolean.')
    
    for intorfloat in [AdjustFlow, TWDF, WSEStDev, SpecifiedDpth]:
        if not isinstance(intorfloat, int) and not isinstance(intorfloat, float):
            raise ValueError(f'The value {intorfloat} is invalid. Please reenter the value as an integer or float.')

    for isint in [WSEDist]:
        if not isinstance(isint, int):
            raise ValueError(f'The value {isint} is invalid. Please reenter the value as an integer.')

    if os.path.exists(MIFN):
        mode = 'a'
    else:
        mode = 'w'

    with open(MIFN, mode) as MIF:
        # Required inputs
        _AR_Write(MIF, "#FloodSpreader_Inputs", "")
        _AR_Write(MIF, 'DEM_File', DEM)
        if Database:
            _AR_Write(MIF, 'Print_VDT_Database', VDT)
            _AR_Write(MIF, 'Comid_Flow_File', COMIDFile)
        else: 
            _AR_Write(MIF, 'Print_VDT', VDT)

        _AR_Write(MIF, '', '')

        # Optional inputs
        _AR_Write(MIF, 'FS_ADJUST_FLOW_BY_FRACTION', AdjustFlow)
        if ARBathy:
            _AR_Write(MIF, 'BATHY_Out_File', ARBathy)
        if FSBathy:
            _AR_Write(MIF, 'FSOutBATHY', FSBathy)
        if Linear_Interp_Bathy:
            _AR_Write(MIF, "Bathy_LinearInterpolation", "")
        if Inverse_Bathy is not None:
            _AR_Write(MIF, "BathyTopWidthDistanceFactor", Inverse_Bathy)
        _AR_Write(MIF, 'TopWidthDistanceFactor', TWDF)
        if AdjstFldDpthMthd == 1:
            _AR_Write(MIF, 'Flood_BadCells', '')
        elif AdjstFldDpthMthd == 2:
            _AR_Write(MIF, 'FloodSpreader_Use_AR_Depths', '')
        elif AdjstFldDpthMthd == 3:
            _AR_Write(MIF, 'FloodSpreader_SmoothWSE', '')
            _AR_Write(MIF, 'FloodSpreader_SmoothWSE_SearchDist', WSEDist)
            _AR_Write(MIF, 'FloodSpreader_SmoothWSE_FractStDev', WSEStDev)
            _AR_Write(MIF, 'FloodSpreader_SmoothWSE_RemoveHighThree', WSERmv3)
        elif AdjstFldDpthMthd == 4:
            _AR_Write(MIF, 'FloodSpreader_Use_AR_Depths_StDev', '')
        elif AdjstFldDpthMthd == 5:
            _AR_Write(MIF, 'FloodSpreader_SpecifyDepth', SpecifiedDpth)
        if UseARTW:
            _AR_Write(MIF, 'FloodSpreader_Use_AR_TopWidth', '')

        # Outputs
        if OutDEP:
            _AR_Write(MIF, 'OutDEP', OutName)
        if OutFLD:
            _AR_Write(MIF, 'OutFLD', OutName)
        if OutVEL:
            _AR_Write(MIF, 'OutVEL', OutName)
        if OutWSE:
            _AR_Write(MIF, 'OutWSE', OutName)

    try:
        if Silent and not DONTRUN:
            c = subprocess.call([EXE, MIFN], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        elif not DONTRUN:
            c = subprocess.call([EXE, MIFN])
    except Exception as e:
        print(e)

def FloodSpreaderPy(DEMfile, VDT, OutName, Database=True, COMIDFile='', AdjustFlow=1, ARBathy='',FSBathy='', TWDF=1.5, WSEDist=10, 
                    WSEStDev=0.25, WSERmv3=False, SpecifiedDpth=10, JstStrmDpths = False, UseARTW=False, OutDEP=False, OutFLD = False, 
                    OutVEL=False, OutWSE=False, DONTRUN=False, MIFN='tempMIF.txt', Silent=True, FloodBadCells=False,UseARDepths=False, 
                    SmoothWSE=False, UseARDepthsStDev=False,SpecifyDepth=False):

    """
    Create a floodmap, using a DEM, a VDT file, and usually a COMID file.

    Returns nothing, but can create a floodmap (.tif)

    Parameters
    ----------
    DEM : string, os.path object
        Path to DEM file. The raster can be projected or geographic, but the elevation values must be in meters.
    VDT : string, os.path object
        Path to a certain file, see Database.
    OutName : string, os.path object
        Path, name of output including file extension
    Database : bool, optional
        By default this is True. If True, COMIDFile must also be specified. If false, uses the Print_VDT file, which  is the main output from AutoRoute that is used as an input for FloodSpreader.  For each stream cell the file has the stream identifier (COMID, HydroID, etc.), row, col, flow, velocity, depth, top-width, elevation, and water surface elevation.  The model uses these values to create a flood map.
        Otherwise, uses the Print_VDT_Database file, which is used as the database file when using FloodSpreader.  The file is created using AutoRoute.  For each stream cell the file has the stream identifier (COMID, HydroID, etc.), row, col, elevation, baseflow, and for several flow rates the flow, velocity, top-width, and water surface elevation associated.  These values are all precomputed.  The flows that you want to simulate are given to the model using the Comid_Flow_File input card and associated file.  Comid_Flow_File is also required if using Print_VDT_Database.
    COMIDFile : string, os.path object, optional
        This card and associated file are used in conjunction with the Print_VDT_Database card.  The Comid_Flow_File simply lists the stream identifiers (i.e. COMID, HydroID) with their associated flow in cms.
    AdjustFlow : int, float, optional
        When assigning flow rates in the model using “Comid_Flow_File”, the flow rate assigned to each stream cell using the COMID_Flow_File is multiplied by the specified floating point.  This is a simple way to change the flow values used within FloodSpreader without having to create a new COMID_Flow_File.  Default value is 1.0, which is no change to the defined flow rate.
    ARBathy : string, os.path object, optional
        The full path to the AutoRoute-generated bathymetry.  All cells that were not included in cross-sections have a value of 0.  The main purpose of this file is as an input into FloodSpreader to create a more complete bathymetric map.
    FSBathy : string, os.path object, optional
        The full path to the FloodSpreader-generated bathymetry.  As you can see, the bathymetry is burned into the original DEM data.
    TWDF: int, float, optional
        As discussed in Follum et al., (2020), higher TopWidthDistanceFactor (α) values increase the impact that each stream cell has on surrounding cells when generating a flood inundation map using FloodSpreader. The paper tested several α values and found higher α values resulted in increased accuracy but also increased the computational burden. The paper found α=1.5 provided good coverage of the river floodplain while remaining computationally efficient.  However, those tests were performed at sites in the U.S. using floating-point elevation datasets.  The α value has a default value of 1.5.
    AdjstFldDpthMthd : int, optional
        An int betweeen 0 and 5. 0 means no methods to adjust flood depths are used; this is the default. The rest are different methods described below:
            1: Flood Bad Cells. If the Top Width or Depth value associated with a cell is less than 1e-16 and the Flood_BadCells flag is specified, then the Depth and Top Width used in the cell is set to the average for the stream identifier (basically the stream reach).  This option was included to omit outliers but has NOT proven to be very useful.
            2: Use AR Depths. Uses the flow depths values originally read-in from the VDT File (Print_VDT).
            3: Smooth WSE. Input card specified within the main input file telling FloodSpreader to omit outliers using a spatial averaging method.
            4: Use AR Depths StDev. In an effort to remove outliers, this flag tells FloodSpreader to set the depth and topwidth values of a stream cell to the average for the stream identifier if the value is outside of one standard deviation.  This option has never worked well. 
            5: Specify Depth. Optional input card that tells FloodSpreader to set a uniform depth for all stream cells.
    WSEDist : int, optional
        Used in conjunction with the AdjstFldDpthMthd=3 input card. An optional input card that specifies the proximity in which water surface elevation (WSE) values are analyzed.  The default value is 10, meaning that WSE values in a box 10 cells above, below, to the left, and to the right will be analyzed.  In total, stream cells within a 441 cell box (stream cell being analyzed is omitted) are analyzed to determine the mean and standard deviation of the WSE.
    WSEStDev : int, float, optional
        Used in conjunction with the AdjstFldDpthMthd=3 input card. An optional input card that specifies the threshold in which a WSE value is adjusted.  The default value is 0.25, meaning that if the WSE of the stream cell being analyzed is outside of 0.25 standard deviations from the mean it is adjusted.  For example, if the mean WSE of surrounding cells is 156.5 m and the SD for that stream reach is 3.2 m:; If the WSE of the stream cell being analyzed is 157.2 m, it will remain 157.2 m; If the WSE of the stream cell being analyzed is 159.5 m, it will be adjusted to 157.3 m (156.5m + 0.25 * 3.2m); If the WSE of the stream cell being analyzed is 154.2 m, it will be adjusted to 155.7 m (156.5m - 0.25 * 3.2m)
    WSERmv3 : bool, optional
        Used in conjunction with the AdjstFldDpthMthd=3 input card. An optional input card that removes the highest 3 flow elevations in the search radius (when using “FloodSpreader_SmoothWSE_SearchDist”).  This was a test to remove outliers.  This option did not work well for a testcase in Croatia, and should not be used until it has shown to be beneficial.
    SpecifiedDpth : int, float, optional
        Used in conjunction with the AdjstFldDpthMthd=5 input card. See AdjstFldDpthMthd.
    JstStrmDpths : bool, optional
        ####
    UseARTW : bool, optional
        Uses the top width values originally read-in from the VDT File (Print_VDT). Default is to use the average top width value for the stream identifier.
    OutDEP : bool, optional
        Output depth (m) map from FloodSpreader.  Cells that are not considered flooded have a value of 0.
    OutFLD : bool, optional
        Output flood map from FloodSpreader.  Cells that are not considered flooded have a value of 0.
    OutVEL : bool, optional
       Output flow velocity (m/s) map from FloodSpreader. Cells that are not considered flooded have a value of 0.
    OutWSE : bool, optional
        Output water surface depth (m) map from FloodSpreader.  Cells that are not considered flooded have the value from the DEM (DEM_File)
    DONTRUN : bool, optional
        If set to True, the program will not run AutoRoute and instead only create the Input File. It is recommended to change the MIF paramter to a desired location
    MIFN : string, os.path object, optional
        Location and name of temporary input file that AutoRoute will use, unless DONTRUN is True, in which case it will not be deleted.
    Silent : bool, optional
        If True, nothing will appear in the console when running AutoRoute. Setting to False may be useful if you want to see the live output and process of AutoRoute
       
        .. versionadded:: 0.1.0


    Returns
    -------
    FloodSpreader : None
        Only creates a .tiff file

    Notes
    -----
    First three lines of an example COMID file:

    COMID,MaxFlow
    959684,32.12
    959690,40.32
    ...

    Examples
    --------

    """
    _checkExistence([DEMfile, VDT])
    if Database:
        _checkExistence([COMIDFile])

    for isbool in [Database, UseARTW, OutDEP, OutFLD, OutVEL, OutWSE, DONTRUN, Silent]:
        if not isinstance(isbool, bool):
            raise ValueError(f'The value {isbool} is invalid. Please reenter the value as a boolean.')
    
    for intorfloat in [AdjustFlow, TWDF, WSEStDev, SpecifiedDpth]:
        if not isinstance(intorfloat, int) and not isinstance(intorfloat, float):
            raise ValueError(f'The value {intorfloat} is invalid. Please reenter the value as an integer or float.')

    for isint in [WSEDist]:
        if not isinstance(isint, int):
            raise ValueError(f'The value {isint} is invalid. Please reenter the value as an integer.')


    Start_Time = time.time()

    # READ IN MIF FILE if needed

    if SpecifiedDpth < 0:
        raise ValueError(f'SpecifiedDpth ({SpecifiedDpth}) is invalid. Please reenter the value as a positive integer')
    if not (OutDEP or OutFLD or OutVEL or OutWSE):
        if not Silent: print('You do not have OUTPUTS for this model, just giving you a warning')
    
    # Get DEM info
    if not Silent: print('Using GDAL to open the rasters')
    gdal.AllRegister()

    DEMdata = gdal.Open(DEMfile, gdal.GA_ReadOnly)
    DEMprojection = DEMdata.GetProjection()
    DEMgeoTransform = DEMdata.GetGeoTransform()

    ncols = int(DEMdata.RasterXSize)
    nrows = int(DEMdata.RasterYSize)
    cellsize = DEMgeoTransform[1]
    yll = DEMgeoTransform[3] - nrows * np.abs(DEMgeoTransform[5])
    yur = DEMgeoTransform[3]
    dy = DEMgeoTransform[5]

    # Make arrays
    DEMBand = DEMdata.GetRasterBand(1)
    DEM = np.array(DEMBand.ReadAsArray(0), dtype=np.float32)
    del DEMdata

    U = np.zeros(DEM.shape,  dtype=np.float32) # Numerator in IDW2 Calculation
    L = np.zeros(DEM.shape,  dtype=np.float32) # Denominator in IDW2 Calculation
    U_V = np.zeros(DEM.shape,  dtype=np.float32) # Numerator in IDW2 Calculation for Velocity
    U_TW = np.zeros(DEM.shape,  dtype=np.float32)  # Numerator in IDW2 Top Width That is Sampled for each Cell
    MTW = np.zeros(DEM.shape,  dtype=np.float32)   # Max Top Width That is Sampled for each Cell
    WSE_Smoothed = np.zeros(DEM.shape,  dtype=np.float32)  # This is a Gridded Water Surface Elevation that gets Smoothed
    Flooded = np.zeros(DEM.shape,  dtype=np.uint8)   
    # Flooded_NoLevee = np.zeros(DEM.shape) # Old and obsolete????
    ID_Raster = np.zeros(DEM.shape,  dtype=np.int32) # This will help identify the cells flooded by each ID

    # Get the cellsize
    cellsize_X = 0
    cellsize_Y = 0
    lat_f = (yll + yur) / 2
    cellsize_Avg, cellsize_X, cellsize_Y = _Cellsize_Conversion(cellsize, lat_f, cellsize_X, cellsize_Y)

    # Open the VDT database and read in some options
    if Database:
        num_lines, numPoints, MinC, MaxC = _VDT_Get_Num_Min_Max_COMIDs(VDT)
        Q_Database = np.zeros((numPoints+1,), dtype=np.uint8)
        ID_Database = np.zeros((numPoints+1,), dtype=np.uint8)
        
        # The way this is set up, we assume the COMID file has only unique COMIDs... this could be a problem maybe
        _VDT_Fill_ID_Vector_COMID_Flow_File(COMIDFile, ID_Database, MinC, MaxC, Q_Database, AdjustFlow)
        VDT_FILE = "C:\\Users\\lrr43\\Desktop\\Lab\\floodspreader\\src\\floodspreader_lrr43\\temp_VDT.txt"
        _Create_VDT_File_Based_On_Database(VDT_FILE, VDT, ID_Database, MinC, Q_Database, num_lines)
        

    # Open temp VDT
    _checkExistence([MIFN])
    
    #Go through VDT file and create a list of V, D, and T data for each COMID
    num_lines, numPoints, MinC, MaxC = _VDT_Get_Num_Min_Max_COMIDs(MIFN)
    if numPoints < 1:
        raise("There are no points in the VDT file, Not Running this Analysis")
    if not Silent:print(f"Min COMID: {MinC}")
    if not Silent:print(f"Max COMID: {MaxC}")
    ID = np.zeros((numPoints+1,))
    T_Max = 0
    with open(MIFN, 'r') as VDT_FILE:
        line = VDT_FILE.readline()
        numCOMIDs = 0
        for line in VDT_FILE:
            line = line.strip().split(",")
            COMID = int(line[0])
            T = float(line[6])
            if ID[COMID-MinC]==0: # Indicates a new COMID
                numCOMIDs += 1
                ID[COMID-MinC] = numCOMIDs
            if T>T_Max:
                T_Max = T
                
    # Create a 2D Vector for making sure only local cells get flooded.... removed since flood local is no longer an option

    # Allocate memory
    if not Silent:print(f"Allocating memory for {numPoints} points")
    Vlist = np.zeros((numCOMIDs+1,))
    Dlist = np.zeros((numCOMIDs+1,))
    Tlist = np.zeros((numCOMIDs+1,))
    Wlist = np.zeros((numCOMIDs+1,))
    n = np.zeros((numCOMIDs+1,))
    StDev = np.zeros((numCOMIDs+1,))
    
    # Calculate the average V, D, and T for each COMID
    num = 0
    with open(MIFN, 'r') as VDT_FILE:
        line = VDT_FILE.readline()
        for line in VDT_FILE:
            line = line.strip().split(",")
            T = float(line[6])
            if T<=.00000000000001 and (FloodBadCells or SmoothWSE): #Not a legit value, so let's fill it with the average
                if Dlist[int(ID[int(line[0])-MinC])]>0:
                        D_Use = float(Dlist[int(ID[int(line[0])-MinC])]/100)
                else:
                    D_Use = 0.1
                U[int(line[1]),int(line[2])] = DEM[int(line[1]),int(line[2])] + float(line[5])

            COMID = int(line[0])
            row = int(line[1])
            col = int(line[2])
            V = float(line[4])   
            W = float(line[8])

            D_Use = W - DEM[row,col]
            if D_Use<0.00000000000001 or D_Use>9999999:
                D_Use = 0.1
                
            Vlist[int(ID[COMID-MinC])] = int(V*100)
            Dlist[int(ID[COMID-MinC])] = int(D_Use*100)
            Tlist[int(ID[COMID-MinC])] = int(T*100)
            Wlist[int(ID[COMID-MinC])] = int(W*100)
            n[int(ID[COMID-MinC])] += 1

            if SmoothWSE: 
                #U[row,col] = DEM[row,col] + float(line[5])
                U[row,col] = W
            U_TW[row,col] = T

    for i in range(numCOMIDs+1):
        if n[i] > 0:
            Vlist[i] /= n[i]
            Dlist[i] /= n[i]
            Tlist[i] /= n[i]
            Wlist[i] /= n[i]
            
    # Calculate the StDev 
    with open(MIFN, 'r') as VDT_FILE:
        line = VDT_FILE.readline()
        for line in VDT_FILE:
                line = line.strip().split(",")
                C = int(line[0])

                D_Use = float(line[8]) - float(line[7])
                StDev[int(ID[C-MinC])] += int((int(D_Use*100)-Dlist[int(ID[C-MinC])]) ** 2)

    for i in range(numCOMIDs+1):
        if n[i] > 0:
            StDev[i] = int(np.sqrt(StDev[i]/n[i]))


    if not Silent:print("\n\nNow going through Each Stream Cell and Mapping the Flood Event")

    if SmoothWSE:
        WSE_Smoothed = U # Default is the number stays the same
        with open(MIFN, 'r') as VDT_FILE:
            line = VDT_FILE.readline()
            for line in VDT_FILE:
                    line = line.strip().split(",")
                    row = int(line[1])
                    col = int(line[2])
                    WSE_Avg = 0
                    SD = 0
                    ThresholdElev = 999999999999
                    ThresholdTW = 999999999999
                    nn = 0
                    r_start = row - WSEDist
                    r_end = row + WSEDist
                    c_start = col - WSEDist
                    c_end = col + WSEDist
                    
                    if r_start < 0:
                        r_start = 0
                    if c_start < 0:
                        c_start = 0
                    if r_end >= nrows:
                        r_end = nrows - 1
                    if c_end >= ncols:
                        c_end = ncols - 1

                    if WSERmv3:
                        E1=0.2
                        E2=0.1
                        E3=0

                        for rr in range(r_start,r_end+1):
                            for cc in range(c_start,c_end+1):
                                if U_TW[rr,cc] > E3:
                                    if U_TW[rr,cc] >= E1:
                                        E3 = E2
                                        E2 = E1
                                        E1 = U_TW[rr,cc]
                                    elif U_TW[rr,cc]>=E2:
                                        E3 = E2
                                        E2 = U_TW[rr,cc]
                                    else:
                                        E3 = U_TW[rr,cc]
                        if E3 < 1:
                            ThresholdTW = -1000
                        elif E1-E2 >5:
                            ThresholdTW = E2 + 0.001
                        elif E1-E3 >5:
                            ThresholdTW = E3 + 0.001
                        else:
                            ThresholdTW = E1 + 0.001

                    if U_TW[row,col] > ThresholdTW:
                        WSE_Smoothed[row,col] = -1

                    # Calculate average
                    for rr in range(r_start,r_end+1):
                        for cc in range(c_start,c_end+1):
                            if U[rr,cc] > 0 and U[rr,cc] < ThresholdElev and U_TW[rr,cc] < ThresholdTW and row != rr and col!= cc:                  
                                nn += 1
                                WSE_Avg += U[rr,cc]
                    WSE_Avg /= nn

                    # Calculate StDev
                    for rr in range(r_start,r_end+1):
                        for cc in range(c_start,c_end+1):
                            if U[rr,cc] > 0 and U[rr,cc] < ThresholdElev and U_TW[rr,cc] < ThresholdTW and row != rr and col!= cc:
                                SD += (U[rr,cc] - WSE_Avg) ** 2

                    SD = np.sqrt(SD/nn)
                    SD *= WSEStDev

                    if U[row,col] > WSE_Avg + SD:
                        WSE_Smoothed[row,col] = WSE_Avg + SD
                    elif U[row,col] < WSE_Avg - SD:
                        WSE_Smoothed[row,col] = WSE_Avg - SD


        with open(MIFN, 'r') as VDT_FILE:
                line = VDT_FILE.readline()
                for line in VDT_FILE:
                    line = line.strip().split(",")
                    U[int(line[1]),int(line[2])] = 0
        
        # with open(MIFN, 'r') as VDT_FILE:
        #         line = VDT_FILE.readline()
        #         data = np.genfromtxt(VDT_FILE, delimiter=",", usecols=(1, 2))
        #         rows, cols = data.astype(int).T
        #         U[rows, cols] = 0


    # Go Through And Start Flooding
    with open(MIFN, 'r') as VDT_FILE:
        line = VDT_FILE.readline()
        for progress, line in enumerate(VDT_FILE):
            line = line.strip().split(",")
            COMID = int(line[0])
            row = int(line[1])
            col = int(line[2])
            VV = float(line[4])
            DD = float(line[5])
            TT = float(line[6])
            WW = float(line[8])

            DD = WW - DEM[row,col] # This is a better representation of depth if bathymetry is calculated.
            if DD < 0.000000000001:
                DD = 0.1
            idd = int(ID[COMID - MinC])
            WSE = DEM[row,col] + (Dlist[idd]/100)
            TW_Use = Tlist[idd]

            if (TT < 0.000000000001 or DD < 0.0000000000001) and FloodBadCells: # Got a bad value, so use the average values for the COMID
                #WSE = DEM[row,col] + (Dlist[idd]/100)
                WSE = Wlist[idd]/100
            elif UseARDepths: # If FloodSpreader_Use_AR_Depths==1 then you want to use the original AutoRoute Flood Depths at the stream cell
                WSE = WW # Could also say WSE = DEM[row,col] + DD
            elif SmoothWSE:
                WSE = WSE_Smoothed[row,col]
            elif UseARDepthsStDev: # Use the original AutoRoute Flood Depths at the stream cell, but if they are outside of the StDev then we use the average Depth to calculate the Water Surface Elevation
                WSE = DEM[row,col] + DD
                if (WW*100 > Wlist[idd] + StDev[idd]) or (WW*100 < (Wlist[idd] - StDev[idd])):
                    WSE = Wlist[idd]/100
            elif SpecifyDepth:
                WSE = DEM[row,col] + SpecifiedDpth

            if (TT<=0 or DD<=0) and FloodBadCells: # Got a bad value, so use the average values for the COMID
                TW_Use = Tlist[idd]
            elif UseARTW: # If FloodSpreader_Use_AR_TopWidths==1 then you want to use the original AutoRoute Top Widths at the stream cell
                TW_Use = TT
            
            # PUT The TW Limiting Threshold Here.  I think this may be the best place for it.
            if (TW_Use > 0):
                r_start = row - WSEDist
                r_end = row + WSEDist
                c_start = col - WSEDist
                c_end = col + WSEDist

                if r_start < 0:
                    r_start = 0
                if c_start < 0:
                    c_start = 0
                if r_end>=nrows:
                    r_end = nrows-1
                if c_end>= ncols:
                    c_end = ncols-1
                    
                E1=0.2
                E2=0.1
                E3=0

                maxoftemp = np.max(U_TW[r_start:r_end+1,c_start:c_end+1])
                if maxoftemp > E3:
                    if maxoftemp >= E1:
                        E3, E2, E1 = E2, E1, maxoftemp
                    elif maxoftemp >=E2:
                        E3, E2 = E2, maxoftemp
                    else:
                        E3 = maxoftemp
                        
                if E3 < 1:
                    TW_Threshold = -1000
                elif E1-E2 >5:
                    TW_Threshold = E2 + 0.001
                elif E1-E3 >5:
                    TW_Threshold = E3 + 0.001
                else:
                    TW_Threshold = E1 + 0.001

                if TW_Use > TW_Threshold and TW_Threshold > 20: # The "crop-circle" problem typically has a diameter greater than 20 meters.
                    TW_Use = TW_Threshold

            if TW_Use <= 0:
                continue

            if (TT<=0 or DD<=0) and FloodBadCells and JstStrmDpths:
                U[row,col] = DEM[row,col] + (Dlist[idd]/100)
                U_V[row,col] = Vlist[idd]/100
                L[row,col] = 1
            elif JstStrmDpths: # If FloodSpreader_JustStrmDepths==1 then just the Stream Cells are output.
                U[row,col] = WSE
                U_V[row,col] = VV
                L[row,col] = 1
            else: # Find the dims of the scan box
                # The Scan Box is 1.5 (TW_dist_factor) times the Top Width
                if UseARTW: 
                    dist_cells_X = int((TW_Use/100)/cellsize_X) * TWDF + 1
                    dist_cells_Y = int((TW_Use/100)/cellsize_Y) * TWDF + 1
                else:
                    dist_cells_X = int((Tlist[idd]/100)/cellsize_X) * TWDF + 1
                    dist_cells_Y = int((Tlist[idd]/100)/cellsize_Y) * TWDF + 1
                x1 = int(col - dist_cells_X)
                x2 = int(col + dist_cells_X)
                y1 = int(row - dist_cells_Y)
                y2 = int(row + dist_cells_Y)
                if x1 < 0:
                    x1 = 0
                if x2 > ncols -1:
                    x2 = ncols-1
                if y1 < 0:
                    y1 = 0
                if y2 > nrows -1:
                    y2 = nrows-1

                # Here is a place for flood local only, which doesn't seem to be a current option, and so is left out
                # Default is to calculate wieghted wse in scanbox
                cellsize_X_Use = cellsize_X
                cellsize_Y_Use = cellsize_Y
                
                dx = (np.arange(x1,x2+1) - col) * cellsize_X_Use # This is the x distance from the center cell to the cell of interest
                dy = (np.arange(y1, y2+1)[:, np.newaxis] - row) * cellsize_Y_Use # This is the y distance from the center cell to the cell of interest

                # Create an elipse
                is_elipse = (dx**2 / (dist_cells_X * cellsize_X_Use)**2) + (dy**2 / (dist_cells_Y * cellsize_Y_Use)**2)
                mask = is_elipse <= 1
                
                z = dx*dx + dy*dy
                z[np.isclose(z,0)] = 0.001

                weight = 1 / z
                
                u_prev = U[y1:y2+1, x1:x2+1]
                l_prev = L[y1:y2+1, x1:x2+1]
                

                U[y1:y2+1, x1:x2+1][mask] += WSE * weight[mask]
                U_V[y1:y2+1, x1:x2+1][mask] += VV * weight[mask]
                L[y1:y2+1, x1:x2+1][mask] += weight[mask]

                
                MTW[y1:y2+1, x1:x2+1][TW_Use > MTW[y1:y2+1, x1:x2+1]] = TW_Use

                mask = (u_prev <= 0) | ((U[y1:y2+1, x1:x2+1] / L[y1:y2+1, x1:x2+1]) > (u_prev / l_prev))
                ID_Raster[y1:y2+1, x1:x2+1][mask] = COMID
                
            if progress % 1000 == 0:
                _printProgressBar(progress + 1,num_lines, prefix = 'Flooding:', suffix = 'Complete', length = 50)

    #Now go and find Depths and Velocities
    if OutDEP or OutVEL or OutWSE:
        progress = 0
        for r in range(0,nrows):
            for c in range(0,ncols):
                if U[r,c] <=0 or L[r,c] <= 0:
                    U[r,c] = 0
                    U_V[r,c] = 0
                else:
                    depth = U[r,c] / L[r,c]
                    if depth>DEM[r,c]:
                        U[r,c] = depth - DEM[r,c] # U is now the depth of the flood
                        Flooded[r,c] = 1
                    else:
                        U[r,c] = 0

                
            progress += 1
            if progress % 100 == 0:
                _printProgressBar(progress + 1,nrows, prefix = 'Finding Dpths and Vels:', suffix = 'Complete', length = 50)

    depth = np.divide(U,L, out=np.zeros_like(U), where=L!=0)
    if OutWSE or OutDEP:
        U[depth <= DEM] = 0
        U[depth > DEM] = depth[depth > DEM] - DEM[depth > DEM] # U is now the depth of the flood
    Flooded[depth > DEM] = 1
    
    # Now calculate  the weighted average velocity
    if OutVEL:
        U_V = np.divide(U_V,L, out=np.zeros_like(U_V), where=L!=0)

    

    # Delete some memory
    if not Silent:print("\n\n\nDeleteing V, D, and n memory")
    del Dlist
    del n
    del Vlist

    # Here is a section for taking out flooding behind levees, but again this doesn't seem toi be a current option. Leaving as is


    # Write output files
    if not Silent:print("\n\n\nWriting Flood and Depth Files")
    if OutDEP:
        _writeRaster(OutName + "DEP.tif",U,DEMgeoTransform,DEMprojection,ncols,nrows,gdal.GDT_Float32)
    if OutFLD:
        _writeRaster(OutName,Flooded,DEMgeoTransform,DEMprojection,ncols,nrows,gdal.GDT_Byte)
    # There is an option for OutFloodID, which doesn't seem currently supported
    if OutVEL:
        for r in range(0,nrows):
            for c in range(0,ncols):
                if Flooded[r,c]<=0:
                    U_V[r,c] = 0
        _writeRaster(OutName+ "VEL.tif",U_V,DEMgeoTransform,DEMprojection,ncols,nrows,gdal.GDT_Float32)
    if OutWSE:
        U = np.add(U,DEM)
        _writeRaster(OutName+ "WSE.tif",U,DEMgeoTransform,DEMprojection,ncols,nrows,gdal.GDT_Float32)

    # Run Bathymetry Estimation/smoothing
    if not (ARBathy and FSBathy):
        if not Silent:print("\n\nNot enough arguments to Run Bathymetry")
        if not Silent:print("\n   Deleting memory")
        del T
        del ID
        del Flooded
        del U
        del U_V
        del L
        del MTW
        del DEM
        del ID_Raster

        Sim_time = time.time() - Start_Time

        if not Silent:print(f"\n\nSimulation time: {Sim_time:.3f} seconds or {Sim_time/60} minutes")
        return

    if not Silent:print("\n\n\nRunning Bathymetry Analysis, Reading in Bathymetry File from AutoRoute...")

    # Read Bathy data
    _checkExistence([ARBathy])
    IN_BATHY = _readDatasetToArray(ARBathy)
    U = np.zeros((IN_BATHY.shape[0],IN_BATHY.shape[1]))
    L = np.zeros((IN_BATHY.shape[0],IN_BATHY.shape[1]))
    OUT_BATHY = np.zeros((IN_BATHY.shape[0],IN_BATHY.shape[1]))

    # Now go through the files
    TW_dist_factor = 1
    for r in range(0,nrows):
        for c in range(0,ncols):
            if Flooded[r,c] == 1:
                idd = int(ID[COMID-MinC])

                # Find dims of scan box
                dist_cells_X = int((MTW[idd]/100)/cellsize_X) * TW_dist_factor + 1
                dist_cells_Y = int((MTW[idd]/100)/cellsize_Y) * TW_dist_factor + 1
                x1 = int(c - dist_cells_X)
                x2 = int(c + dist_cells_X)
                y1 = int(r - dist_cells_Y)
                y2 = int(r + dist_cells_Y)
                if x1 < 0:
                    x1 = 0
                if x2 > ncols -1:
                    x2 = ncols-1
                if y1 < 0:
                    y1 = 0
                if y2 > nrows -1:
                    y2 = nrows-1

                num_used = 0
                for rr in range(y1,y2+1):
                    for cc in range(x1,x2+1):
                        if cc == c and rr == r:
                            dx = cellsize_X * 0.5
                            dy = cellsize_Y * 0.5
                            is_elipse = 0.1
                        else:
                            dx = (cc-c)*cellsize_X
                            dy = (rr-r)*cellsize_Y
                            DX = dist_cells_X * cellsize_X
                            DY = dist_cells_Y * cellsize_Y
                            is_elipse = (dx*dx/(DX*DX) + dy*dy/(DY*DY))
                        if is_elipse > 1 or IN_BATHY[rr,cc]<=0.1:
                            continue # This coordinate point is not within the elipse or does not have a legitimate Bathymetric point
                        num_used += 1
                        z = np.sqrt(dx*dx + dy*dy)
                        weight = 1/(z*z)
                        U[r,c] = U[r,c] + (weight*IN_BATHY[rr,cc])
                        L[r,c] = L[r,c] + weight
                
                if num_used == 0:
                    L[r,c] = 1
                
                OUT_BATHY[r,c] = U[r,c]/L[r,c]
            
            if OUT_BATHY[r,c] < 0.01 or OUT_BATHY[r,c] > DEM[r,c]:
                OUT_BATHY[r,c] = DEM[r,c]

    # Write the output bathy
    _writeRaster(FSBathy,OUT_BATHY,DEMgeoTransform,DEMprojection,ncols,nrows,gdal.GDT_Float32)

    # Finish
    if not Silent:print("\n\nNot enough arguments to Run Bathymetry")
    Sim_time = time.time() - Start_Time
    if not Silent:print(f"\n\nSimulation time: {Sim_time:.3f} seconds or {Sim_time/60} minutes")
    return
                    
def  _VDT_Get_Num_Min_Max_COMIDs(VDT: str):
    # Gets the number of lines in the file and the number of points between the min and max COMIDs
    with open(VDT, 'r') as f:
        f.readline()
        line = f.readline().strip()
        num_cols = len(line.split(','))
    df = pd.read_csv(VDT, header=None, names=[f'col_{i}' for i in range(num_cols)], skiprows=1)
    MinC = int(df.iloc[:,0].min())
    MaxC = int(df.iloc[:,0].max())
    num_lines = len(df.index)

    if num_lines == 0:
        return 
    numPoints = MaxC - MinC + 1
    return num_lines, numPoints, MinC, MaxC

def _writeRaster(Outname,Output_Raster,GeoTransform,projection,ncols,nrows,datatype):
    hDriver = gdal.GetDriverByName("GTiff")
    DEP_OUT = hDriver.Create(Outname, xsize=ncols, ysize=nrows, bands=1, eType=datatype)
    DEP_OUT.SetGeoTransform(GeoTransform)
    DEP_OUT.SetProjection(projection)
    DEP_OUT.GetRasterBand(1).WriteArray(Output_Raster)
    DEP_OUT = None

def _printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()
        
def _Cellsize_Conversion(cellsize: float, lat_f: float, cellsize_X: float, cellsize_Y: float):

    """
    These are based on lat-long conversions from www.zodiacal.com/tools/lat_table.php
    """
    lat = np.abs(lat_f)
    if cellsize > 0.5: # This indicates that the DEM is projected, so no need to convert from geographic into projected.
        cellsize_X = cellsize
        cellsize_Y =  cellsize
    if lat < 0.00000000000000000001:
        raise ValueError(f'Please use lat and long values greater than or equal to 0. (Got {lat})  Thanks!')
    if( lat>=0 and lat<=10 ): 
        lat_up=110.61
        lat_down=110.57
        lon_up=109.64
        lon_down=111.32
        lat_base=0.0
    elif( lat>10 and lat<=20 ):
        lat_up=110.7 
        lat_down=110.61
        lon_up=104.64 
        lon_down=109.64
        lat_base=10.0
    elif( lat>20 and lat<=30 ):
        lat_up=110.85 
        lat_down=110.7 
        lon_up=96.49 
        lon_down=104.65
        lat_base=20.0
    elif( lat>30 and lat<=40 ):
        lat_up=111.03 
        lat_down=110.85 
        lon_up=85.39 
        lon_down=96.49
        lat_base=30.0
    elif( lat>40 and lat<=50 ): 
        lat_up=111.23 
        lat_down=111.03 
        lon_up=71.70 
        lon_down=85.39 
        lat_base=40.0
    elif( lat>50 and lat<=60 ): 
        lat_up=111.41 
        lat_down=111.23 
        lon_up=55.80 
        lon_down=71.70 
        lat_base=50.0
    elif( lat>60 and lat<=70 ): 
        lat_up=111.56 
        lat_down=111.41 
        lon_up=38.19 
        lon_down=55.80 
        lat_base=60.0
    elif( lat>70 and lat<=80 ): 
        lat_up=111.66 
        lat_down=111.56 
        lon_up=19.39 
        lon_down=38.19 
        lat_base=70.0
    elif( lat>80 and lat<=90 ): 
        lat_up=111.69 
        lat_down=111.66 
        lon_up=0.0 
        lon_down=19.39 
        lat_base=80.0
    else:
        raise ValueError(f'Please use legit (0-90) lat and long values. (Got {lat})  Thanks!')

    # Latitude conversion
    lat_conv = lat_down + (lat_up - lat_down) * (lat-lat_base)/10
    cellsize_Y = cellsize * lat_conv * 1000 # Converts from deg to m

    # Longitude conversion
    lon_conv = lon_down + (lon_up - lon_down) * (lat-lat_base)/10
    cellsize_X = cellsize * lon_conv * 1000 # Converts from deg to m

    if lat_conv < lat_down or lat_conv > lat_up or lon_conv < lon_up or lon_conv > lon_down:
        raise("Problem in cellsize conversion.")

    return (1000*(lat_conv+lon_conv)/2) * cellsize, cellsize_X, cellsize_Y

def _VDT_Fill_ID_Vector_COMID_Flow_File(COMID_FLOW_FILE: str, ID: np.ndarray, MinC: int, MaxC: int, Q_database: np.ndarray, FS_Adjust_Flow_By_Fraction: float) -> int:
    df = pd.read_csv(COMID_FLOW_FILE)
    num = 0
    for row in df.itertuples(index=False):
        C = int(row[0])
        Q = float(row[1])
        num += 1
        ID[C-MinC] = num
        Q_database[C-MinC] = int(100 * Q * FS_Adjust_Flow_By_Fraction)
    #print(np.nonzero(ID)[0])
    print(Q_database[Q_database != 0])
    return num

def _Create_VDT_File_Based_On_Database(VDT_FILE: str, VDT_DATABASE_FILE: str, ID_Database: np.ndarray, MinC: int, Q_Database: np.ndarray, numPoints: int)  -> int:
    # Q_Prev = 999999999999
    # num_ordinates = 0
    # with open(VDT_DATABASE_FILE, 'r') as VDT:
    #     line = VDT.readline()
    #     line = VDT.readline().split(",")[5:]
    #     n = 0
    #     while (n*4)<len(line):
    #         if float(line[n*4]) > Q_Prev:
    #             break
    #         Q_Prev = float(line[n*4])
    #         num_ordinates += 1
    #         n += 1

    # dtypes = {'COMID': int, 'Row': int, 'Col': int}
    # for i in range(num_ordinates*4):
    #     dtypes[f'Ord{i+1}'] = float
    # cols = ['COMID', 'Row', 'Col', 'Elev', 'QBaseflow'] + [f'Ord{i+1}' for i in range(num_ordinates*4)]
    # df = pd.read_csv(VDT_DATABASE_FILE, skiprows=1, header=None, names=cols, dtype=dtypes)

    # V_Prev, W_Prev, T_Prev = 0,0,0
    # with open(VDT_FILE, 'w') as f:
    #     f.write('COMID,Row,Col,Q,V,D,T,Elev,WSE')
    #     for n in range(numPoints):
    #         C = df.iloc[n,0]
    #         Q_Want = Q_Database[C-MinC]/100
    #         Q, V, T, W = 0,0,0,0

    #         for ords in range(num_ordinates):
    #             Q,V,T,W = df.iloc[n,5+ords*4], df.iloc[n,6+ords*4], df.iloc[n,7+ords*4], df.iloc[n,8+ords*4]
    #             if Q<Q_Want:
    #                 if ords>0 and V_Prev>0 and W_Prev>1 and T_Prev>1 and V>0 and W>1 and T>0:
    #                     V = V + (V_Prev-V)*(Q_Want-Q) / (Q_Prev-Q)
    #                     T = T + (T_Prev-T)*(Q_Want-Q) / (Q_Prev-Q)
    #                     W = W + (W_Prev-W)*(Q_Want-Q) / (Q_Prev-Q)
    #                 if ords < num_ordinates-1:
    #                     break
    #             if Q>0 and V>0 and T>0 and W>0:
    #                 Q_Prev=Q
    #                 V_Prev=V
    #                 T_Prev=T
    #                 W_Prev=W

    #         if Q>0 and V>0 and T>0 and W>0:
    #             f.write(f"\n{C},{df.iloc[n,1]},{df.iloc[n,2]},{Q_Want},{V},{W-df.iloc[n,3]},{T},{df.iloc[n,3]},{W}")
    #         else:   
    #             f.write(f"\n{C},{df.iloc[n,1]},{df.iloc[n,2]},{Q_Want},{V},{0.0},{T},{df.iloc[n,3]},{W}")
    #         if n % 10000 == 0:
    #             _printProgressBar(n,numPoints)


        
    n, r, c, C = 0, 0, 0, 0
    ords, num_ordinates = 0, 0
    Q, QBaseflow, V, D, T, E, W = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    Q_Prev, V_Prev, T_Prev, W_Prev, Q_Want = 999999999999.9, 0.0, 0.0, 0.0, 0.0

    Q_Database = np.array(Q_Database) / 100.0

    with open(VDT_DATABASE_FILE, 'r') as f:
        line = f.readline()
        line = f.readline().split(",")[5:]
        n = 0
        while (n*4)<len(line):
            if float(line[n*4]) > Q_Prev:
                break
            Q_Prev = float(line[n*4])
            num_ordinates += 1
            n += 1

        f.seek(0)
        with open(VDT_FILE, 'w') as f_out:
            f.readline()
            f_out.write("COMID,Row,Col,Q,V,D,T,Elev,WSE")
    

            for n in range(numPoints):
                array = np.genfromtxt(f.readline().split(','))
                C, r, c, E = int(array[0]), int(array[1]), int(array[2]), float(array[3])
                Q_Want = Q_Database[C-MinC]

                Q, V, T, W = 0.0, 0.0, 0.0, 0.0
                for ords in range(num_ordinates):
                    Q, V, T, W = float(array[5+ords*4]), float(array[6+ords*4]), float(array[7+ords*4]), float(array[8+ords*4])
                    if(Q < Q_Want):
                        if(ords > 0 and V_Prev > 0.0 and W_Prev > 1.0 and T_Prev > 1.0 and V > 0.0 and W > 1.0 and T > 0.0): # Just keep the existing values
                            V = V + (V_Prev-V) * (Q_Want-Q) / (Q_Prev-Q)
                            T = T + (T_Prev-T) * (Q_Want-Q) / (Q_Prev-Q)
                            W = W + (W_Prev-W) * (Q_Want-Q) / (Q_Prev-Q)
                        if(ords < (num_ordinates-1)):
                            continue
                        break

                    if(Q > 0.0 and V > 0.0 and T > 0.0 and W > 0.0):
                        Q_Prev, V_Prev, T_Prev, W_Prev = Q, V, T, W

                if(Q > 0.0 and V > 0.0 and T > 0.0 and W > 0.0):
                    f_out.write(f"\n{C},{r},{c},{Q_Want},{V},{W-E},{T},{E},{W}")
                else:
                    f_out.write(f"\n{C},{r},{c},{Q_Want},{V},0.0,{T},{E},{W}")
                if n % 10000 == 0:
                    _printProgressBar(n,numPoints)




    ##Takes in a path to a folder of folders containing DEMs. By default we match the FABDEM naming convention for the pattern but others can be used. Extent is giving in the format of (minx, miny, maxx, maxy)

def StreamLine_Parser(dem: str, root_folder: str, extents: str, output_strm: str, field: str = 'TDXHydroLi'):
    # Load the DEM extent
    minx, miny, maxx, maxy, _, _, ncols, nrows, geoTransform, _ = _Get_Raster_Details(dem)

    extents_df = pd.read_parquet(extents)
    filtered_gdf = extents_df[
        (minx <= extents_df.maxx) &
        (maxx >= extents_df.minx) &
        (miny <= extents_df.maxy) &
        (maxy >= extents_df.miny)
    ]

    vpus = filtered_gdf.VPUCode.unique()
    
    for folder in os.listdir(root_folder):
            folder_path = os.path.join(root_folder, folder)
            if not os.path.isdir(folder_path):
                continue

            for file in os.listdir(folder_path):
                if file.endswith('.parquet') and int(file[4:7]) in vpus:
                    vpu_df = gpd.read_parquet(os.path.join(folder_path, file))
                    filtered_gdf = filtered_gdf.merge(vpu_df[['TDXHydroLinkNo', 'geometry']], on='TDXHydroLinkNo', how='left')

    geometry_columns = [col for col in filtered_gdf.columns if col.startswith('geometry_')]

    # # Create a new 'geometry' column by applying a lambda function
    def merge_geometries(row):
        geometries = row.dropna().values
        if len(geometries) > 1:
            return MultiLineString(geometries)
        else:
            return geometries[0]
        
    filtered_gdf['geometry'] = filtered_gdf[geometry_columns].apply(merge_geometries, axis=1)

    # # Drop the individual geometry columns
    gpd.GeoDataFrame(filtered_gdf.drop(columns=geometry_columns)[['TDXHydroLinkNo', 'geometry']], crs='EPSG:4326').to_file('temp.shp')

    # Open the data source and read in the extent
    source_ds = ogr.Open('temp.shp')
    if source_ds is None:
        raise Exception("Failed to open the source data source.")
 
    source_layer = source_ds.GetLayer()

    # Create the destination data source
    target_ds = gdal.GetDriverByName('GTiff').Create(output_strm, ncols, nrows, 1, gdal.GDT_UInt32)
    target_ds.SetGeoTransform(geoTransform)
    band = target_ds.GetRasterBand(1)
    band.SetNoDataValue(0)

    # Rasterize
    gdal.RasterizeLayer(target_ds, [1], source_layer, options=["ATTRIBUTE=" + field])
    os.remove('temp.shp')

 