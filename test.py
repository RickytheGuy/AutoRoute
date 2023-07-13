from src.floodspreader_lrr43.main import *
import os
import time
starttime = time.time()
exe = "C:\\Users\\lrr43\\Desktop\\Lab\\ARFS\\exes\\AutoRoute_w_GDAL.exe"
fexe = "C:\\Users\\lrr43\\Desktop\\Lab\\ARFS\\exes\\AutoRoute_FloodSpreader.exe"

dem = "C:\\Users\\lrr43\\Desktop\\Lab\\ARFS\\HobbleCreek\\DEM\\N40W112_FABDEM_V1-0.tif"
strm = "C:\\Users\\lrr43\\Desktop\\Lab\\ARFS\\HobbleCreek\STRM\\strm_fabdem.tif"
flowfile = "C:\\Users\\lrr43\\Desktop\\Lab\\ARFS\\HobbleCreek\\flows\\flows.csv"
RAPIDflow = "C:\\Users\\lrr43\\Desktop\\Lab\\ARFS\\HobbleCreek\\flows\\RAPIDflow_fabdem.txt"
maxflowfile = "C:\\Users\\lrr43\\Desktop\\Lab\\ARFS\\Kailey-DR\\flows\\New_100_Flows.csv"
land = "C:\\Users\\lrr43\\Desktop\Lab\\ARFS\\HobbleCreek\\LAND\\W120N60__EPSG-4326.tif"
LU = "C:\\Users\\lrr43\\Desktop\\Lab\\ARFS\\HobbleCreek\\LAND\\fadbem_land.tif"
mannings = "C:\\Users\\lrr43\\Desktop\\Lab\\ARFS\\Kailey-DR\\LAND\\manning.txt"

outvdt = "C:\\Users\\lrr43\\Desktop\\Lab\\ARFS\\HobbleCreek\\vdt\\hc_fabdem.txt"
metafile = "testVDTMETA.txt"
outforDR = "C:\\Users\\lrr43\\Desktop\\Lab\\ARFS\\Kailey-DRlast\\FLOODMAPS\\DR_"


mannings_dict = {0:0,111:45,113:47,112:46,114:48,115:49,116:50,121:60,123:62,122:61,124:63,125:64,126:65,20:5,30:10,90:10,100:10,60:25,40:15,50:20,70:30,80:35,200:100,255:0}


#PrepareLU(dem,land,LU, mannings_dict)
# "C:\\Users\\lrr43\\Desktop\\Lab\\DeSoto\\LandCover\\W100N40_PROBAV_LC100_global_v3.0.1_2019-nrt_Discrete-Classification-map_EPSG-4326.tif","testLAND.tif",normalize=True)

# print(test)
shpname = "C:\\Users\\lrr43\\Desktop\\Lab\\ARFS\\HobbleCreek\\shapefile\\hobble_creek.gpkg"
ogshpname = "C:\\Users\\lrr43\\Desktop\\Lab\\GEOGLOWSData\\ShapeFiles\\DR\\OG_DR\\DR_OG.shp"
MIFN = "C:\\Users\\lrr43\\Desktop\\Lab\\DRTestCase\\MyCase\\VDT\\DR_TDX_Temp_VDT_File.txt"

#PrepareStream(dem,shpname,strm, field='reach_id')
#MakeRAPIDFile(flowfile,strm,RAPIDflow, flows=['100_year', 'Base'], field='reach_id')
#FloodSpreader(fexe,dem,outvdt,'test_julia.tif', COMIDFile=flowfile, OutFLD=True, DONTRUN=True)
AutoRoute(exe,dem,strm, RAPIDflow, LU, outvdt, mannings, '100_year Base', Silent=False, 
          RAPIDid='reach_id',LowSpotRange=20, Bathy=True, RAPIDbaseflow = 'Base', X_distance=2000)
# direct = "C:\\Users\\lrr43\\Desktop\\Lab\\ARFS\\Kailey-DRlast\\flows"
# for files in os.listdir(direct):
#     if 'Max' in files:
#         FloodSpreader(fexe,dem,outvdt,outforDR+os.path.basename(files)+'.tif', COMIDFile=os.path.join(direct,files), OutFLD=True, 
#             AdjstFldDpthMthd=1,Silent=False, TWDF=1.7)
#FloodSpreaderPy(dem, outvdt, 'test.tif', COMIDFile=maxflowfile, OutFLD=True,MIFN=MIFN,SmoothWSE=False,Silent=False,TWDF=1.7,FloodBadCells=True)
print(round(time.time()-starttime, 2), ' secs')

