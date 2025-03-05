#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:50:23 2023
@author: ben
"""

import geopandas as gpd
import shapely as shp
import rioxarray as rio
import xarray as xr
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import Majadas_utils
# from pysheds.grid import Grid


AOI = 'Buffer_5000' #H2_Bassin
# AOI = 'Buffer_100'
# AOI = 'H2_Bassin'

#%% Define path and crs projection 
prepoEOPath = Path('/run/media/z0272571a/SENET/iberia_daily/E030N006T6')
rootDataPath = Path('/run/media/z0272571a/LVM_16Tb/Ben/EOMAJI/MAJADAS/')
figPath = Path('../figures/')
file_pattern = '*ET_0-gf*.tif'
folder_weather_path = rootDataPath/'E030N006T6'
file_pattern = '*ET_0*.tif'
ET_0_filelist = list(prepoEOPath.glob(file_pattern))
def extract_filedate(file_path):
    file_name = file_path.name
    date_str = file_name.split('_')[0]
    return datetime.strptime(date_str, '%Y%m%d')
crs_ET_0 = rio.open_rasterio(ET_0_filelist[0]).rio.crs

#%% Read AOI points and plots
# -----------------------------------------------------------------------------

majadas_POIs = gpd.read_file('../data/AOI/POI_Majadas.geojson')
majadas_POIs.to_crs(crs_ET_0, inplace=True)
majadas_aoi = Majadas_utils.get_Majadas_aoi(buffer=5000)
majadas_aoi.crs

# DTM_tile = rio.open_rasterio('../data/Spain/DEM_Spain_tiles/E030N012T6.tif')
DTM_tile = rio.open_rasterio('../data/Spain/DEM_Spain_tiles/E030N006T6.tif')
DTM_tile.rio.crs 

clipped_DTM_rxr = DTM_tile.rio.clip_box(
                                       minx=majadas_aoi.bounds['minx'],
                                       miny=majadas_aoi.bounds['miny'],
                                       maxx=majadas_aoi.bounds['maxx'],
                                       maxy=majadas_aoi.bounds['maxy'],
                                      crs=majadas_aoi.crs,
                                    )  
clipped_DTM_rxr_reproj = clipped_DTM_rxr.rio.reproject(crs_ET_0)
clipped_DTM_rxr_reproj.isel(band=0).plot.imshow()
clipped_DTM_rxr_reproj.rio.resolution()
clipped_DTM_rxr_reproj.rio.to_raster(f'../data/Spain/clipped_DTM_Majadas_AOI_{AOI}.tif', 
                              # compress='LZMA', 
                              # tiled=True, 
                              # dtype="int32"
                              )

#%%

majadas_aoi = gpd.read_file('../data/Spain/GIS_catchment_majadas/BassinH2_Majadas_corrected.shp')
majadas_aoi.crs
majadas_aoi.to_crs(crs_ET_0, inplace=True)

clipped_DTM_H2_rxr = DTM_tile.rio.clip_box(
                                       minx=majadas_aoi.bounds['minx'],
                                       miny=majadas_aoi.bounds['miny'],
                                       maxx=majadas_aoi.bounds['maxx'],
                                       maxy=majadas_aoi.bounds['maxy'],
                                      crs=majadas_aoi.crs,
                                    )  
clipped_DTM_H2_rxr_reproj = clipped_DTM_H2_rxr.rio.reproject(crs_ET_0)
clipped_DTM_H2_rxr_reproj.isel(band=0).plot.imshow()
clipped_DTM_H2_rxr_reproj.rio.resolution()
clipped_DTM_H2_rxr_reproj.rio.to_raster('../data/Spain/clipped_DTM_Majadas_AOI_H2.tif', 
                              # compress='LZMA', 
                              # tiled=True, 
                              # dtype="int32"
                              )


#%% Read Majadas DTM
# -----------------------------------------------------------------------------
# DTM_Global = rio.open_rasterio('../data/Spain/DTM_Global/dem.vrt')
# DTM_Global.rio.crs 
# # DTM_Global.rio.resolution() 

# # clipped_DTM_rxr = DTM_Global.rio.clip_box(
# #                                       minx=majadas_aoi_buff.bounds['minx'],
# #                                       miny=majadas_aoi_buff.bounds['miny'],
# #                                       maxx=majadas_aoi_buff.bounds['maxx'],
# #                                       maxy=majadas_aoi_buff.bounds['maxy'],
# #                                        crs=majadas_aoi_buff.crs,
# #                                     ) 
# # clipped_DTM_rxr.isel(band=0).plot.imshow()
# # clipped_DTM_rxr_reproj = clipped_DTM_rxr.rio.reproject(crs_ET_0)


# clipped_DTM_rxr = DTM_Global.rio.clip_box(
#                                        minx=majadas_aoi.bounds['minx'],
#                                        miny=majadas_aoi.bounds['miny'],
#                                        maxx=majadas_aoi.bounds['maxx'],
#                                        maxy=majadas_aoi.bounds['maxy'],
#                                       crs=majadas_aoi.crs,
#                                     )  
# clipped_DTM_rxr_reproj = clipped_DTM_rxr.rio.reproject(crs_ET_0)
# clipped_DTM_rxr_reproj.isel(band=0).plot.imshow()
# clipped_DTM_rxr_reproj.rio.resolution()

# # dd
# # clipped_DTM_rxr = DTM_Global.rio.clip(
# #                                       majadas_aoi_crs.geometry.values,  
# #                                       # minx=majadas_aoi.bounds['minx'],
# #                                       # miny=majadas_aoi.bounds['miny'],
# #                                       # maxx=majadas_aoi.bounds['maxx'],
# #                                       # maxy=majadas_aoi.bounds['maxy'],
# #                                       crs=majadas_aoi.crs,
# #                                     )  
 


# clipped_DTM_rxr_reproj.isel(band=0).plot.imshow()
# clipped_DTM_rxr_reproj.rio.to_raster('../data/Spain/DTM_Global/clipped_DTM_Majadas_AOI.tif', 
#                               # compress='LZMA', 
#                               # tiled=True, 
#                               # dtype="int32"
#                               )
#%% Look for catchment delineation
# -----------------------------------------------------------------------------
# grid = Grid.from_raster('../data/Spain/DTM_Global/clipped_DTM_Majadas_AOI.tif')
# dem = grid.read_raster('../data/Spain/DTM_Global/clipped_DTM_Majadas_AOI.tif')

# # Condition DEM
# # Fill pits in DEM
# pit_filled_dem = grid.fill_pits(dem)

# # Fill depressions in DEM
# flooded_dem = grid.fill_depressions(pit_filled_dem)
    
# # Resolve flats in DEM
# inflated_dem = grid.resolve_flats(flooded_dem)

# # Determine D8 flow directions from DEM
# # ----------------------
# # Specify directional mapping
# # Resolve flats and compute flow directions
# fdir = grid.flowdir(inflated_dem)    

# grid.viewfinder = fdir.viewfinder
# # Compute accumulation
# acc = grid.accumulation(fdir)
# x, y = -7, 39

# # Snap pour point to high accumulation cell
# x_snap, y_snap = grid.snap_to_mask(acc > 1000, (x, y))
# # Delineate the catchment
# catch = grid.catchment(x=x_snap, y=y_snap, fdir=fdir, xytype='coordinate')

# # Plot the result
# grid.clip_to(catch)
# catch_view = grid.view(catch)
# fig, ax = plt.subplots(figsize=(8,6))
# fig.patch.set_alpha(0)

# plt.grid('on', zorder=0)
# im = ax.imshow(np.where(catch_view, catch_view, np.nan), extent=grid.extent,
#                zorder=1, cmap='Greys_r')
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.title('Delineated Catchment', size=14)
# grid.to_raster(fdir, '../data/Spain/DTM_Global/test_dir.tif', 
#                apply_output_mask=True,
#                 blockxsize=16, blockysize=16
#                 )
# catch_view = grid.view(catch, dtype=np.uint8)
# shapes = grid.polygonize(catch_view)
# schema = {
#         'geometry': 'Polygon',
#         'properties': {'LABEL': 'float:16'}
# }
# import fiona
# with fiona.open('../data/Spain/DTM_Global/catchment.shp', 'w',
#                 driver='ESRI Shapefile',
#                 crs=grid.crs.srs,
#                 schema=schema) as c:
#     i = 0
#     for shape, value in shapes:
#         rec = {}
#         rec['geometry'] = shape
#         rec['properties'] = {'LABEL' : str(value)}
#         rec['id'] = str(i)
#         c.write(rec)
#         i += 1
        
#%% Save DEM
# -----------------------------------------------------------------------------
# dem_rio = rio.open_rasterio('../data/Spain/clipped_DTM_Majadas_AOI.tif')
# del_catchement = gpd.read_file('../data/Spain/DTM_Global/catchment.shp')
# dem_rio_cliped = dem_rio.rio.clip(
#                                   del_catchement.geometry.values,
#                                   crs=del_catchement.crs,
#                                 )  
# dem_rio_cliped.isel(band=0).plot.imshow()
# dem_rio_cliped.rio.to_raster('../data/Spain/DTM_Global/clipped_DTM_Majadas_Bassin.tif', 
#                               # compress='LZMA', 
#                               # tiled=True, 
#                               # dtype="int32"
#                               )

#%% plot majadas DTM 
# -----------------------------------------------------------------------------
# fig, axs = plt.subplots(1,2)
# clipped_DTM_rxr_reproj.isel(band=0).plot.imshow(vmin=0,ax=axs[0])
# majadas_aoi.plot(ax=axs[1],edgecolor='black', facecolor='none')
# fig.savefig(figPath/'DTM_Majadas.png', dpi=300)
