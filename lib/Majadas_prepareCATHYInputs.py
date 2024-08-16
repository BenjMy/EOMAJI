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
import utils

#%% Define path and crs projection 

# rootDataPath = Path('/run/media/z0272571a/LVM_16Tb/Ben/EOMAJI/MAJADAS/')
prepoEOPath = Path('/run/media/z0272571a/SENET/iberia_daily/E030N006T6')
rootDataPath = Path('/run/media/z0272571a/LVM_16Tb/Ben/EOMAJI/MAJADAS/')
figPath = Path('../figures/')
file_pattern = '*ET_0-gf*.tif'
folder_weather_path = rootDataPath/'E030N006T6'


file_pattern = '*ET_0*.tif'
ET_0_filelist = list(prepoEOPath.glob(file_pattern))

file_pattern = '*TPday*.tif'
rain_filelist = list(prepoEOPath.glob(file_pattern))




crs_ET_0 = rio.open_rasterio(ET_0_filelist[0]).rio.crs

#%% Read AOI points and plots

majadas_aoi = Majadas_utils.get_Majadas_aoi()

majadas_POIs, POIs_coords = Majadas_utils.get_Majadas_POIs()


# majadas_aoi = gpd.read_file('../data/AOI/POI_Majadas.kmz')


#%% Read Majadas DTM
# DTM EPSG:25830 - ETRS89 / UTM zone 30N - Projected
from rioxarray.merge import merge_arrays

DTMfiles = [
            '../data/DTM/MDT02-ETRS89-HU30-0624-1-COB2_reproj.tif',
            '../data/DTM/MDT02-ETRS89-HU30-0624-3-COB2_reproj.tif',
            ]
DTM_rxr = []
for dtm in DTMfiles:
    DTM_rxr.append(rio.open_rasterio(dtm))
DTM_rxr = merge_arrays(DTM_rxr)
# DTM_rxr.rio.crs= crs_ET_0
DTM_rxr.rio.write_crs(crs_ET_0, inplace=True)


clipped_DTM_rxr = DTM_rxr.rio.clip_box(
                                      minx=majadas_aoi.bounds['minx'],
                                      miny=majadas_aoi.bounds['miny'],
                                      maxx=majadas_aoi.bounds['maxx'],
                                      maxy=majadas_aoi.bounds['maxy'],
                                      crs=majadas_aoi.crs,
                                    )  

#%% plot majadas DTM 
fig, axs = plt.subplots(1,2)
DTM_rxr.isel(band=0).plot.imshow(vmin=0,ax=axs[0])
majadas_POIs.plot(ax=axs[0])
majadas_aoi.plot(ax=axs[0],edgecolor='black', facecolor='none')

clipped_DTM_rxr.isel(band=0).plot.imshow(vmin=250,vmax=280,ax=axs[1])
majadas_POIs.plot(ax=axs[1])
majadas_aoi.plot(ax=axs[1],edgecolor='black', facecolor='none')

fig.savefig(figPath/'DTM_Majadas.png', dpi=300)

#%% Read and CLIP ! Majadas grids: S3/Meteo = E030N006T6 and S2/Landast = X0033_Y0044
# S3/Meteo EPGS CRS EPSG:27704 - WGS 84 / Equi7 Europe - Projected

# for m in ET_0_filelist:
#     etrefi = rio.open_rasterio(m)
#     clipped_etrefi = etrefi.rio.clip_box(
#                                           minx=majadas_aoi.bounds['minx'],
#                                           miny=majadas_aoi.bounds['miny'],
#                                           maxx=majadas_aoi.bounds['maxx'],
#                                           maxy=majadas_aoi.bounds['maxy'],
#                                         crs=majadas_aoi.crs,
#                                         )   
#     clipped_etrefi['time']=utils.extract_filedate(m)
#     clipped_etrefi.rio.to_raster('../prepro/Majadas/' + m.name)
    
# for m in rain_filelist:
#     etrefi = rio.open_rasterio(m)
#     clipped_etrefi = etrefi.rio.clip_box(
#                                           minx=majadas_aoi.bounds['minx'],
#                                           miny=majadas_aoi.bounds['miny'],
#                                           maxx=majadas_aoi.bounds['maxx'],
#                                           maxy=majadas_aoi.bounds['maxy'],
#                                         crs=majadas_aoi.crs,
#                                         )   
#     clipped_etrefi['time']=utils.extract_filedate(m)
#     clipped_etrefi.rio.to_raster('../prepro/Majadas/' + m.name)
    

#%% Read CLIPPED and save as NETCDF
 # ! Majadas grids: S3/Meteo = E030N006T6 and S2/Landast = X0033_Y0044
# S3/Meteo EPGS CRS EPSG:27704 - WGS 84 / Equi7 Europe - Projected


# file_pattern = '*ET_0-gf*.tif'
# ET_0_clipped_filelist = list(Path('../prepro/Majadas').glob(file_pattern))

# file_pattern = '*TPday*.tif'
# rain_clipped_filelist = list(Path('../prepro/Majadas').glob(file_pattern))


# ETp_l = []
# ETp_dates = []
# for m in ET_0_clipped_filelist:
#     ETpfi = rio.open_rasterio(m)
#     ETpfi['time']=utils.extract_filedate(m)
#     ETp_l.append(ETpfi)
#     ETp_dates.append(ETpfi['time'])

# rain = []
# rain_dates = []
# for m in rain_clipped_filelist:
#     rainfi = rio.open_rasterio(m)
#     rainfi['time']=utils.extract_filedate(m)
#     rain.append(rainfi)
#     rain_dates.append(rainfi['time'])

# ETp = xr.concat(ETp_l,dim='time')
# ETp.to_netcdf('../prepro/ETp_Majadas.netcdf')
# RAIN = xr.concat(rain,dim='time')
# RAIN.to_netcdf('../prepro/RAIN_Majadas.netcdf')


#%%
ETp_ds = xr.open_dataset('../prepro/ETp_Majadas.netcdf')
ETp_ds = ETp_ds.rename({"__xarray_dataarray_variable__": "ETp"})
RAIN_ds = xr.open_dataset('../prepro/RAIN_Majadas.netcdf')
RAIN_ds = RAIN_ds.rename({"__xarray_dataarray_variable__": "RAIN"})


ETp = ETp_ds.to_dataarray().isel(variable=0,band=0).sortby('time')
RAIN = RAIN_ds.to_dataarray().isel(variable=0,band=0).sortby('time')

# Issue with rain!

print('Errrrrorrr in rain evaluation in the input!')
# data_array = data_array.where((data_array <= 300) & (data_array > 0), other=np.nan)
RAIN = RAIN.where((RAIN <= 300) & (RAIN > 0), other=0)

#%% Check variations on POI

labels_POIs = ['Lake','Intensive Irrigation','Tree-Grass']
fig, ax = plt.subplots()
for i, ppc in enumerate(POIs_coords):
    ETp_poi = ETp.sel(x=ppc[0],y=ppc[1], method="nearest")
    ax.plot(ETp_poi.time.values,ETp_poi,label=labels_POIs[i])
ax.legend()
ax.set_xlabel('Datetime')
ax.set_ylabel('ETp (mm/day)')
fig.savefig(figPath/'ETp_serie_Majadas.png', dpi=300)

fig, ax = plt.subplots()
for ppc in POIs_coords:
    RAIN_poi = RAIN.sel(x=ppc[0],y=ppc[1], method="nearest")
    ax.scatter(RAIN_poi.time,RAIN_poi.values)
ax.legend()
ax.set_xlabel('Datetime')
ax.set_ylabel('Rain (mm)')

fig.savefig(figPath/'rain_serie_Majadas.png', dpi=300)

#%%
fig, ax = plt.subplots()
DTM_rxr.isel(band=0).plot.imshow(vmin=0)

#%% Read Majadas grids: S3/Meteo = E030N006T6 and S2/Landast = X0033_Y0044
# S3/Meteo EPGS CRS EPSG:27704 - WGS 84 / Equi7 Europe - Projected

# dates = []
# ETref = []
for m in ET_0_filelist:
    etrefi = rio.open_rasterio(m)
    clipped_etrefi = etrefi.rio.clip_box(
                                         minx=majadas_aoi.bounds['minx'],
                                         miny=majadas_aoi.bounds['miny'],
                                         maxx=majadas_aoi.bounds['maxx'],
                                         maxy=majadas_aoi.bounds['maxy'],
                                        crs=majadas_aoi.crs,
                                        )   
    clipped_etrefi['time']=utils.extract_filedate(m)
    clipped_etrefi.rio.to_raster('../prepro/Majadas/' + m.name + '.tif')
    
for m in rain_filelist:
    etrefi = rio.open_rasterio(m)
    clipped_etrefi = etrefi.rio.clip_box(
                                         minx=majadas_aoi.bounds['minx'],
                                         miny=majadas_aoi.bounds['miny'],
                                         maxx=majadas_aoi.bounds['maxx'],
                                         maxy=majadas_aoi.bounds['maxy'],
                                        crs=majadas_aoi.crs,
                                        )   
    clipped_etrefi['time']=utils.extract_filedate(m)
    clipped_etrefi.rio.to_raster('../prepro/Majadas/' + m.name + '.tif')
    
clipped_etrefi.rio.crs
    
#     ETref.append(clipped_etrefi)
    
# ETref_alldates = xr.concat(ETref, dim='time')

# # fig, ax = plt.subplots()

# ETref_alldates.isel(band=0,time=0).plot.imshow()
# ETref_alldates.isel(band=0,time=1).plot.imshow()


#%%
fig, ax = plt.subplots()
# ETref_alldates.isel(time=0,band=0).plot.imshow(ax=ax)
DTM_rxr.isel(band=0).plot.imshow(ax=ax,vmin=0)
fill_value = DTM_rxr.attrs['_FillValue']
DTM_rxr = DTM_rxr.where(DTM_rxr != fill_value, -9999)
DTM_rxr.attrs['_FillValue'] = -9999

# majadas_aoi.plot(ax=ax)

DTM_rxr.rio.resolution()

# clipped_etrefi.isel(band=0).plot.imshow()
# clipped_DTM_rxr.rio.resolution()
# etrefi.rio.resolution()
# from rasterio.enums import Resampling
# # etrefi_upsampled = etrefi.rio.reproject(
# #                                     etrefi.rio.crs,
# #                                     shape=(
# #                                             int(clipped_DTM_rxr.rio.resolution()[0]), 
# #                                             int(abs(clipped_DTM_rxr.rio.resolution()[1])), 
# #                                            ),
# #                                     resampling=Resampling.bilinear,
# #                                 )
# clipped_DTM_rxr_upsampled = clipped_DTM_rxr.rio.reproject(
#                                     clipped_DTM_rxr.rio.crs,
#                                     shape=(
#                                             int(etrefi.rio.resolution()[0]), 
#                                             int(abs(etrefi.rio.resolution()[1])), 
#                                            ),
#                                     resampling=Resampling.bilinear,
#                                 )

# clipped_DTM_rxr_upsampled.isel(band=0).plot.imshow()


