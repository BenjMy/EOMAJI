#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:50:23 2023
@author: ben

Prepare Majadas inputs:
    - TSEB tif files to xarray dataset (compatible with CATHY)
    #- Corinne Land Cover shapefile to raster
    #- Plot DEM, CLC, TSEB ouptuts, ...
"""
import geopandas as gpd
import rioxarray as rxr
import xarray as xr
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from geocube.api.core import make_geocube
import matplotlib.colors as mcolors
import BurkinaFaso_utils

import sys
# Get the parent directory and add it to sys.path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

import EOMAJI_utils
from centum import utils
plt.rcParams['font.family'] = 'serif'  # You can also use 'Times New Roman', 'STIXGeneral', etc.

fieldsite = 'BurkinaFaso'
#%% Define path and crs projection
AOI = 'burkina_faso_aoi'
reprocess = False
shapefile_raster_resolution = 300

rootPath = Path('../../../')
prepoEOPath = Path(f'/run/media/z0272571a/LVM_16Tb/Ben/EOMAJI/prepro/{fieldsite}')
rootDataPath_ECMWF = Path(f'/run/media/z0272571a/LVM_16Tb/Ben/EOMAJI/data/{fieldsite}/ECMWF/E024N060T6/')
rootDataPath_tile1 = Path(f'/run/media/z0272571a/LVM_16Tb/Ben/EOMAJI/data/{fieldsite}/X0027_Y0029/')
rootDataPath_tile2 = Path(f'/run/media/z0272571a/LVM_16Tb/Ben/EOMAJI/data/{fieldsite}/X0028_Y0029/')
figPath = rootPath/'figures/BurkinaFaso_test'

#%% Define files outputs from TSEB for Majadas
# -----------------------------------------------------------------------------
file_pattern = '*TPday*.tif'
rain_filelist = list(rootDataPath_ECMWF.glob(file_pattern))

ET_0_filelist = []
ET_filelist = []
for rootDataPath_tilei in [rootDataPath_tile1, rootDataPath_tile2]:
    # file_pattern = '*ET_0*.tif'
    file_pattern = '*ET_0-gf*.tif'
    ET_0_filelist.append(list(rootDataPath_tilei.glob(file_pattern)))
    file_pattern = '*ET-gf*.tif'
    ET_filelist.append(list(rootDataPath_tilei.glob(file_pattern)))

ET_filelist = np.hstack(ET_filelist)
ET_0_filelist = np.hstack(ET_0_filelist)

crs_ET = rxr.open_rasterio(ET_0_filelist[0]).rio.crs
ET_test = rxr.open_rasterio(ET_0_filelist[0])

# ET_filelist = ET_filelist[0:10]
# ET_0_filelist = ET_0_filelist[0:10]
# rain_filelist = rain_filelist[0:10]

#%% Read AOI points and plots
# -----------------------------------------------------------------------------
# BurkinaFaso_aoi = gpd.read_file(f'../../data/AOI/{AOI}.geojson')
# BurkinaFaso_aoi_reproj = BurkinaFaso_aoi.to_crs(crs_ET)

BurkinaFaso_aoi_reproj = BurkinaFaso_utils.get_AOI_POI_BurkinaFaso(AOI,crs_ET)

fig, ax = plt.subplots()
ET_test.isel(band=0).plot.imshow(ax=ax)
BurkinaFaso_aoi_reproj.plot(ax=ax)
# dd
#%% Read and CLIP ! Majadas grids: S3/Meteo = E030N006T6 and S2/Landast = X0033_Y0044
# S3/Meteo EPGS CRS EPSG:27704 - WGS 84 / Equi7 Europe - Projected
# ss
if reprocess:
    #% CLIP to Majadas bassin
    # -------------------------------------------------------------------------
    EOMAJI_utils.clip_rioxarray(
                                fieldsite,
                                ET_filelist,
                                ET_0_filelist,
                                rain_filelist,
                                BurkinaFaso_aoi_reproj,
                                prepoEOPath,
                                addtilename = True,
                                )

#%%
tiles = ['X0027_Y0029', 'X0028_Y0029']
ETa_ALL = []
ETp_ALL = []
RAIN_ALL = []

for tilei in tiles:
    ETp, ETa, RAIN = EOMAJI_utils.export_tif2netcdf(
                                    # pathTif2read=prepoEOPath/f'X0027_Y0029/',
                                    pathTif2read=prepoEOPath,
                                    fieldsite=fieldsite,
                                    tile=tilei
                                   )

    ETp.to_netcdf(prepoEOPath/f'../../prepro/{fieldsite}/{tilei}_ETp_{fieldsite}.netcdf')
    ETa.to_netcdf(prepoEOPath/f'../../prepro/{fieldsite}/{tilei}_ETa_{fieldsite}.netcdf')
    RAIN.to_netcdf(prepoEOPath/f'../../prepro/{fieldsite}/{tilei}_RAIN_{fieldsite}.netcdf')


    #% Read CLIPPED and save as NETCDF
    # -------------------------------------------------------------------------
    ETa_ds = xr.open_dataset(prepoEOPath/f'{tilei}_ETa_{fieldsite}.netcdf')
    ETa_ds = ETa_ds.rename({"__xarray_dataarray_variable__": "ETa"})
    ETa_ALL.append(ETa_ds)
    
    ETp_ds = xr.open_dataset(prepoEOPath/f'{tilei}_ETp_{fieldsite}.netcdf')
    ETp_ds = ETp_ds.rename({"__xarray_dataarray_variable__": "ETp"})
    ETp_ALL.append(ETp_ds)

    # RAIN_ds = xr.open_dataset(prepoEOPath/f'{tilei}_RAIN_{fieldsite}.netcdf')
    # RAIN_ds = RAIN_ds.rename({"__xarray_dataarray_variable__": "RAIN"})
    # RAIN_ALL.append(RAIN_ds)

ETa_ALL_merged = xr.merge(ETa_ALL).isel(band=0).sortby('time')
ETp_ALL_merged = xr.merge(ETp_ALL).isel(band=0).sortby('time')
RAIN_ALL_merged = RAIN.to_dataset(name='RAIN').isel(band=0).sortby('time')
# )).isel(band=0).sortby('time')

# ETa_ALL_merged = ETa_ALL_merged.rio.write_crs(crs_ET)
# ETp_ALL_merged= ETp_ALL_merged.rio.write_crs(crs_ET)
# RAIN_ALL_merged = RAIN_ALL_merged.rio.write_crs(crs_ET)

ETa_ALL_merged.to_netcdf(prepoEOPath/f'../../prepro/{fieldsite}/ETa_{fieldsite}.netcdf')
ETp_ALL_merged.to_netcdf(prepoEOPath/f'../../prepro/{fieldsite}/ETp_{fieldsite}.netcdf')
RAIN_ALL_merged.to_netcdf(prepoEOPath/f'../../prepro/{fieldsite}/RAIN_{fieldsite}.netcdf')
# RAIN_ALL_merged.to_netcdf(f'RAIN_{fieldsite}.netcdf')



print(ETa_ALL[0].time)
print(ETa_ALL[1].time)
#%% Corinne land cover shapefile to raster
# -----------------------------------------------------------------------------
clc_codes = utils.get_CLC_code_def()
CLC_BurkinaFaso_clipped = BurkinaFaso_utils.get_LandCoverMap()
CLC_BurkinaFaso_clipped['landcover'].plot.imshow()

#%%
fig, ax = plt.subplots()

# RAIN_ALL[0]['ETa'].isel(band=0,time=0).plot.imshow(ax=ax)
# RAIN_ALL[0]['RAIN'].isel(band=0,time=0).plot.imshow(ax=ax)
# RAIN[0].isel(band=0).plot.imshow(ax=ax)
ETa[0].isel(band=0).plot.imshow(ax=ax)
BurkinaFaso_aoi_reproj.boundary.plot(
    ax=ax, 
    edgecolor='red', 
    linewidth=1.5,
    linestyle='--'
)


# RAIN = RAIN_ds.to_dataarray().isel(variable=0,band=0).sortby('time')


# fig, axs = plt.subplots(1,2, sharey=True)
# ETa_ALL[0]['ETa'].isel(band=0,time=0).plot.imshow(ax=axs[0])
# ETa_ALL[1]['ETa'].isel(band=0,time=0).plot.imshow(ax=axs[1])


fig, ax = plt.subplots()

ETa_ALL[0]['ETa'].isel(band=0,time=0).plot.imshow(ax=ax)
ETa_ALL[1]['ETa'].isel(band=0,time=0).plot.imshow(ax=ax)
BurkinaFaso_aoi_reproj.boundary.plot(
    ax=ax, 
    edgecolor='red', 
    linewidth=1.5,
    linestyle='--'
)

fig, axs = plt.subplots(1,2,sharey=True)
axs = axs.ravel()

for i in range(2):
    ETa_ALL_merged['ETa'].isel(time=i+100).plot.imshow(ax=axs[i],
                                                   vmax=7)
    BurkinaFaso_aoi_reproj.boundary.plot(
        ax=axs[i], 
        edgecolor='red', 
        linewidth=1.5,
        linestyle='--'
    )



fig, ax = plt.subplots()

RAIN_ALL_merged['RAIN'].isel(time=10).plot.imshow(ax=ax)
BurkinaFaso_aoi_reproj.boundary.plot(
    ax=ax, 
    edgecolor='red', 
    linewidth=1.5,
    linestyle='--'
)

# np.sum(RAIN_ALL_merged['RAIN'])


# ETa_ds['ETa'].isel(band=0,time=0).plot.imshow(ax=axs[0])
# ETa_merged['ETa'].isel(time=0).plot.imshow(ax=axs[1])


