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

import pyCATHY 
from pyCATHY import CATHY
from pyCATHY.importers import cathy_inputs as in_CT
import matplotlib.pyplot as plt

#%% Define path and crs projection 

# rootDataPath = Path('/run/media/z0272571a/LVM_16Tb/Ben/EOMAJI/MAJADAS/')
prepoEOPath = Path('/run/media/z0272571a/SENET/iberia_daily/E030N006T6')
rootDataPath = Path('/run/media/z0272571a/LVM_16Tb/Ben/EOMAJI/MAJADAS/')
figDataPath = Path('../figures/')
file_pattern = '*ET_0-gf*.tif'
folder_weather_path = rootDataPath/'E030N006T6'


file_pattern = '*ET_0*.tif'
ET_0_filelist = list(prepoEOPath.glob(file_pattern))

file_pattern = '*TPday*.tif'
rain_filelist = list(prepoEOPath.glob(file_pattern))


def extract_filedate(file_path):
    file_name = file_path.name
    date_str = file_name.split('_')[0]
    return datetime.strptime(date_str, '%Y%m%d')

crs_ET_0 = rio.open_rasterio(ET_0_filelist[0]).rio.crs

#%% Read AOI points and plots

majadas_aoi = gpd.read_file('../data/AOI/majadas_aoi.geojson')
majadas_aoi.crs
majadas_aoi.to_crs(crs_ET_0, inplace=True)


majadas_POIs = gpd.read_file('../data/AOI/POI_Majadas.geojson')
majadas_POIs.to_crs(crs_ET_0, inplace=True)
majadas_aoi = gpd.read_file('../data/AOI/POI_Majadas.kmz')


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

fig.savefig(figDataPath/'DTM_Majadas.png', dpi=300)

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
#     clipped_etrefi['time']=extract_filedate(m)
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
#     clipped_etrefi['time']=extract_filedate(m)
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
#     ETpfi['time']=extract_filedate(m)
#     ETp_l.append(ETpfi)
#     ETp_dates.append(ETpfi['time'])

# rain = []
# rain_dates = []
# for m in rain_clipped_filelist:
#     rainfi = rio.open_rasterio(m)
#     rainfi['time']=extract_filedate(m)
#     rain.append(rainfi)
#     rain_dates.append(rainfi['time'])

# ETp = xr.concat(ETp_l,dim='time')
# ETp.to_netcdf('../prepro/ETp_Majadas.netcdf')
# RAIN = xr.concat(rain,dim='time')
# RAIN.to_netcdf('../prepro/RAIN_Majadas.netcdf')


#%%
ETp = xr.open_dataset('../prepro/ETp_Majadas.netcdf')
ETp = ETp.rename({"__xarray_dataarray_variable__": "ETp"})
RAIN = xr.open_dataset('../prepro/RAIN_Majadas.netcdf')
RAIN = RAIN.rename({"__xarray_dataarray_variable__": "RAIN"})

ETp['ETp'].isel(time=0,band=0).plot.imshow()
RAIN['RAIN'].isel(time=0,band=0).plot.imshow()

#%% Check variations on POI


multipoint_geom = majadas_POIs.geometry.iloc[0]
POIs_coords = np.array([point.coords[0] for point in multipoint_geom.geoms])


#%%
labels_POIs = ['Lake','Intensive Irrigation','Tree-Grass']
fig, ax = plt.subplots()
for i, ppc in enumerate(POIs_coords):
    ETp_poi = ETp.sel(x=ppc[0],y=ppc[1], method="nearest").isel(band=0)['ETp']
    ETp_poi_sorted = ETp_poi.sortby('time')
    ax.scatter(ETp_poi.time,ETp_poi.values,label=labels_POIs[i])
ax.legend()
        
fig, ax = plt.subplots()
for ppc in POIs_coords:
    RAIN_poi = RAIN.sel(x=ppc[0],y=ppc[1], method="nearest").isel(band=0)['RAIN']
    ax.scatter(RAIN_poi.time,RAIN_poi.values)

#%%
fig, ax = plt.subplots()
DTM_rxr.isel(band=0).plot.imshow(vmin=0)

#%% Read Majadas grids: S3/Meteo = E030N006T6 and S2/Landast = X0033_Y0044
# S3/Meteo EPGS CRS EPSG:27704 - WGS 84 / Equi7 Europe - Projected

# dates = []
# ETref = []
for m in ET_0_filelist:
    # dates.append(extract_filedate(m))
    etrefi = rio.open_rasterio(m)
    clipped_etrefi = etrefi.rio.clip_box(
                                         minx=majadas_aoi.bounds['minx'],
                                         miny=majadas_aoi.bounds['miny'],
                                         maxx=majadas_aoi.bounds['maxx'],
                                         maxy=majadas_aoi.bounds['maxy'],
                                        crs=majadas_aoi.crs,
                                        )   
    clipped_etrefi['time']=extract_filedate(m)
    clipped_etrefi.rio.to_raster('../prepro/Majadas/' + m.name + '.tif')
    
for m in rain_filelist:
    # dates.append(extract_filedate(m))
    etrefi = rio.open_rasterio(m)
    clipped_etrefi = etrefi.rio.clip_box(
                                         minx=majadas_aoi.bounds['minx'],
                                         miny=majadas_aoi.bounds['miny'],
                                         maxx=majadas_aoi.bounds['maxx'],
                                         maxy=majadas_aoi.bounds['maxy'],
                                        crs=majadas_aoi.crs,
                                        )   
    clipped_etrefi['time']=extract_filedate(m)
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

#%% Create CATHY mesh based on DEM

aa
#%% Create CATHY mesh based on DEM

import pyCATHY 
from pyCATHY import CATHY
from pyCATHY.importers import cathy_inputs as in_CT
import matplotlib.pyplot as plt

hydro_Majadas = CATHY(
                        dirName='../hydro/',
                        prj_name="Majadas"
                      )

DEM_notopo = np.ones(np.shape(RAIN['RAIN'].isel(time=0,band=0)))
DEM_notopo[-1,-1]= 1-1e-3

#%% Create CATHY mesh based on DEM

# 

# # dem_mat, str_hd_dem = in_CT.read_dem(DTM_rxr.values)

# fig, ax = plt.subplots(1)
# img = ax.imshow(
#                 DTM_rxr.values[0],
#                 vmin=0
#                 )
# plt.colorbar(img)


# hydro_Majadas.show_input(prop="dem")



#%% Update prepro inputs and mesh
hydro_Majadas.update_prepo_inputs(
                                DEM=DEM_notopo,
                                # N=np.shape(dem_mat)[1],
                                # M=np.shape(dem_mat)[0],
                                xllcorner=RAIN.x.min().values,
                                yllcorner=RAIN.y.min().values
# dem_mat, str_hd_dem = in_CT.read_dem(DTM_rxr.values)

fig, ax = plt.subplots(1)
img = ax.imshow(
                DTM_rxr.values[0],
                vmin=0
                )
plt.colorbar(img)


hydro_Majadas.show_input(prop="dem")

hydro_Majadas.update_prepo_inputs(
                                DEM=DTM_rxr.values[0],
                                # N=np.shape(dem_mat)[1],
                                # M=np.shape(dem_mat)[0],
                            )

fig = plt.figure()
ax = plt.axes(projection="3d")
hydro_Majadas.show_input(prop="dem", ax=ax)

hydro_Majadas.run_preprocessor(verbose=True)

hydro_Majadas.create_mesh_vtk(verbose=True)

#%% Update atmbc

# Determine the overlapping time range
start_time = max(RAIN.time.min(), ETp.time.min())
end_time = min(RAIN.time.max(), ETp.time.max())

# Create a mask for the common time range
mask_time = (RAIN.time >= start_time) & (RAIN.time <= end_time)
mask_time2 = (ETp.time >= start_time) & (ETp.time <= end_time)

# Filter the DataArrays using the mask
filtered_RAIN = RAIN['RAIN'].sel(time=mask_time).isel(band=0)
filtered_ETp = ETp['ETp'].sel(time=mask_time2).isel(band=0)

RAIN_3d = np.array([filtered_RAIN.isel(time=t).values for t in range(filtered_RAIN.sizes['time'])])
ETp_3d = np.array([filtered_ETp.isel(time=t).values for t in range(filtered_ETp.sizes['time'])])

# np.shape(RAIN_list2d)

filtered_RAIN['Elapsed_Time_s'] = (filtered_RAIN.time - filtered_RAIN.time[0]).dt.total_seconds()

# net_ATMBC_m_s = RAIN_list2d - 

hydro_Majadas.update_atmbc(HSPATM=0,
                           IETO=1,
                           time=filtered_RAIN['Elapsed_Time_s'] .values
                           # netValue=
                           )
hydro_Majadas.update_nansfdirbc()
hydro_Majadas.update_nansfneubc()
hydro_Majadas.update_sfbc()

