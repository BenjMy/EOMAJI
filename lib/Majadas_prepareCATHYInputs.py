#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:50:23 2023
@author: ben
"""

import geopandas as gpd
import shapely as shp
import rioxarray as rxr
import xarray as xr
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from rioxarray.merge import merge_arrays
from geocube.api.core import make_geocube
import matplotlib.colors as mcolors

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


crs_ET = rxr.open_rasterio(ET_0_filelist[0]).rio.crs
ET_test = rxr.open_rasterio(ET_0_filelist[0])

ETp_ds = xr.open_dataset('../prepro/ETp_Majadas.netcdf')
ETp_ds = ETp_ds.rename({"__xarray_dataarray_variable__": "ETp"})
RAIN_ds = xr.open_dataset('../prepro/RAIN_Majadas.netcdf')
RAIN_ds = RAIN_ds.rename({"__xarray_dataarray_variable__": "RAIN"})


#%% Read AOI points and plots
    

majadas_aoi = Majadas_utils.get_Majadas_aoi()

majadas_POIs, POIs_coords = Majadas_utils.get_Majadas_POIs()


# majadas_aoi = gpd.read_file('../data/AOI/POI_Majadas.kmz')
#%%
clc_codes = utils.get_CLC_code_def()

CLC_path = Path('../data/95732/Results/U2018_CLC2018_V2020_20u1.shp/U2018_CLC2018_V2020_20u1.shp')
CLC_Majadas = gpd.read_file(CLC_path)
CLC_Majadas = CLC_Majadas.to_crs(majadas_aoi.crs)
CLC_clipped = gpd.clip(CLC_Majadas, majadas_aoi)

# CLC_clipped['Land_Cover_Name'] = CLC_clipped['Code_18'].map(clc_codes)
# CLC_clipped.plot('Land_Cover_Name')

# CLC_Majadas.Code_18=CLC_Majadas.Code_18.astype(int)

categorical_enums = {'Code_18': CLC_clipped['Code_18']}

CLC_Majadas_clipped_grid = make_geocube(
    vector_data=CLC_clipped,
    output_crs=crs_ET,
    # group_by='Land_Cover_Name',
    resolution=(-100, 100),
    categorical_enums= categorical_enums
)

# CLC_Majadas_clipped_grid.rio.crs
# CLC_Majadas_clipped_grid = CLC_Majadas_clipped_grid.rio.reproject(crs_ET_0)


# CLC_Majadas_clipped_grid = CLC_Majadas_clipped_grid.transpose('y', 'x')
# CLC_Majadas_clipped_grid = CLC_Majadas_clipped_grid.transpose('x', 'y', 'Code_18_categories')


#%%

# rgb_Majadas = rxr.open_rasterio('../data/Majadas.tif')
# rgb_Majadas.rio.crs
# rgb_Majadas = rgb_Majadas.rio.reproject(crs_ET_0)

# rgb_image = rgb_Majadas.isel(band=[0, 1, 2])

# plt.figure(figsize=(10, 10))
# rgb_image.plot.imshow(rgb='band', ax=plt.gca())

#%%
import leafmap

bbox = majadas_aoi.total_bounds  # returns (minx, miny, maxx, maxy)

# m = leafmap.Map()
# image = m.add_basemap("SATELLITE")

# Define the region of interest (ROI) using the bounding box
# roi = [bbox[1], bbox[0], bbox[3], bbox[2]]  # [miny, minx, maxy, maxx]

# Download the satellite image for the specified ROI
# Saving the image to a file
# image_path = "satellite_image.tif"
# m.download_ee_image_to_local(
#     image, output=image_path, region=roi, scale=10, crs="EPSG:4326"
# )
leafmap.map_tiles_to_geotiff("satellite.tif", 
                             bbox, 
                             zoom=13, 
                             source="Esri.WorldImagery"
                             )

# # Open the downloaded image using rioxarray
# raster = rxr.open_rasterio(image_path)

# # Clip the image using the shapefile
# clipped_raster = raster.rio.clip(majadas_aoi.geometry, majadas_aoi.crs)

#%%

# Assuming `raster_xarray` is your xarray.Dataset with the variables loaded

# Extract the categorical mapping (if available)
categories = CLC_Majadas_clipped_grid['Code_18_categories'].values

# Create a dictionary mapping the Code_18 values to the categories
category_mapping = {i: cat for i, cat in enumerate(categories)}

# Extract the Code_18 data
code_18_data = CLC_Majadas_clipped_grid['Code_18']

# Define a colormap for the categories
cmap = mcolors.ListedColormap(plt.cm.get_cmap('tab20').colors[:len(categories)])
norm = mcolors.BoundaryNorm(boundaries=range(len(categories) + 1), ncolors=len(categories))

# Plot using imshow with the defined colormap
fig, axs = plt.subplots(2,2,
                        # sharex=True,
                        # sharey=True
                        )
axs = axs.flatten()
CLC_clipped.plot('Code_18',ax=axs[0])
# plt.figure(figsize=(12, 8))
code_18_data.plot.imshow(cmap=cmap, norm=norm, add_colorbar=False,
                         ax=axs[1])

majadas_aoi.boundary.plot(ax=axs[0],color='r')
majadas_POIs.plot(ax=axs[0],color='r')
majadas_aoi.boundary.plot(ax=axs[1],color='r')

# Create a custom legend
legend_labels = {i: category_mapping[i] for i in range(len(categories))}
legend_patches = [plt.Line2D([0], [0], marker='o', color='w', label=legend_labels[i],
                             markerfacecolor=cmap(i), markersize=10) for i in range(len(categories))]

axs[1].legend(handles=legend_patches, title="Land Cover Types", loc="lower right", fontsize='small', title_fontsize='medium')

# Show the plot
axs[1].set_title("Land Cover Map - Code_18")

# rgb_image.plot.imshow(rgb='band', ax=axs[2])
# majadas_aoi.boundary.plot(ax=axs[2],color='r')
# providers = cx.providers.flatten()

ETp_ds.isel(band=0)['ETp'].isel(time=0).plot.imshow(ax=axs[3])
majadas_aoi.boundary.plot(ax=axs[3],color='r')

# cx.add_basemap(ax=axs[1],
#                source=cx.providers.Esri.WorldImagery
#                )


fig.savefig(figPath/'LandCoverRaster_Majadas.png',dpi=300)

#%%
CLC_Majadas_clipped_grid.Code_18.isel(Land_Cover_Name=0).plot.imshow()
# CLC_string = CLC_Majadas_clipped_grid['Land_Cover_Name'][CLC_Majadas_clipped_grid['Code_18'].astype(int)].drop('Land_Cover_Name')
    
# CLC_Majadas_grid['drclassdcd'] = drclassdcd_string
# CLC_Majadas_grid


#%% Read Majadas DTM
# DTM EPSG:25830 - ETRS89 / UTM zone 30N - Projected


DTMfiles = [
            '../data/DTM/MDT02-ETRS89-HU30-0624-1-COB2_reproj.tif',
            '../data/DTM/MDT02-ETRS89-HU30-0624-3-COB2_reproj.tif',
            ]
DTM_rxr = []
for dtm in DTMfiles:
    DTM_rxr.append(rxr.open_rasterio(dtm))
DTM_rxr = merge_arrays(DTM_rxr)
# DTM_rxr.rio.crs= crs_ET_0
DTM_rxr.rio.write_crs(crs_ET, inplace=True)
DTM_rxr.rio.crs

clipped_DTM_rxr = DTM_rxr.rio.clip_box(
                                      minx=majadas_aoi.bounds['minx'],
                                      miny=majadas_aoi.bounds['miny'],
                                      maxx=majadas_aoi.bounds['maxx'],
                                      maxy=majadas_aoi.bounds['maxy'],
                                      crs=majadas_aoi.crs,
                                    )  


#%%

CLC_Majadas.plot('Land_Cover_Name')

#%%
CLC_Majadas.set_index('Code_18').to_xarray()
CLC_Majadas["Code_18"].rio.to_raster("my_rasterized_column.tif")



# CLC_Majadas.plot()
# CLC_Majadas.crs
# CLC_Majadas_reprojected = CLC_Majadas.to_crs(epsg=crs_ET_0)
CLC_Majadas['Code_18'].to_xarray()
CLC_Majadas.to_xarray()

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
    etrefi = rxr.open_rasterio(m)
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
    etrefi = rxr.open_rasterio(m)
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


