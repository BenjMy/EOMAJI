#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:50:23 2023
@author: ben

Prepare Majadas inputs: 
    - TSEB tif files to xarray dataset (compatible with CATHY)
    - Corinne Land Cover shapefile to raster 
    - Plot DEM, CLC, TSEB ouptuts, ...
"""
import geopandas as gpd
import rioxarray as rxr
import xarray as xr
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from geocube.api.core import make_geocube
import matplotlib.colors as mcolors

import Majadas_utils
from centum import utils
plt.rcParams['font.family'] = 'serif'  # You can also use 'Times New Roman', 'STIXGeneral', etc.

#%% Define path and crs projection 
# AOI = 'Buffer_5000' #H2_Bassin
# AOI = 'Buffer_20000' #H2_Bassin
# AOI = 'Buffer_100'
AOI = 'H2_Bassin'
reprocess = False
shapefile_raster_resolution = 300

rootPath = Path('../../')
prepoEOPath = Path('/run/media/z0272571a/SENET/iberia_daily/E030N006T6')
rootDataPath = Path('/run/media/z0272571a/LVM_16Tb/Ben/EOMAJI/MAJADAS/')
figPath = Path('../../figures/Majadas_test')
folder_weather_path = rootDataPath/'E030N006T6'

crs_ET_0 = Majadas_utils.get_crs_ET()

#%% Read AOI points and plots
# -----------------------------------------------------------------------------
# majadas_aoi = Majadas_utils.get_Majadas_aoi()
# majadas_aoi = Majadas_utils.get_Majadas_aoi(buffer=20000)
majadas_aoi = gpd.read_file('../../data/Spain/GIS_catchment_majadas/BassinH2_Majadas_corrected.shp')
# majadas_aoi.to_crs(crs_ET_0, inplace=True)
majadas_POIs, POIs_coords = Majadas_utils.get_Majadas_POIs()

#%% Define files outputs from TSEB for Majadas
# -----------------------------------------------------------------------------
# file_pattern = '*ET_0*.tif'
file_pattern = '*ET_0-gf*.tif'
ET_0_filelist = list(prepoEOPath.glob(file_pattern))
file_pattern = '*ET-gf*.tif'
ET_filelist = list(prepoEOPath.glob(file_pattern))
file_pattern = '*TPday*.tif'
rain_filelist = list(prepoEOPath.glob(file_pattern))
crs_ET = rxr.open_rasterio(ET_0_filelist[0]).rio.crs
ET_test = rxr.open_rasterio(ET_0_filelist[0])

#%% Read and CLIP ! Majadas grids: S3/Meteo = E030N006T6 and S2/Landast = X0033_Y0044
# S3/Meteo EPGS CRS EPSG:27704 - WGS 84 / Equi7 Europe - Projected
# ss
if reprocess:
    #% CLIP to Majadas bassin
    # -------------------------------------------------------------------------
    Majadas_utils.clip_rioxarray(ET_filelist,
                          ET_0_filelist,
                          rain_filelist,
                          majadas_aoi
                          )
    #% Read CLIPPED and save as NETCDF
    # -------------------------------------------------------------------------
     # ! Majadas grids: S3/Meteo = E030N006T6 and S2/Landast = X0033_Y0044
    # S3/Meteo EPGS CRS EPSG:27704 - WGS 84 / Equi7 Europe - Projected
    Majadas_utils.export_tif2netcdf(fieldsite='Majadas')
    
    ETa_ds = xr.open_dataset(rootPath / 'prepro/Majadas/{AOI}/ETa_Majadas.netcdf')
    ETa_ds = ETa_ds.rename({"__xarray_dataarray_variable__": "ETa"})
    ETp_ds = xr.open_dataset('../prepro/Majadas/{AOI}/ETp_Majadas.netcdf')
    ETp_ds = ETp_ds.rename({"__xarray_dataarray_variable__": "ETp"})
    RAIN_ds = xr.open_dataset('../prepro/Majadas/{AOI}/RAIN_Majadas.netcdf')
    RAIN_ds = RAIN_ds.rename({"__xarray_dataarray_variable__": "RAIN"})
else:   
    ETa_ds = xr.open_dataset(rootPath /f'prepro/Majadas/{AOI}/ETa_Majadas.netcdf')
    ETa_ds = ETa_ds.rename({"__xarray_dataarray_variable__": "ETa"})
    ETp_ds = xr.open_dataset(rootPath /f'prepro/Majadas/{AOI}/ETp_Majadas.netcdf')
    ETp_ds = ETp_ds.rename({"__xarray_dataarray_variable__": "ETp"})
    RAIN_ds = xr.open_dataset(rootPath /f'prepro/Majadas/{AOI}/RAIN_Majadas.netcdf')
    RAIN_ds = RAIN_ds.rename({"__xarray_dataarray_variable__": "RAIN"})

#%% Read TDR Majadas
# -----------------------------------------------------------------------------
coord_SWC_CT, gdf_SWC_CT = Majadas_utils.get_SWC_pos(
                                                    target_crs=None
                                                    )
#%% Download basemap from leafmap 
# -----------------------------------------------------------------------------
# import leafmap
# majadas_aoi_reproj = majadas_aoi.to_crs("EPSG:4326")  # returns (minx, miny, maxx, maxy)
# bbox = majadas_aoi_reproj.total_bounds  # returns (minx, miny, maxx, maxy)
# leafmap.map_tiles_to_geotiff("../data/satellite.tif", 
#                              list(bbox), 
#                              zoom=14,
#                               # resolution=300,
#                               source="Esri.WorldImagery"
#                              )

# # Open the downloaded image using rioxarray
# Majadas_satellite = rxr.open_rasterio("../data/satellite.tif")
# Majadas_satellite = Majadas_satellite.rio.reproject(ET_test.rio.crs)
# Majadas_satellite.rio.crs 
# majadas_aoi = gpd.read_file('../data/AOI/POI_Majadas.kmz')
#%% Corinne land cover shapefile to raster 
# -----------------------------------------------------------------------------
clc_codes = utils.get_CLC_code_def()
CLC_Majadas = Majadas_utils.get_LandCoverMap()
CLC_clipped = gpd.clip(CLC_Majadas, 
                        majadas_aoi
                        )
CLC_clipped.columns
CLC_clipped.to_file('../../prepro/Majadas/CLC_Majadas_clipped.shp')
# sdd
# clc_codes_int = [int(clci) for clci in clc_codes.keys()]
categorical_enums = {'Code_18': clc_codes}

# shapefile to raster with 300 m resolution
# -----------------------------------------
CLC_Majadas_clipped_grid = make_geocube(
    vector_data=CLC_clipped,
    output_crs=crs_ET,
    # group_by='Land_Cover_Name',
    resolution=(-shapefile_raster_resolution, shapefile_raster_resolution),
    categorical_enums= categorical_enums,
    # fill=np.nan
)

Code_18_Majadas = CLC_Majadas_clipped_grid['Code_18'].astype(int)
Code_18_categories = CLC_Majadas_clipped_grid['Code_18_categories']
Code_18_string = Code_18_categories[Code_18_Majadas].drop('Code_18_categories')

CLC_Majadas_clipped_grid['Code_CLC'] = Code_18_string
CLC_Majadas_clipped_grid = CLC_Majadas_clipped_grid.rio.write_crs(crs_ET)
nodata_value = CLC_Majadas_clipped_grid.Code_18.rio.nodata

# CLC_Majadas_clipped_grid['Code_18']= CLC_Majadas_clipped_grid.Code_18.where(CLC_Majadas_clipped_grid.Code_18 != nodata_value, -9999)
# Step 1: Replace 'nodata' with NaN
CLC_Majadas_clipped_grid['Code_CLC']= CLC_Majadas_clipped_grid.Code_CLC.where(
    CLC_Majadas_clipped_grid.Code_CLC != 'nodata', other=np.nan
)

# Step 2: Convert the remaining string values to integers
CLC_Majadas_clipped_grid['Code_CLC'] = CLC_Majadas_clipped_grid.Code_CLC.astype(float)

# export to netcdf
# -----------------------------------------
CLC_Majadas_clipped_grid.to_netcdf(f'../../prepro/Majadas/{AOI}/CLCover_Majadas.netcdf')

# plot
# -----------------------------------------
fig, ax = plt.subplots()
CLC_Majadas_clipped_grid.Code_CLC.plot(cmap='Paired',
                                      vmin=0
                                      )

unique_labels = np.unique(CLC_Majadas_clipped_grid.Code_CLC.values) #[:-1]
# s
#%% Plot sum up preparation of inputs 
# -----------------------------------------------------------------------------


# # Extract the categorical mapping (if available)
categories = CLC_Majadas_clipped_grid['Code_18_categories'].values

# # Create a dictionary mapping the Code_18 values to the categories
category_mapping = {i: cat for i, cat in enumerate(categories)}

# # Extract the Code_18 data
# code_18_data = CLC_Majadas_clipped_grid['Code_18']

# Define a colormap for the categories
cmap = mcolors.ListedColormap(plt.cm.get_cmap('tab20').colors[:len(unique_labels)])
norm = mcolors.BoundaryNorm(boundaries=range(len(unique_labels) + 1), ncolors=len(unique_labels))

# Plot using imshow with the defined colormap
fig, axs = plt.subplots(1,3,
                        sharex=True,
                        sharey=True
                        )
axs = axs.flatten()
CLC_clipped.plot('Code_18',ax=axs[0])


CLC_Majadas_clipped_grid.Code_18.plot(cmap='Paired',
                                      vmin=0,
                                      ax=axs[1]
                                      )
axs[1].set_title('')
axs[1].set_xlabel('')
axs[1].set_ylabel('')

majadas_aoi.boundary.plot(ax=axs[0],color='r')
majadas_POIs.plot(ax=axs[0],color='r')
majadas_aoi.boundary.plot(ax=axs[1],color='r')

# Create a custom legend
legend_labels = {i: category_mapping[i] for i in range(len(categories))}
legend_patches = [plt.Line2D([0], [0], marker='o', color='w', label=legend_labels[i],
                             markerfacecolor=cmap(i), markersize=10) for i in range(len(categories))]
# Show the plot
majadas_aoi.boundary.plot(ax=axs[1],color='r')
try:
    ETp_ds.isel(band=0)['ETp'].isel(time=0).plot.imshow(ax=axs[2])
except:
    pass
majadas_aoi.boundary.plot(ax=axs[2],color='r')
axs[2].set_title('')
axs[2].set_xlabel('')
axs[2].set_ylabel('')

# cx.add_basemap(ax=axs[1],
#                source=cx.providers.Esri.WorldImagery
#                )
fig.savefig(figPath/f'LandCoverRaster_Majadas_{AOI}.png',dpi=300)

#%% Read Majadas DTM
# import os
# os.getcwd()
clipped_DTM_rxr = rxr.open_rasterio(f'../../data/Spain/clipped_DTM_Majadas_AOI_{AOI}.tif', 
                                    )
#%% plot majadas DTM 
fig, ax = plt.subplots()
clipped_DTM_rxr.isel(band=0).plot.imshow(ax=ax)
majadas_POIs.plot(ax=ax,color='red')
majadas_aoi.plot(ax=ax,edgecolor='black', facecolor='none')
ax.set_title(f'DTM_Majadas_{AOI}')
fig.savefig(figPath/f'DTM_Majadas_{AOI}.png', dpi=300)

#%%

ETp = ETp_ds.to_dataarray().isel(variable=0,band=0).sortby('time')
RAIN = RAIN_ds.to_dataarray().isel(variable=0,band=0).sortby('time')
print('Errrrrorrr in rain evaluation in the input!')
RAIN = RAIN.where((RAIN <= 300) & (RAIN > 0), other=0)


ds_analysis_EO = ETa_ds.isel(band=0)
ds_analysis_EO['ETp'] = ETp
ds_analysis_EO['RAIN'] = RAIN
CLC_Majadas_clipped_grid_no_spatial_ref = CLC_Majadas_clipped_grid.drop_vars('spatial_ref', errors='ignore')
ds_analysis_EO['CLC'] = CLC_Majadas_clipped_grid_no_spatial_ref.Code_18
ds_analysis_EO.to_netcdf(f'../../prepro/ds_analysis_EO_{AOI}.netcdf')
ds_analysis_EO = ds_analysis_EO.sortby('time')

nulltimeETa = np.where(ds_analysis_EO.ETa.isel(x=0,y=0).isnull())[0]
valid_mask = ~ds_analysis_EO.time.isin(ds_analysis_EO.time[nulltimeETa])
ds_analysis_EO = ds_analysis_EO.isel(time=valid_mask)

ds_analysis_EO = utils.compute_ratio_ETap_local(ds_analysis_EO,
                                                 ETa_name='ETa',
                                                 ETp_name='ETp',
                                                )

ds_analysis_EO = utils.compute_regional_ETap(ds_analysis_EO,
                                             ETa_name='ETa',
                                             ETp_name='ETp',
                                             window_size_x=5
                                             )
    
ds_analysis_EO = utils.compute_ratio_ETap_regional(
                                                ds_analysis_EO,
                                                ETa_name='ETa',
                                                ETp_name='ETp',
                                                )

fig, axs = plt.subplots(2,2)
axs = axs.ravel()
ds_analysis_EO['ETa'].isel(time=10).plot.imshow(ax=axs[0])
ds_analysis_EO['ETp'].isel(time=10).plot.imshow(ax=axs[1])
ds_analysis_EO['ratio_ETap_local'].isel(time=10).plot.imshow(ax=axs[2])

#%% Check variations on POI

# labels_POIs = ['Lake','Intensive Irrigation','Tree-Grass', 'Agricutural fields']

# fig, ax = plt.subplots()
# for i, ppc in enumerate(POIs_coords):
#     ETa_poi = ds_analysis_EO['ETa'].sel(x=ppc[0],y=ppc[1], 
#                                         method="nearest")
#     ax.plot(ETa_poi.time.values,ETa_poi,label=labels_POIs[i])
# ax.legend()
# ax.set_xlabel('Datetime')
# ax.set_ylabel('ETa (mm/day)')
# fig.savefig(figPath/'ETa_serie_Majadas.png', dpi=300)



# fig, ax = plt.subplots()
# for i, ppc in enumerate(POIs_coords):
#     RAIN_poi = ds_analysis_EO['RAIN'].sel(x=ppc[0],y=ppc[1],
#                                           method="nearest")
#     ax.scatter(RAIN_poi.time,RAIN_poi.values,label=labels_POIs[i])
# ax.legend()
# ax.set_xlabel('Datetime')
# ax.set_ylabel('Rain (mm)')

# fig.savefig(figPath/'rain_serie_Majadas.png', dpi=300)


# fig, ax = plt.subplots()
# for i, ppc in enumerate(POIs_coords):
#     ETap_poi = ds_analysis_EO['ratio_ETap_local'].sel(x=ppc[0],y=ppc[1], 
#                                                       method="nearest")
#     ax.scatter(ETap_poi.time,ETap_poi.values,label=labels_POIs[i])
# ax.legend()
# ax.set_xlabel('Datetime')
# ax.set_ylabel('ETap local (mm/day)')

# fig.savefig(figPath/'ETap_local_serie_Majadas.png', dpi=300)

#%%

# import matplotlib.pyplot as plt
# import pandas as pd
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# # Create the main figure and axis
# fig, ax = plt.subplots()

# # Plot the main data series
# for i, ppc in enumerate(POIs_coords):
#     ETp_poi = ds_analysis_EO['ETp'].sel(x=ppc[0], y=ppc[1], method="nearest")
#     ax.plot(ETp_poi.time.values, ETp_poi, label=labels_POIs[i])

# # Set labels and legend for the main plot
# ax.legend()
# ax.set_xlabel('Datetime')
# ax.set_ylabel('ETp (mm/day)')

# # Define the time range for the inset (6 months from 03/2022 to 09/2022)
# inset_start = pd.Timestamp('2020-05-01')
# inset_end = pd.Timestamp('2020-10-01')

# # Create an inset axis within the main plot
# ax_inset = inset_axes(ax, width="40%", height="40%", loc='upper right')

# # Plot the data within the specified time range in the inset plot
# for i, ppc in enumerate(POIs_coords):
#     ETp_poi = ds_analysis_EO['ETp'].sel(x=ppc[0], y=ppc[1], method="nearest")
#     ax_inset.plot(ETp_poi.time.values, ETp_poi, label=labels_POIs[i])

# # Set the limits for the inset to zoom in on the specified time range
# ax_inset.set_xlim(inset_start, inset_end)
# ax_inset.set_ylim(ETp_poi.sel(time=slice(inset_start, inset_end)).min(), 
#                   ETp_poi.sel(time=slice(inset_start, inset_end)).max())

# # Optionally, remove the x and y labels for the inset for clarity
# # ax_inset.set_xticks([])
# # ax_inset.set_yticks([])

# # Indicate the zoomed area on the main plot
# mark_inset(ax, ax_inset, loc1=2, loc2=4, fc="none", ec="0.5")

# # Save the figure with the inset plot and zoom indication
# fig.savefig(figPath/'ETp_serie_Majadas.png', dpi=300)
# plt.show()



#%%
# # Define the mosaic layout
# mosaic_layout = """
#                 a
#                 b
#                 b
#                 b
#                 """
# fig, ax = plt.subplot_mosaic(mosaic_layout,
#                              sharex=True,
#                              figsize=(8,4)
#                              )

# # Plot the first dataset on the primary y-axis
# # ax['b'].plot( )  
# for i, ppc in enumerate(POIs_coords):
#     ETap_poi = ds_analysis_EO['ratio_ETap_local'].sel(x=ppc[0],y=ppc[1], 
#                                                       method="nearest")
#     ax['b'].scatter(ETap_poi.time,ETap_poi.values,label=labels_POIs[i])
# ax['b'].legend()
# ax['b'].set_xlabel('Datetime')
# ax['b'].set_ylabel('ETap local (mm/day)')


# for i, ppc in enumerate(POIs_coords):
#     RAIN_poi = ds_analysis_EO['RAIN'].sel(x=ppc[0],y=ppc[1],
#                                           method="nearest",
#                                           )
#     ax['a'].scatter(RAIN_poi.time,RAIN_poi.values,label=labels_POIs[i])
# ax['a'].legend()
# ax['a'].set_xlabel('Datetime')
# ax['a'].invert_yaxis()
# ax['a'].set_ylabel('rain (mm)', color='b')

# plt.tight_layout()
# plt.savefig(figPath/'saturation_simu_Majadas.png',
#             dpi=300
#             )
