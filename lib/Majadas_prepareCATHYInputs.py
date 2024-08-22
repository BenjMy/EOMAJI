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
figPath = Path('../figures/Majadas')
folder_weather_path = rootDataPath/'E030N006T6'


# file_pattern = '*ET_0*.tif'
file_pattern = '*ET_0-gf*.tif'
ET_0_filelist = list(prepoEOPath.glob(file_pattern))

file_pattern = '*ET-gf*.tif'
ET_filelist = list(prepoEOPath.glob(file_pattern))

file_pattern = '*TPday*.tif'
rain_filelist = list(prepoEOPath.glob(file_pattern))


crs_ET = rxr.open_rasterio(ET_0_filelist[0]).rio.crs
ET_test = rxr.open_rasterio(ET_0_filelist[0])

ETa_ds = xr.open_dataset('../prepro/ETa_Majadas.netcdf')
ETa_ds = ETa_ds.rename({"__xarray_dataarray_variable__": "ETa"})
ETp_ds = xr.open_dataset('../prepro/ETp_Majadas.netcdf')
ETp_ds = ETp_ds.rename({"__xarray_dataarray_variable__": "ETp"})
RAIN_ds = xr.open_dataset('../prepro/RAIN_Majadas.netcdf')
RAIN_ds = RAIN_ds.rename({"__xarray_dataarray_variable__": "RAIN"})


#%% Read AOI points and plots
    

majadas_aoi = Majadas_utils.get_Majadas_aoi()

majadas_POIs, POIs_coords = Majadas_utils.get_Majadas_POIs()

#%% Read TDR

coord_SWC_CT, gdf_SWC_CT = Majadas_utils.get_SWC_pos(
                                                        target_crs=None
                                                        )


#%%
import leafmap

majadas_aoi_reproj = majadas_aoi.to_crs("EPSG:4326")  # returns (minx, miny, maxx, maxy)
bbox = majadas_aoi_reproj.total_bounds  # returns (minx, miny, maxx, maxy)
leafmap.map_tiles_to_geotiff("../data/satellite.tif", 
                             list(bbox), 
                             zoom=14,
                              # resolution=300,
                              source="Esri.WorldImagery"
                             )

# Open the downloaded image using rioxarray
Majadas_satellite = rxr.open_rasterio("../data/satellite.tif")
Majadas_satellite = Majadas_satellite.rio.reproject(ET_test.rio.crs)
Majadas_satellite.rio.crs 

# majadas_aoi = gpd.read_file('../data/AOI/POI_Majadas.kmz')
#%%
clc_codes = utils.get_CLC_code_def()

CLC_path = Path('../data/Copernicus_95732/U2018_CLC2018_V2020_20u1.shp/U2018_CLC2018_V2020_20u1.shp')
CLC_Majadas = gpd.read_file(CLC_path)
CLC_Majadas = CLC_Majadas.to_crs(majadas_aoi.crs)

# CLC_clipped = gpd.clip(CLC_Majadas, majadas_aoi)
CLC_clipped = gpd.clip(CLC_Majadas, 
                        mask=majadas_aoi.total_bounds
                        # mask= [
                        #         majadas_aoi.bounds['minx'].values[0],
                        #         majadas_aoi.bounds['miny'].values[0],
                        #         majadas_aoi.bounds['maxx'].values[0] #+300,
                        #         majadas_aoi.bounds['maxy'].values[0] #+300,    
                        #         ]
                        )

# categorical_enums = {'Code_18': list(clc_codes.values())}
clc_codes_int = [int(clci) for clci in clc_codes.keys()]
categorical_enums = {'Code_18': clc_codes}

CLC_Majadas_clipped_grid = make_geocube(
    vector_data=CLC_clipped,
    output_crs=crs_ET,
    # group_by='Land_Cover_Name',
    resolution=(-300, 300),
    categorical_enums= categorical_enums,
    # fill=np.nan
)
Code_18_string = CLC_Majadas_clipped_grid['Code_18_categories'][CLC_Majadas_clipped_grid['Code_18'].astype(int)]\
    .drop('Code_18_categories')
    
CLC_Majadas_clipped_grid['Code_18_str'] = Code_18_string
CLC_Majadas_clipped_grid = CLC_Majadas_clipped_grid.rio.write_crs(crs_ET)
nodata_value = CLC_Majadas_clipped_grid.Code_18.rio.nodata

CLC_Majadas_clipped_grid['Code_18']= CLC_Majadas_clipped_grid.Code_18.where(CLC_Majadas_clipped_grid.Code_18 != nodata_value, -9999)
CLC_Majadas_clipped_grid.to_netcdf('../prepro/CLCover_Majadas.netcdf')


# CLC_Majadas_clipped_grid_test = xr.open_dataset('../prepro/CLCover_Majadas.netcdf')
# CLC_Majadas_clipped_grid_test.Code_18.values

#%%
code18_xr = CLC_Majadas_clipped_grid.Code_18
code18_xr.where(code18_xr!=code18_xr.rio.nodata).plot(cmap='Paired')

#%% 
masked_code18_xr = code18_xr.where(code18_xr != code18_xr.rio.nodata)

unique_labels = np.unique(Code_18_string.values)[:-1]
mapped_values = [clc_codes.get(value, 'Unknown') for value in unique_labels]
unique_values = np.unique(code18_xr.values[~np.isnan(code18_xr.values)])

#%%

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
                        sharex=True,
                        sharey=True
                        )
axs = axs.flatten()
CLC_clipped.plot('Code_18',ax=axs[0])
# plt.figure(figsize=(12, 8))

# Create a color map with a unique color for each unique value
cmap = plt.get_cmap('Paired', len(unique_values))  # Choose a colormap and set the number of colors
norm = mcolors.BoundaryNorm(boundaries=np.append(unique_values, 
                                                 unique_values[-1] + 1), 
                            ncolors=len(unique_values)
                            )
im = masked_code18_xr.plot.imshow(cmap=cmap, 
                                  norm=norm, 
                                  add_colorbar=False,
                                  ax=axs[1]
                                  )  # Apply custom colormap

cbar = plt.colorbar(im, ticks=unique_values[1:])
cbar.set_label('Land Use Categories')
cbar.set_ticklabels(mapped_values)


majadas_aoi.boundary.plot(ax=axs[0],color='r')
majadas_POIs.plot(ax=axs[0],color='r')
majadas_aoi.boundary.plot(ax=axs[1],color='r')

# Create a custom legend
legend_labels = {i: category_mapping[i] for i in range(len(categories))}
legend_patches = [plt.Line2D([0], [0], marker='o', color='w', label=legend_labels[i],
                             markerfacecolor=cmap(i), markersize=10) for i in range(len(categories))]

# axs[1].legend(handles=legend_patches, title="Land Cover Types", loc="lower right", fontsize='small', title_fontsize='medium')

# Show the plot
# axs[1].set_title("Land Cover Map - Code_18")

Majadas_satellite.plot.imshow(rgb='band', ax=axs[2])
majadas_aoi.boundary.plot(ax=axs[2],color='r')
# providers = cx.providers.flatten()

try:
    ETp_ds.isel(band=0)['ETp'].isel(time=0).plot.imshow(ax=axs[3])
except:
    pass
majadas_aoi.boundary.plot(ax=axs[3],color='r')

# cx.add_basemap(ax=axs[1],
#                source=cx.providers.Esri.WorldImagery
#                )


fig.savefig(figPath/'LandCoverRaster_Majadas.png',dpi=300)



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
# clipped_DTM_rxr = DTM_rxr.rio.clip_box(
#                                       bbox,
#                                       crs=majadas_aoi.crs,
#                                     )  


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
# ss

# utils.clip_rioxarray(ET_filelist,
#                      ET_0_filelist,
#                      rain_filelist,
#                      majadas_aoi
#                      )

#%% Read CLIPPED and save as NETCDF
 # ! Majadas grids: S3/Meteo = E030N006T6 and S2/Landast = X0033_Y0044
# S3/Meteo EPGS CRS EPSG:27704 - WGS 84 / Equi7 Europe - Projected

# utils.export_tif2netcdf()

#%%

# ETa = ETa_ds.to_dataarray().isel(variable=0,band=0).sortby('time')
ETp = ETp_ds.to_dataarray().isel(variable=0,band=0).sortby('time')
RAIN = RAIN_ds.to_dataarray().isel(variable=0,band=0).sortby('time')
print('Errrrrorrr in rain evaluation in the input!')
RAIN = RAIN.where((RAIN <= 300) & (RAIN > 0), other=0)

# ETa = ETa_ds.to_dataarray().isel(variable=0,band=0).sortby('time')

ds_analysis_EO = ETa_ds.isel(band=0)
# da.drop_vars('x')

ds_analysis_EO['ETp'] = ETp
ds_analysis_EO['RAIN'] = RAIN
CLC_Majadas_clipped_grid_no_spatial_ref = CLC_Majadas_clipped_grid.drop_vars('spatial_ref', errors='ignore')
ds_analysis_EO['CLC'] = CLC_Majadas_clipped_grid_no_spatial_ref.Code_18
ds_analysis_EO.to_netcdf('../prepro/ds_analysis_EO.netcdf')
ds_analysis_EO = ds_analysis_EO.sortby('time')

nulltimeETa = np.where(ds_analysis_EO.ETa.isel(x=0,y=0).isnull())[0]
valid_mask = ~ds_analysis_EO.time.isin(ds_analysis_EO.time[nulltimeETa])
ds_analysis_EO = ds_analysis_EO.isel(time=valid_mask)


# ds_analysis_EO = xr.open_dataset('../prepro/ds_analysis_EO.netcdf')

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

from scipy.signal import butter, filtfilt


# Define the low-pass filter function using Butterworth
def lowpass_filter(da, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    # Apply the filter to the data
    filtered_data = filtfilt(b, a, da)
    return xr.DataArray(filtered_data, dims=da.dims, coords=da.coords, name=f"{da.name}_lowpass")

# Parameters for the lowpass filter
cutoff_frequency = 1  # Cutoff frequency (cycles per unit of time)
sampling_frequency = 10  # Sampling frequency (data points per unit of time)

# np.where(ETa_poi.isnull())
# np.where(ETp_poi.isnull())
# np.where(RAIN.isnull())

# ETp_poi[1461]

fig, ax = plt.subplots()
for i, ppc in enumerate(POIs_coords):
    ETa_poi = ds_analysis_EO['ETa'].sel(x=ppc[0],y=ppc[1], 
                                        method="nearest")
    # Apply the low-pass filter to the DataArray
    ETa_poi_lowpass = lowpass_filter(ETa_poi, 
                                       cutoff=cutoff_frequency, 
                                       fs=sampling_frequency
                                       )
    ETa_poi_detrended = ETa_poi - ETa_poi_lowpass 
    ETa_poi_detrended.plot(ax=ax)


# ds_analysis_EO['ETa_detrended']

# trend.plot()


#%% Check variations on POI

labels_POIs = ['Lake','Intensive Irrigation','Tree-Grass', 'Agricutural fields']

fig, ax = plt.subplots()
for i, ppc in enumerate(POIs_coords):
    ETa_poi = ds_analysis_EO['ETa'].sel(x=ppc[0],y=ppc[1], 
                                        method="nearest")
    ax.plot(ETa_poi.time.values,ETa_poi,label=labels_POIs[i])
ax.legend()
ax.set_xlabel('Datetime')
ax.set_ylabel('ETa (mm/day)')
fig.savefig(figPath/'ETa_serie_Majadas.png', dpi=300)



fig, ax = plt.subplots()
for i, ppc in enumerate(POIs_coords):
    RAIN_poi = ds_analysis_EO['RAIN'].sel(x=ppc[0],y=ppc[1],
                                          method="nearest")
    ax.scatter(RAIN_poi.time,RAIN_poi.values,label=labels_POIs[i])
ax.legend()
ax.set_xlabel('Datetime')
ax.set_ylabel('Rain (mm)')

fig.savefig(figPath/'rain_serie_Majadas.png', dpi=300)


fig, ax = plt.subplots()
for i, ppc in enumerate(POIs_coords):
    ETap_poi = ds_analysis_EO['ratio_ETap_local'].sel(x=ppc[0],y=ppc[1], 
                                                      method="nearest")
    ax.scatter(ETap_poi.time,ETap_poi.values,label=labels_POIs[i])
ax.legend()
ax.set_xlabel('Datetime')
ax.set_ylabel('ETap local (mm/day)')

fig.savefig(figPath/'ETap_local_serie_Majadas.png', dpi=300)

#%%

import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# Create the main figure and axis
fig, ax = plt.subplots()

# Plot the main data series
for i, ppc in enumerate(POIs_coords):
    ETp_poi = ds_analysis_EO['ETp'].sel(x=ppc[0], y=ppc[1], method="nearest")
    ax.plot(ETp_poi.time.values, ETp_poi, label=labels_POIs[i])

# Set labels and legend for the main plot
ax.legend()
ax.set_xlabel('Datetime')
ax.set_ylabel('ETp (mm/day)')

# Define the time range for the inset (6 months from 03/2022 to 09/2022)
inset_start = pd.Timestamp('2020-05-01')
inset_end = pd.Timestamp('2020-10-01')

# Create an inset axis within the main plot
ax_inset = inset_axes(ax, width="40%", height="40%", loc='upper right')

# Plot the data within the specified time range in the inset plot
for i, ppc in enumerate(POIs_coords):
    ETp_poi = ds_analysis_EO['ETp'].sel(x=ppc[0], y=ppc[1], method="nearest")
    ax_inset.plot(ETp_poi.time.values, ETp_poi, label=labels_POIs[i])

# Set the limits for the inset to zoom in on the specified time range
ax_inset.set_xlim(inset_start, inset_end)
ax_inset.set_ylim(ETp_poi.sel(time=slice(inset_start, inset_end)).min(), 
                  ETp_poi.sel(time=slice(inset_start, inset_end)).max())

# Optionally, remove the x and y labels for the inset for clarity
# ax_inset.set_xticks([])
# ax_inset.set_yticks([])

# Indicate the zoomed area on the main plot
mark_inset(ax, ax_inset, loc1=2, loc2=4, fc="none", ec="0.5")

# Save the figure with the inset plot and zoom indication
fig.savefig(figPath/'ETp_serie_Majadas.png', dpi=300)
plt.show()



#%%


# Define the mosaic layout
mosaic_layout = """
                a
                b
                b
                b
                """
fig, ax = plt.subplot_mosaic(mosaic_layout,
                             sharex=True,
                             figsize=(8,4)
                             )

# Plot the first dataset on the primary y-axis
# ax['b'].plot( )  
for i, ppc in enumerate(POIs_coords):
    ETap_poi = ds_analysis_EO['ratio_ETap_local'].sel(x=ppc[0],y=ppc[1], 
                                                      method="nearest")
    ax['b'].scatter(ETap_poi.time,ETap_poi.values,label=labels_POIs[i])
ax['b'].legend()
ax['b'].set_xlabel('Datetime')
ax['b'].set_ylabel('ETap local (mm/day)')


for i, ppc in enumerate(POIs_coords):
    RAIN_poi = ds_analysis_EO['RAIN'].sel(x=ppc[0],y=ppc[1],
                                          method="nearest",
                                          )
    ax['a'].scatter(RAIN_poi.time,RAIN_poi.values,label=labels_POIs[i])
ax['a'].legend()
ax['a'].set_xlabel('Datetime')
ax['a'].invert_yaxis()
ax['a'].set_ylabel('rain (mm)', color='b')

plt.tight_layout()
plt.savefig(figPath/'saturation_simu_Majadas.png',
            dpi=300
            )


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


