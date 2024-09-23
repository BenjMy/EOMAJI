#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 17:43:15 2024
"""

import cdsapi
import xarray as xr
import rioxarray as rio
from pathlib import Path
# import Majadas_utils
import contextily as cx

dataPath = Path('../data/Spain/Spain_ETp_Copernicus_CDS/')

#%%
# LCmap_Majadas = Majadas_utils.get_LandCoverMap()
# LCmap_Majadas.crs
# LCmap_Majadas.plot()

#%% Crop productivity and evapotranspiration indicators from 2000 to present derived from satellite observations
# https://cds-beta.climate.copernicus.eu/datasets/sis-agroproductivity-indicators?tab=documentation
# dataset = "sis-agroproductivity-indicators"
# request = {
#     'product_family': ['evapotranspiration_indicators'],
#     'variable': ['actual_evaporation', 'potential_evaporation'],
#     'year': '2018',
#     'month': ['01', '02', '03', '04', '05', '06', '07', '08', '09'],
#     'day': ['01', '11', '21']
# }
# client = cdsapi.Client()
# client.retrieve(dataset, request).download()
# ET_C3S = rio.open_rasterio(dataPath + 'ET_C3S-glob-agric_GLS_CDS_EO_dek_20180101-20180110_v1.nc')
# ET_C3S.ACTUAL_ET.isel(time=0).plot.imshow()

#%%

import cdsapi

dataset = "reanalysis-era5-single-levels"
request = {
    'product_type': ['reanalysis'],
    'variable': ['2m_temperature', 
                 'total_precipitation', 
                 'evaporation', 
                 'potential_evaporation', 
                 'soil_type', 
                 # 'volumetric_soil_water_layer_1'
                 ],
    'year': ['2022', '2023'],
    'month': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'],
    'day': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31'],
    'time': ['00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00', '08:00', '09:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00', '16:00', '17:00', '18:00', '19:00', '20:00', '21:00', '22:00', '23:00'],
    'data_format': 'netcdf',
    'download_format': 'zip',
    'area': [43.82, -9.37, 35.95, 3.39]
}

client = cdsapi.Client()
client.retrieve(dataset, request).download()

#%%

# import cdsapi

# dataset = "reanalysis-era5-single-levels"
# request = {
#     'product_type': ['reanalysis'],
#     'variable': ['evaporation', 'potential_evaporation'],
#     'year': ['2024'],
#     'month': ['01', '02', '03', '04', '05', '06', '07', '08', '09'],
#     'day': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31'],
#     'time': ['00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00', '08:00', '09:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00', '16:00', '17:00', '18:00', '19:00', '20:00', '21:00', '22:00', '23:00'],
#     'data_format': 'netcdf',
#     'download_format': 'zip',
#     'area': [43.82, -9.37, 35.95, 3.39]
# }

# client = cdsapi.Client()
# client.retrieve(dataset, request).download()


#%%
# climate_ds = rio.open_rasterio(dataPath / 'data_stream-oper.nc',
#                                )

# climate_ds = xr.open_dataset(dataPath / 'data_stream-oper.nc',
#                                )

# climate_ds = climate_ds.rio.write_crs("EPSG:4326")

# # ds = ds.rio.reproject(LCmap_Majadas.crs)

# import matplotlib.pyplot as plt

# fig, ax = plt.subplots()
# climate_ds.pev.isel(valid_time=0).plot.imshow(ax=ax,
#                                               add_colorbar=False
#                                               )
# cx.add_basemap(ax, 
#                crs=climate_ds.rio.crs,
#                )
# climate_ds

# # Extract ETp (Evapotranspiration) and Rain variables
# ETp = climate_ds['pev']
# # Rain = climate_ds['Rain']

#%%

# climate_ds_test = rio.open_rasterio(dataPath / 'ETp/PET_C3S-glob-agric_GLS_CDS_EO_dek_20180911-20180920_v1.nc')

# climate_ds = xr.open_dataset(dataPath / 'ETp/PET_C3S-glob-agric_GLS_CDS_EO_dek_20180911-20180920_v1.nc'
#                                )



# fig, ax = plt.subplots()
# climate_ds_test.POTENTIAL_ET.isel(time=0).plot.imshow(ax=ax,
#                                               add_colorbar=False
#                                               )
# cx.add_basemap(ax, 
#                crs=climate_ds.rio.crs,
#                )
# climate_ds


#%% ERA5 hourly data on single levels from 1940 to present

# import cdsapi

# dataset = "reanalysis-era5-single-levels"
# request = {
#     'product_type': ['reanalysis'],
#     'variable': ['total_precipitation', 
#                  'evaporation', 
#                  'potential_evaporation', 
#                  'soil_type', 
#                  'volumetric_soil_water_layer_1'
#                  ],
#     'year': [
#             '2020', 
#              '2021', 
#              '2022', 
#              '2023', 
#              '2024'
#              ],
#     'month': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'],
#     'day': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31'],
#     'time': ['00:00'],
#     'data_format': 'netcdf',
#     'download_format': 'zip',
#     'area': [43.7, -7.6, 36, -5.6]
# }

# client = cdsapi.Client()
# client.retrieve(dataset, request).download()
