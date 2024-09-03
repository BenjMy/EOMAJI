#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 17:43:15 2024
"""

import cdsapi


#%% Crop productivity and evapotranspiration indicators from 2000 to present derived from satellite observations


# https://cds-beta.climate.copernicus.eu/datasets/sis-agroproductivity-indicators?tab=documentation


dataset = "sis-agroproductivity-indicators"
request = {
    'product_family': ['evapotranspiration_indicators'],
    'variable': ['actual_evaporation', 'potential_evaporation'],
    'year': '2018',
    'month': ['01', '02', '03', '04', '05', '06', '07', '08', '09'],
    'day': ['01', '11', '21']
}

client = cdsapi.Client()
client.retrieve(dataset, request).download()


#%% ERA5 hourly data on single levels from 1940 to present


# dataset = "reanalysis-era5-single-levels"
# request = {
#     'product_type': ['reanalysis'],
#     'variable': ['total_precipitation', 'evaporation', 'potential_evaporation', 'soil_type', 'volumetric_soil_water_layer_1', 'volumetric_soil_water_layer_2', 'volumetric_soil_water_layer_3', 'volumetric_soil_water_layer_4', 'type_of_high_vegetation', 'type_of_low_vegetation'],
#     'year': ['2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024'],
#     'month': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'],
#     'day': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31'],
#     'time': ['00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00', '08:00', '09:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00', '16:00', '17:00', '18:00', '19:00', '20:00', '21:00', '22:00', '23:00'],
#     'data_format': 'netcdf',
#     'download_format': 'zip'
# }

# client = cdsapi.Client()
# client.retrieve(dataset, request).download()
