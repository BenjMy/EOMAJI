#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 16:17:07 2024
"""
import numpy as np
import xarray as xr
import geopandas as gpd
import rioxarray as rio
import pandas as pd

#%%

# SMC field sensors position
# --------------------------

def import_SWC_pos(path='../data/Majadas_coord_SWC_sensors_Benjamin.csv'):
    '''
    Import SWC content locations Majadas de Tietar
    '''
    coord_SWC_CT = pd.read_csv(path)
    crs = 'EPSG:4326'
    return coord_SWC_CT, crs

# Read SMC field sensors
# ----------------------

def get_SMC_sensors_dataset():
    return 

def get_LandCoverMap():
    # get Corine Land Cover map for Majadas 
    pass

def get_crs_ET(path='/run/media/z0272571a/SENET/iberia_daily/E030N006T6/20190205_LEVEL4_300M_ET_0-gf.tif'):
    return rio.open_rasterio(path).rio.crs
    
def get_Majadas_aoi(crs=None):
    if crs is None:
        crs = get_crs_ET()
    majadas_aoi = gpd.read_file('../data/AOI/majadas_aoi.geojson')
    majadas_aoi.to_crs(crs, inplace=True)
    return majadas_aoi

def get_Majadas_POIs(crs=None):
    if crs is None:
        crs = get_crs_ET()
    majadas_POIs = gpd.read_file('../data/AOI/POI_Majadas.geojson')
    majadas_POIs.to_crs(crs, inplace=True)
    multipoint_geom = majadas_POIs.geometry.iloc[0]
    POIs_coords = np.array([point.coords[0] for point in multipoint_geom.geoms])
    return majadas_POIs, POIs_coords


def xarraytoDEM_pad(data_array):
    # Get the resolution (pixel size) directly from the DataArray's transform
    # Get the Affine transform
    transform = data_array.rio.transform()
    
    # Extract pixel size from the transform
    pixel_size_x = transform.a  # Pixel width (x-direction)
    pixel_size_y = -transform.e  # Pixel height (y-direction, note the negative sign for y)
    

    # Define padding in pixels
    pad_pixels_y = 1  # Padding in y-direction (top and bottom)
    pad_pixels_x = 1  # Padding in x-direction (left and right)
    
    # Calculate padding in meters (or coordinate units)
    pad_m_y = pad_pixels_y * (pixel_size_y / 2)  # Padding in y-direction
    pad_m_x = pad_pixels_x * (pixel_size_x / 2)  # Padding in x-direction
    
    # Apply padding using numpy.pad
    pad_width = ((0, 0), (pad_pixels_y, 0), (0, pad_pixels_x))  # (time, y, x)
    padded_array_np = np.pad(data_array.values, 
                             pad_width, 
                             mode='edge', 
                             # constant_values=np.nan
                             )
    
    # Create a new xarray.DataArray with the padded data
    padded_data_array = xr.DataArray(
        padded_array_np,
        dims=['time', 'y', 'x'],
        coords={
            'time': data_array.time,
            'y': np.concatenate([data_array.y.values - pad_m_y, [data_array.y.values[-1] + pad_m_y]]),
            'x': np.concatenate([data_array.x.values - pad_m_x, [data_array.x.values[-1] + pad_m_x]])
        },
        attrs=data_array.attrs  # Preserve metadata
    )
    return padded_data_array
