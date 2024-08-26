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
from shapely.geometry import Point
import utils
from pathlib import Path

#%%
def get_AOI_POI_Majadas(crs_ET):
    
    # AOI define in EOMAJI 
    # -------------------------------------------------------------------------
    majadas_aoi = get_Majadas_aoi()

    # POI picked from google earth
    # -------------------------------------------------------------------------
    majadas_POIs, POIs_coords = get_Majadas_POIs()
    labels_POIs = ['Lake',
                   'Intensive Irrigation',
                   'Tree-Grass', 
                   'Agricutural fields'
                   ]
    # soil water content sensors
    # -------------------------------------------------------------------------
    coord_SWC_CT, gdf_SWC_CT = get_SWC_pos(
                                            target_crs=crs_ET
                                            )
    gdf_SWC_CT['POI/AOI'] = 'SWC sensor'

    # Corinne Land cover dataset
    # -------------------------------------------------------------------------
    clc_codes = utils.get_CLC_code_def()
    CLC_path = Path('../data/Copernicus_95732/U2018_CLC2018_V2020_20u1.shp/U2018_CLC2018_V2020_20u1.shp')
    CLC_Majadas = gpd.read_file(CLC_path)
    CLC_Majadas = CLC_Majadas.to_crs(crs_ET)
    
    CLC_clipped = gpd.clip(CLC_Majadas, 
                            mask=majadas_aoi.total_bounds
                            # mask= [
                            #         majadas_aoi.bounds['minx'].values[0],
                            #         majadas_aoi.bounds['miny'].values[0],
                            #         majadas_aoi.bounds['maxx'].values[0] #+300,
                            #         majadas_aoi.bounds['maxy'].values[0] #+300,    
                            #         ]
                            )
    mask_agroforestry = CLC_clipped['Code_18'] == '244'
    mask_irrigated = CLC_clipped['Code_18'] == '212'
    agroforestry_landcover = CLC_clipped[mask_agroforestry]
    irrigated_landcover = CLC_clipped[mask_irrigated]
    agroforestry_landcover.to_crs(gdf_SWC_CT.crs, inplace=True)
    irrigated_landcover.to_crs(gdf_SWC_CT.crs, inplace=True)
    agroforestry_landcover['POI/AOI'] = 'agroforestry'
    irrigated_landcover['POI/AOI'] = 'irrigated'
    
    # Create geodataframe
    # -------------------------------------------------------------------------
    gdf_AOI_POI_Majadas = gpd.GeoDataFrame(
                                        labels_POIs, 
                                        geometry=gpd.points_from_xy(POIs_coords[:,0], 
                                                                    POIs_coords[:,1]), 
                                        crs=gdf_SWC_CT.crs
                                        )
    gdf_AOI_POI_Majadas.rename({0:'id'})
    gdf_AOI_POI_Majadas = gdf_AOI_POI_Majadas.rename({0:'POI/AOI'},axis=1)
    
    gdf_AOI_POI_Majadas = pd.concat([gdf_AOI_POI_Majadas,
                                     gdf_SWC_CT,
                                     agroforestry_landcover,
                                     irrigated_landcover
                                     ],
                                    ignore_index=True
                                    )
    
    print('add towers water footprint areas')
    return gdf_AOI_POI_Majadas


# SMC field sensors position
# --------------------------

def get_SWC_pos(path='../data/TDR/Majadas_coord_SWC_sensors_Benjamin.csv',
                   target_crs=None):
    '''
    Import SWC content locations Majadas de Tietar
    '''
    coord_SWC_CT = pd.read_csv(path)
    crs = 'EPSG:4326'
    
    col2sel = ['SWC sensor'] +  list(coord_SWC_CT.columns[coord_SWC_CT.columns.str.contains('wgs84')])
    coord_SWC_CT_WGS84 = coord_SWC_CT[col2sel]

    geometry = [Point(lon, lat) for lon, lat in zip(coord_SWC_CT['longetrs89'], 
                                                    coord_SWC_CT['latwgs84'])]
    # Create GeoDataFrame
    gdf_SWC_CT = gpd.GeoDataFrame(coord_SWC_CT, geometry=geometry)

    # Set the CRS (Coordinate Reference System)
    # Assuming WGS84 for lat/lon coordinates
    gdf_SWC_CT.set_crs(epsg=4326, inplace=True)
    
    if target_crs is not None:
        gdf_SWC_CT = gdf_SWC_CT.to_crs(crs=target_crs)

    # fig, ax = plt.subplots()
    # ETp_ds.isel(band=0,time=0).ETp.plot.imshow(ax=ax)
    # gdf_SWC_CT.plot(ax=ax,color='r')

    return coord_SWC_CT, gdf_SWC_CT

# Read SMC field sensors
# ----------------------
def get_SWC_data(path='../data/TDR/LMA_Meteo_2022-2023_Benjamin.csv'):
    TDR = pd.read_csv(path)
    TDR.set_index('rDate',inplace=True)
    TDR.index = pd.to_datetime(TDR.index, format='%d/%m/%Y %H:%M')
    TDR_SWC_columns = TDR.columns[TDR.columns.str.startswith('SWC_')]
    TDR_SWC = TDR[TDR_SWC_columns]
    # TDR_SWC = TDR_SWC.T
    
    # rootName = 'SWC_2014'
    profilName = ['S','NW','SE','NE']
    depths = [10,20,40,50,100]
    bareSoil = ['NW','S']
    return TDR_SWC, depths


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
