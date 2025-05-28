#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 16:17:07 2024
"""
import numpy as np
import xarray as xr
import geopandas as gpd
import rioxarray as rxr
import pandas as pd
from shapely.geometry import Point
from centum import utils
# import utils
from pathlib import Path
from shapely.geometry import box
from shapely.geometry import mapping
from scipy import stats
from datetime import datetime

#%%

def read_prepo_EO_datasets(fieldsite='Majadas',
                           AOI='Buffer_5000',
                           crs=None,
                           rootPath = Path('../../')
                           ):
            
    ETa_ds = xr.open_dataset(rootPath/f'prepro/Majadas/{AOI}/ETa_{fieldsite}.netcdf',
                             # engine='scipy'
                             )
    ETa_ds = ETa_ds.rename({"__xarray_dataarray_variable__": "ETa"})
    ETp_ds = xr.open_dataset(rootPath/f'prepro/Majadas/{AOI}/ETp_{fieldsite}.netcdf')
    ETp_ds = ETp_ds.rename({"__xarray_dataarray_variable__": "ETp"})
    RAIN_ds = xr.open_dataset(rootPath/f'prepro/Majadas/{AOI}/RAIN_{fieldsite}.netcdf')
    RAIN_ds = RAIN_ds.rename({"__xarray_dataarray_variable__": "RAIN"})
    CLC_ds = xr.open_dataset(rootPath/f'prepro/Majadas/{AOI}/CLCover_{fieldsite}.netcdf')

    ds_analysis_EO = ETa_ds.drop_vars('spatial_ref', errors='ignore').isel(band=0)
    ds_analysis_EO['ETp'] = ETp_ds.to_dataarray().isel(band=0).sel(variable='ETp')
    ds_analysis_EO['RAIN'] = RAIN_ds.to_dataarray().isel(band=0).sel(variable='RAIN')

    CLC_ds = CLC_ds.drop_vars('spatial_ref', errors='ignore')
    ds_analysis_EO['CLC_code18'] = CLC_ds.Code_18
    ds_analysis_EO = ds_analysis_EO.sortby('time')
    
    ds_analysis_EO = ds_analysis_EO.rio.write_crs(crs)
    # ds_analysis_EO.ETa
    
    nulltimeETa = np.where(ds_analysis_EO.ETa.isnull().all())[0]
    valid_mask = ~ds_analysis_EO.time.isin(ds_analysis_EO.time[nulltimeETa])
    
    if len(nulltimeETa)>1:
        print('times with null ETa values!!')
    ds_analysis_EO = ds_analysis_EO.isel(time=valid_mask)
    
    print('Errrrrorrr in rain evaluation in the input!')
    # data_array = data_array.where((data_array <= 300) & (data_array > 0), other=np.nan)
    # ds_analysis_EO['RAIN'] = ds_analysis_EO['RAIN'].where((ds_analysis_EO['RAIN'] <= 300) & (ds_analysis_EO['RAIN'] > 0), 
    #                                                       other=0)
    
    ds_analysis_EO['RAIN'] = ds_analysis_EO['RAIN'].where(
                                    (ds_analysis_EO['RAIN'] <= 300) & (ds_analysis_EO['RAIN'] > 0) | ds_analysis_EO['RAIN'].isnull(), 
                                    other=0
                                )
    # Determine the overlapping time range
    start_time = max(ds_analysis_EO['RAIN'].time.min(), ds_analysis_EO['ETp'].time.min())
    end_time = min(ds_analysis_EO['RAIN'].time.max(), ds_analysis_EO['ETp'].time.max())

    # Create a mask for the common time range
    mask_time = (ds_analysis_EO['RAIN'].time >= start_time) & (ds_analysis_EO['RAIN'].time <= end_time)
    mask_time2 = (ds_analysis_EO['ETp'].time >= start_time) & (ds_analysis_EO['ETp'].time <= end_time)

    # Filter the DataArrays using the mask
    ds_analysis_EO = ds_analysis_EO.sel(time=mask_time)
    ds_analysis_EO = ds_analysis_EO.sel(time=mask_time2)

    return ds_analysis_EO
    
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
    
    CLC_Majadas = get_LandCoverMap()
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
    
    print('add towers water footprint areas?!')
    return gdf_AOI_POI_Majadas


# SMC field sensors position
# --------------------------

def get_SWC_pos(
                # path='../../data/Spain/Majadas/TDR/Majadas_coord_SWC_sensors_Benjamin.csv',
                path='/run/media/z0272571a/LVM_16Tb/Ben/EOMAJI/data/Spain/Majadas/TDR/Majadas_coord_SWC_sensors_Benjamin.csv',
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
def get_SWC_data(
        # path='../../data/Spain/Majadas/TDR/LMA_Meteo_2022-2023_Benjamin.csv'
        path='/run/media/z0272571a/LVM_16Tb/Ben/EOMAJI/data/Spain/Majadas/TDR/LMA_Meteo_2022-2023_Benjamin.csv'
        
        ):   
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


def get_LandCoverMap(crs=None):
    if crs is None:
        crs = get_crs_ET()
    # get Corine Land Cover map for Majadas 
    # CLC_path = Path('../../data/Spain/Copernicus_95732/U2018_CLC2018_V2020_20u1.shp/U2018_CLC2018_V2020_20u1.shp')
    CLC_path = Path('/run/media/z0272571a/LVM_16Tb/Ben/EOMAJI/data/Spain/Copernicus_95732/U2018_CLC2018_V2020_20u1.shp/U2018_CLC2018_V2020_20u1.shp')
    CLC_Majadas = gpd.read_file(CLC_path)
    CLC_Majadas.to_crs(crs, inplace=True)

    return CLC_Majadas

def get_crs_ET(path='/run/media/z0272571a/SENET/iberia_daily/E030N006T6/20190205_LEVEL4_300M_ET_0-gf.tif'):
    return rxr.open_rasterio(path).rio.crs
    
def get_Majadas_aoi(crs=None,buffer=3000):
    if crs is None:
        crs = get_crs_ET()
    majadas_aoi = gpd.read_file('../../data/AOI/majadas_aoi.geojson')
    majadas_aoi.to_crs(crs, inplace=True)

    # buffered_aoi = majadas_aoi.geometry.buffer(buffer)  # Adjust the buffer distance as needed
    # majadas_aoi = gpd.GeoDataFrame({'name': ['Majadas de Tietar Larger AOI'], 
    #                                     'geometry': buffered_aoi}
    #                                   )
    if buffer>0:
    # Get the bounding box of the geometry and apply a buffer to enlarge it
    # buffer_distance = 5000  # Adjust this value as needed
        minx, miny, maxx, maxy = majadas_aoi.total_bounds
        buffered_box = box(minx - buffer, 
                           miny - buffer, 
                           maxx + buffer, 
                           maxy + buffer)
        
        # Create a new GeoDataFrame for the rectangular AOI
        majadas_aoi = gpd.GeoDataFrame({'name': ['Majadas de Tietar Larger AOI'], 
                                        'geometry': [buffered_box]}, 
                                       crs=majadas_aoi.crs
                                       )


    # majadas_aoi = gpd.read_file('../data/AOI/test_DEM_CATHY-polygon.shp')
    # majadas_aoi = gpd.read_file('../data/Spain/GIS_catchment_majadas/BassinH2_Majadas_corrected.shp')
    return majadas_aoi

def get_Majadas_POIs(crs=None):
    if crs is None:
        crs = get_crs_ET()
    majadas_POIs = gpd.read_file('../../data/AOI/POI_Majadas.geojson')
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


def get_Majadas_root_map_from_CLC(xrx_grid_target,
                                  xrx_CLC_to_map,
                                  crs_target):
    
    xrx_CLC_to_map = xrx_CLC_to_map.rio.write_crs(crs_target)
    xrx_grid_target = xrx_grid_target.rio.write_crs(crs_target)
    reprojected_CLC_Majadas = xrx_CLC_to_map.rio.reproject_match(xrx_grid_target,
                                                                )
    CLC_values_unique = np.unique(xrx_CLC_to_map.Code_CLC.values)
    code18_str_rootmap_indice = [ (cci,i+1) for i, cci in enumerate(CLC_values_unique[:-1])]
    replacement_dict = dict(code18_str_rootmap_indice)
    replacement_dict[np.nan] = 0
    mapped_data = np.zeros(np.shape(reprojected_CLC_Majadas.Code_CLC))
    i = 1
    for key, value in replacement_dict.items():
        mapped_data[reprojected_CLC_Majadas.Code_CLC.values == key] = i
        if np.sum(reprojected_CLC_Majadas.Code_CLC.values == key) > 1:
            i += 1 
    
    return reprojected_CLC_Majadas, mapped_data



def clip_ET_withLandCover(LCnames,
                          gdf_AOI,
                          ETxr,
                          ETname = 'ACT. ETRA',
                          crs_ET = None,
                          axs = None
                          ):
    
    for axi, lcn in zip(axs,LCnames):
        CLC_mask = gdf_AOI.set_index('POI/AOI').loc[lcn].geometry
        ETxr = ETxr.rio.write_crs(crs_ET)
        mask_ETA = ETxr[ETname].rio.clip(CLC_mask.apply(mapping), 
                                 crs_ET, 
                                 drop=False
                                 )
    
        ETxr[lcn + '_CLCmask'] = mask_ETA
        ETxr.isel(time=0)[lcn + '_CLCmask'].plot.imshow(ax=axi,
                                                        )
        axi.set_title(lcn)
        axi.set_aspect('equal')
        
    return ETxr

def perf_linreg(x,y): 
    # Perform linear regression using scipy.stats.linregress
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, 
                                                                   y
                                                                   )
    y_pred = slope * x.values + intercept
    r2 = r_value**2  # Compute R^2 value
    
    return y_pred, r2



def clip_rioxarray(ET_filelist,
                   ET_0_filelist,
                   rain_filelist,
                   majadas_aoi):
        
    for m in ET_filelist:
        print()
        etai = rxr.open_rasterio(m)
        # clipped_etai = etai.rio.clip_box(
        #                                   minx=majadas_aoi.bounds['minx'],
        #                                   miny=majadas_aoi.bounds['miny'],
        #                                   maxx=majadas_aoi.bounds['maxx'],
        #                                   maxy=majadas_aoi.bounds['maxy'],
        #                                   crs=majadas_aoi.crs,
        #                                 )   
        
        clipped_etai = etai.rio.clip(
                                        majadas_aoi.geometry.values,
                                        crs=majadas_aoi.crs,
                                        )  
        
        clipped_etai['time']=extract_filedate(m)
        clipped_etai.rio.to_raster('../../prepro/Majadas/' + m.name)
        
    
    for m in ET_0_filelist:
        etrefi = rxr.open_rasterio(m)
        # clipped_etrefi = etrefi.rio.clip_box(
        #                                       minx=majadas_aoi.bounds['minx'],
        #                                       miny=majadas_aoi.bounds['miny'],
        #                                       maxx=majadas_aoi.bounds['maxx'],
        #                                       maxy=majadas_aoi.bounds['maxy'],
        #                                     crs=majadas_aoi.crs,
        #                                     )   
        clipped_etrefi = etrefi.rio.clip(
                                            majadas_aoi.geometry.values,
                                            crs=majadas_aoi.crs,
                                            )   
        clipped_etrefi['time']=extract_filedate(m)
        clipped_etrefi.rio.to_raster('../../prepro/Majadas/' + m.name)
        
    for m in rain_filelist:
        raini = rxr.open_rasterio(m)
        # clipped_raini = raini.rio.clip_box(
        #                                       minx=majadas_aoi.bounds['minx'],
        #                                       miny=majadas_aoi.bounds['miny'],
        #                                       maxx=majadas_aoi.bounds['maxx'],
        #                                       maxy=majadas_aoi.bounds['maxy'],
        #                                     crs=majadas_aoi.crs,
        #                                     )   
        clipped_raini = raini.rio.clip(
                                            majadas_aoi.geometry.values,
                                            crs=majadas_aoi.crs,
                                            )   
    
        clipped_raini['time']=extract_filedate(m)
        clipped_raini.rio.to_raster('../../prepro/Majadas/' + m.name)
        
    return clipped_etai, clipped_etrefi, clipped_raini


def extract_filedate(file_path):
    file_name = file_path.name
    date_str = file_name.split('_')[0]
    return datetime.strptime(date_str, '%Y%m%d')


        
def export_tif2netcdf(pathTif2read='../../prepro/Majadas/',fieldsite='Majadas'):
    
    file_pattern = '*ET-gf*.tif'
    ET_clipped_filelist = list(Path(pathTif2read).glob(file_pattern))
    
    file_pattern = '*ET_0-gf*.tif'
    ET_0_clipped_filelist = list(Path(pathTif2read).glob(file_pattern))
    
    file_pattern = '*TPday*.tif'
    rain_clipped_filelist = list(Path(pathTif2read).glob(file_pattern))
    
    ETa_l = []
    ETa_dates = []
    for m in ET_clipped_filelist:
        ETafi = rxr.open_rasterio(m)
        ETafi['time']=extract_filedate(m)
        ETa_l.append(ETafi)
        ETa_dates.append(ETafi['time'])
    
    
    ETp_l = []
    ETp_dates = []
    for m in ET_0_clipped_filelist:
        ETpfi = rxr.open_rasterio(m)
        ETpfi['time']=extract_filedate(m)
        ETp_l.append(ETpfi)
        ETp_dates.append(ETpfi['time'])
    
    rain_l = []
    rain_dates = []
    for m in rain_clipped_filelist:
        rainfi = rxr.open_rasterio(m)
        rainfi['time']=extract_filedate(m)
        rain_l.append(rainfi)
        rain_dates.append(rainfi['time'])
    
    ETp = xr.concat(ETp_l,dim='time')
    ETp.to_netcdf(f'../../prepro/Majadas/ETp_{fieldsite}.netcdf')
    RAIN = xr.concat(rain_l,dim='time')
    RAIN.to_netcdf(f'../../prepro/Majadas/RAIN_{fieldsite}.netcdf')
    ETa = xr.concat(ETa_l,dim='time')
    ETa.to_netcdf(f'../../prepro/Majadas/ETa_{fieldsite}.netcdf')


