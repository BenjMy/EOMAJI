#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 16:03:19 2024
"""

import scenarii2pyCATHY
import pyCATHY
import numpy as np
from pyCATHY import CATHY
from pyCATHY.plotters import cathy_plots as cplt
from pyCATHY.importers import cathy_outputs as out_CT
import pyCATHY.meshtools as msh_CT
import utils
from datetime import datetime
from pathlib import Path
import rioxarray as rxr 

import os
import pyvista as pv
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

from matplotlib.animation import FuncAnimation
from matplotlib import animation
from scipy.ndimage import label, center_of_mass

import july
from pyCATHY.cathy_utils import change_x2date
import matplotlib.dates as mdates

from aquacrop import AquaCropModel, Soil, Crop, InitialWaterContent, IrrigationManagement
from scipy.ndimage import uniform_filter


def extract_filedate(file_path):
    file_name = file_path.name
    date_str = file_name.split('_')[0]
    return datetime.strptime(date_str, '%Y%m%d')

#%% Function to build the analysis xarray dataset

def get_analysis_ds(out_with_IRR):
    ds_multiindex = out_with_IRR.reset_index()
    ds_multiindex = ds_multiindex.set_index(['time', 'x', 'y'])
    ds_multiindex = ds_multiindex.to_xarray()
    return ds_multiindex

def add_ETp2ds(ETp,ds_analysis):
    # Raster zones to nodes
    # ----------------------------------------------------------------------
    pad_width = ((0, 1), (0, 1))  # Padding only the left and top
    padded_ETp = np.pad(ETp, 
                           pad_width, 
                           mode='constant', 
                           constant_values=ETp.mean()
                           )   
    ds_analysis["ETp"] = (("time", "x", "y"), [padded_ETp]*len(ds_analysis.time))
    return ds_analysis


def apply_time_window_mean(ds_analysis,time_window=10,p=''):
    # Calculate the absolute time difference between consecutive time steps
    time_diff = np.diff(ds_analysis['time'].values)  # In units of numpy timedelta64
    
    # Convert time differences to days (assuming time is datetime64[ns])
    time_diff_days = time_diff / np.timedelta64(1, 'D')

    # Mask: Create a boolean array to flag where the time difference is <= 1 day
    time_mask = np.concatenate([[True], time_diff_days <= 1.1])  # Keep the first value True
    
    # Apply the time window rolling mean only when the time difference is <= 1 day
    # ds_analysis["ratio_ETap_local_time_avg"] = (
    #     ds_analysis["ratio_ETap_local_diff"]
    #     .where(time_mask, drop=False)  # Only consider valid time steps
    #     .rolling(time=time_window, center=True)
    #     .mean()
    # )
    
    ds_analysis[p + "_time_avg"] = (
                                    ds_analysis[p]
                                    .where(time_mask[:, np.newaxis, np.newaxis], drop=False)  # Only apply to valid time steps
                                    .rolling(time=time_window, center=True)
                                    .mean()
                                    )
    
    # ds_analysis["ratio_ETap_local"].time.where(time_mask, drop=False)
                                                    
    return ds_analysis


 
    
    
def compute_ratio_ETap_local(ds_analysis, 
                              ETa_name='ACT. ETRA', 
                              ETp_name='ETp', 
                              time_window=None
                              ):
        
    # Apply the time difference condition and calculate the absolute difference in 'ratio_ETap_local'
    ds_analysis["ratio_ETap_local"] = ds_analysis[f'{ETa_name}']/ds_analysis[f'{ETp_name}']
    ds_analysis["ratio_ETap_local_diff"] = abs(ds_analysis["ratio_ETap_local"].diff(dim='time'))
    ds_analysis = apply_time_window_mean(ds_analysis, p='ratio_ETap_local',time_window=time_window)
        
    return ds_analysis



def compute_regional_ETap(ds_analysis, 
                          ETa_name='ACT. ETRA', 
                          ETp_name='ETp', 
                          window_size_x=10,  # in km
                          stat='mean',
                        ):
    # Calculate grid resolution in km (assuming UTM or similar CRS with meters)
    x_step_km = ds_analysis.rio.resolution()[0] * 1e-3  # Grid resolution in km
    window_steps = int(window_size_x / x_step_km)  # Convert window size to number of grid steps
    
    # Choose the appropriate function for the statistic
    if stat == 'mean':
        aggregation_func = uniform_filter  # This applies a mean filter (moving window average)
    
    results = {}
    for pp in [ETa_name, ETp_name]:
        data = ds_analysis[pp]

        # Initialize a list to store the results for each time slice
        time_aggregated = []

        # Loop over time steps
        for t in range(data.sizes['time']):
            # Extract the time slice (2D data: y, x)
            data_slice = data.isel(time=t)
            
            # Apply the moving window mean using uniform_filter
            mean_data = aggregation_func(data_slice, 
                                         size=(window_steps, window_steps), 
                                         mode='reflect')

            # Create a DataArray for the current time slice to keep metadata
            mean_dataarray = xr.DataArray(mean_data, 
                                          coords=data_slice.coords, 
                                          dims=data_slice.dims)
            
            # Append the result to the list
            time_aggregated.append(mean_dataarray)
        
        # Concatenate along the time dimension to recreate the full dataset
        results[pp] = xr.concat(time_aggregated, dim='time')
    
    return results


def compute_ratio_ETap_regional(ds_analysis,
                                ETa_name='ACT. ETRA',
                                ETp_name='ETp',
                                stat = 'mean',
                                window_size_x=10, # in km,
                                time_window=None
                                ):
    #
    # calculating the change in **ETa/p** between the time on which irrigation
    # is to be detect and most recent previous time on which ET estimates are available.
    #
    if stat == 'mean':
        reg_analysis = compute_regional_ETap(
                                            ds_analysis,
                                            stat = 'mean',
                                            window_size_x=10 # in km
                                            )
        ds_analysis["ratio_ETap_regional_spatial_avg"] = reg_analysis[f'{ETa_name}']/reg_analysis[f'{ETp_name}']
        ds_analysis["ratio_ETap_regional_diff"] = abs(ds_analysis["ratio_ETap_regional_spatial_avg"].diff(dim='time'))
        ds_analysis = apply_time_window_mean(ds_analysis,p='ratio_ETap_regional_spatial_avg',
                                             time_window=time_window)
                

    return ds_analysis


def compute_bool_threshold_decision_local(ds_analysis,
                                          threshold_local=0.25,
                                          checkp='ratio_ETap_local_time_avg'
                                          ):
        # Initialize 'threshold_local' with False values
    ds_analysis["threshold_local"] = xr.DataArray(False, 
                                                  coords=ds_analysis.coords, 
                                                  dims=ds_analysis.dims
                                                     )
    # Set 'threshold_local' to True where condition is met
    checkon = ds_analysis[checkp]
    ds_analysis["threshold_local"] = xr.where(checkon > threshold_local, True, False)
    return ds_analysis

def compute_bool_threshold_decision_regional(ds_analysis,
                                             threshold_regional=0.25,
                                             checkp='ratio_ETap_regional_spatial_avg_time_avg'
                                             ):
    
    ds_analysis["threshold_regional"] = xr.DataArray(False, 
                                                  coords=ds_analysis.coords, 
                                                  dims=ds_analysis.dims
                                                     )
    checkon = ds_analysis[checkp]
    ds_analysis["threshold_regional"] = xr.where(checkon > threshold_regional, True, False)
    return ds_analysis


def define_decision_thresholds(ds_analysis,
                               threshold_local=-0.25,
                               threshold_regional=-0.25
                               ):
    '''
    
    The local and regional changes are then **compared to a number of thresholds** to try to detect if:
    - a) There is no input of water into the soil (e.g. local ETa/p does not increase above a threshold)
    - b) There is input of water into the soil but due to rainfall (e.g. increase in regional ETa/p is over a
    threshold and larger or similar to increase in local Eta/p)
    - c) There is input of water to the soil due to irrigation (e.g. increase in local ETa/p is over a
    threshold and significantly larger than increase in regional ETa/p)


    '''
    
    ds_analysis = compute_bool_threshold_decision_local(ds_analysis,
                                                        threshold=threshold_local
                                                        )
    ds_analysis = compute_bool_threshold_decision_regional(ds_analysis,
                                                           threshold=threshold_regional
                                                           )
    # compare_local_regional_ratios(ds_analysis)
    
    return ds_analysis
    
def compare_local_regional_ratios(ds_analysis):
    pass
    # differenciation between rain and irrigation events
    # --------------------------------------------------
    # compute local vs regional ETa/ETp

def compute_rolling_time_mean(ds_analysis):
    ds_analysis.rolling(time=3).mean()
    return ds_analysis
#%%

def read_outputs(simu):

    sw, sw_times = simu.read_outputs('sw')
    sw_df = pd.DataFrame(sw.T)
    
    psi = simu.read_outputs('psi')

    atmbc_df = simu.read_inputs('atmbc')
    
    
    fort777 = os.path.join(simu.workdir,
                            simu.project_name,
                            'fort.777'
                            )
    ETa = out_CT.read_fort777(fort777)
    ETa.drop_duplicates(inplace=True)
    
    # create a dictionnary
    out_data = {
                'times': sw_times,
                'sw': sw,
                'psi': psi,
                'atmbc_df': atmbc_df,   
                'ETa': ETa,
                }
    return out_data


#%%
def plot_3d_SatPre(path):

    cplt.show_vtk(
                    unit="pressure",
                    path=path,
                    savefig=True,
                    timeStep=4,
                    )
    
    cplt.show_vtk(
                    unit="saturation",
                    path=path,
                    savefig=True,
                    timeStep=4,
                    )
    
    #%%
    cplt.show_vtk_TL(
                    unit="saturation",
                    path=path,
                    savefig=True,
                    show=False,
                    # timeStep=4,
                    )
    
    #%%
    cplt.show_vtk_TL(
                    unit="saturation",
                    path=path,
                    savefig=True,
                    show=False,
                    # timeStep=4,
                    )
    
#%%

def plot_in_subplot(ax,
                    node_index,
                    out_with_IRR,
                    out_baseline,
                    prop='sw',
                    **kwargs
                    ):

    dates = None
    if 'dates' in kwargs:
        dates = kwargs.pop('dates')
        
    sw2plot_with_IRR =  out_with_IRR[prop].iloc[:,node_index].values
    sw2plot_baseline =  out_baseline[prop].iloc[:,node_index].values
    
    if dates is None:
        t = out_with_IRR['times']/86400
        y = sw2plot_with_IRR
        # tb = out_baseline['times']/86400
        yb = sw2plot_baseline
    else:
        t = dates
        y = sw2plot_with_IRR[1:]
        yb = sw2plot_baseline[1:]
        
    
    ax.plot(
            t,
            y,
            # label='Irrigated',
            marker='.',
            color='blue'
            )
    ax.plot(
            t,
            yb,
            # label='Baseline',
            marker='.',
            color='red',
            )   
    
    # ax.legend()
    ax.grid('major')
    # ax.set_title('Saturation')
    ax.set_xlabel('Time')
    ax.set_ylabel(prop)

#%%

# Convert seconds to days for x-tick labels
def seconds_to_days(seconds):
    return seconds / 86400


# def find_irr_surface_node(index,out_with_IRR,out_baseline):
    
#     ETa1d_index = np.where(out_with_IRR['ETa']['SURFACE NODE']==index[0])[0]
#     ETa1d_with_IRR = out_with_IRR['ETa']['ACT. ETRA'].iloc[ETa1d_index[1:]]
#     ETa1d_baseline = out_baseline['ETa']['ACT. ETRA'].iloc[ETa1d_index[1:]]
    
#     return ETa1d_index, ETa1d_with_IRR, ETa1d_baseline

def plot_ETa1d(
                out_with_IRR,
                out_baseline,
                node_index,
                ETp,
                dates,
                timeIrr_sec,
                axs,
               ):
    ETa1d_index = np.where(out_with_IRR['ETa']['SURFACE NODE']==node_index[0])[0]
    ETa1d_with_IRR = out_with_IRR['ETa']['ACT. ETRA'].iloc[ETa1d_index[1:]]
    ETa1d_baseline = out_baseline['ETa']['ACT. ETRA'].iloc[ETa1d_index[1:]]
    axs[0].set_xlabel('')
    axs[1].set_xlabel('')

    if dates is None:
        x = out_baseline['ETa'].time_sec.unique()[1:]/86400
    else:
        x = dates
        
    indexplot = (3)
    axs[indexplot].plot(x ,
                        ETa1d_with_IRR,
                        label='Irr',
                        color='blue',
                        marker='*'
                        )
    axs[indexplot].plot(x ,
                        ETa1d_baseline,
                        label='baseline',
                        color='red',
                        marker='*'
                        )
    
    if type(ETp) != float:
        axs[indexplot].plot(x,
                            abs(ETp), 
                            color='k', 
                            linestyle='--', 
                            label='ETp'
                            )
    else:
        axs[indexplot].axhline(y=abs(ETp), 
                                color='k', 
                                linestyle='--', 
                                label='ETp')
    axs[indexplot].legend()
    axs[indexplot].set_xlabel('time')
    axs[indexplot].set_ylabel('ETa (m/s)')

    if timeIrr_sec is not None:
        axs[indexplot].axvline(x=timeIrr_sec/86400, 
                               color='r', 
                               linestyle='--', 
                               label='Start Irr.')
        
            
def plot_1d_evol(simu,
                 node_index,
                 out_with_IRR,
                 out_baseline,
                 ETp,
                 axs,
                 **kwargs
                 ):
    
    timeIrr_sec = None
    if 'timeIrr_sec' in kwargs:
        timeIrr_sec = kwargs.pop('timeIrr_sec')
    
    dates = None 
    if 'dates' in kwargs:
        dates = kwargs.pop('dates')
        
    df_atmbc = simu.read_inputs('atmbc')
    df_atmbc['idnode'] = np.tile(np.arange(0,int(len(df_atmbc)/len(np.unique(df_atmbc.time)))),
                                 len(np.unique(df_atmbc.time))
                                 )
    mask_node = df_atmbc['idnode']==node_index[0]
    if dates is None:
        axs[0].bar(df_atmbc[mask_node].time/86400,
                   df_atmbc[mask_node].value.values
                    )
        df_atmbc_dates = None
    else:
        df_atmbc_dates = change_x2date(df_atmbc[mask_node].time, 
                                      dates[0],
                                      formatIn="%Y%m%d",
                                      formatOut="%Y-%m-%d %H:%M:%S"
                                    )
        axs[0].bar(df_atmbc_dates,
                   df_atmbc[mask_node].value.values
                    )
                
    utils.plot_in_subplot(
                            axs[1],
                            node_index,
                            out_with_IRR,
                            out_baseline,
                            prop='sw',
                            dates=df_atmbc_dates
                            )
    
    utils.plot_in_subplot(
                            axs[2],
                            node_index,
                            out_with_IRR,
                            out_baseline,
                            prop='psi',
                            dates=df_atmbc_dates
                            )
    plot_ETa1d(
                out_with_IRR,
                out_baseline,
                node_index,
                ETp,
                dates,
                timeIrr_sec,
                axs,
                )
    
    
    
 
def DEPRECATED_set_index_IN_OUT_irr_area(simu,maxDEM=1):
    index_irrArea, coords_IN_irrArea = simu.find_nearest_node([500,
                                                 500,
                                                 maxDEM]
                                                )
    index_out_irrArea, coords_OUT_irrArea = simu.find_nearest_node([10,
                                                 10,
                                                 maxDEM]
                                                )
    return index_irrArea, index_out_irrArea



def get_CLC_code_def():
    
    clc_codes = {
        "111": "Continuous urban fabric",
        "112": "Discontinuous urban fabric",
        "121": "Industrial or commercial units",
        "122": "Road and rail networks and associated land",
        "123": "Port areas",
        "124": "Airports",
        "131": "Mineral extraction sites",
        "132": "Dump sites",
        "133": "Construction sites",
        "141": "Green urban areas",
        "142": "Sport and leisure facilities",
        "211": "Non-irrigated arable land",
        "212": "Permanently irrigated land",
        "213": "Rice fields",
        "221": "Vineyards",
        "222": "Fruit trees and berry plantations",
        "223": "Olive groves",
        "231": "Pastures",
        "241": "Annual crops associated with permanent crops",
        "242": "Complex cultivation patterns",
        "243": "Land principally occupied by agriculture, with significant areas of natural vegetation",
        "244": "Agro-forestry areas",
        "311": "Broad-leaved forest",
        "312": "Coniferous forest",
        "313": "Mixed forest",
        "321": "Natural grasslands",
        "322": "Moors and heathland",
        "323": "Sclerophyllous vegetation",
        "324": "Transitional woodland-shrub",
        "331": "Beaches, dunes, sands",
        "332": "Bare rocks",
        "333": "Sparsely vegetated areas",
        "334": "Burnt areas",
        "335": "Glaciers and perpetual snow",
        "411": "Inland marshes",
        "412": "Peat bogs",
        "421": "Salt marshes",
        "422": "Salines",
        "423": "Intertidal flats",
        "511": "Water courses",
        "512": "Water bodies",
        "521": "Coastal lagoons",
        "522": "Estuaries",
        "523": "Sea and ocean"
    }
    
    return clc_codes


def CLC_2_rootdepth():
    
    # {'111': 'Continuous urban fabric',
    #  '112': 'Discontinuous urban fabric',
    #  '121': 'Industrial or commercial units',
    #  '122': 'Road and rail networks and associated land',
    #  '123': 'Port areas',
    #  '124': 'Airports',
    #  '131': 'Mineral extraction sites',
    #  '132': 'Dump sites',
    #  '133': 'Construction sites',
    #  '141': 'Green urban areas',
    #  '142': 'Sport and leisure facilities',
    #  '211': 'Non-irrigated arable land',
    #  '212': 'Permanently irrigated land',
    #  '213': 'Rice fields',
    #  '221': 'Vineyards',
    #  '222': 'Fruit trees and berry plantations',
    #  '223': 'Olive groves',
    #  '231': 'Pastures',
    #  '241': 'Annual crops associated with permanent crops',
    #  '242': 'Complex cultivation patterns',
    #  '243': 'Land principally occupied by agriculture, with significant areas of natural vegetation',
    #  '244': 'Agro-forestry areas',
    #  '311': 'Broad-leaved forest',
    #  '312': 'Coniferous forest',
    #  '313': 'Mixed forest',
    #  '321': 'Natural grasslands',
    #  '322': 'Moors and heathland',
    #  '323': 'Sclerophyllous vegetation',
    #  '324': 'Transitional woodland-shrub',
    #  '331': 'Beaches, dunes, sands',
    #  '332': 'Bare rocks',
    #  '333': 'Sparsely vegetated areas',
    #  '334': 'Burnt areas',
    #  '335': 'Glaciers and perpetual snow',
    #  '411': 'Inland marshes',
    #  '412': 'Peat bogs',
    #  '421': 'Salt marshes',
    #  '422': 'Salines',
    #  '423': 'Intertidal flats',
    #  '511': 'Water courses',
    #  '512': 'Water bodies',
    #  '521': 'Coastal lagoons',
    #  '522': 'Estuaries',
    #  '523': 'Sea and ocean'}
    
    
    CLC_root_depth = {
        'Road and rail networks and associated land': 1e-3,
        'Permanently irrigated land': 0.3,
        'Olive groves': 3,
        'Complex cultivation patterns': 0.5,
        'Agro-forestry areas': 3,
        'Coniferous forest': 3,
        'Natural grasslands': 0.2,
        'Sclerophyllous vegetation': 0.5,
        'Transitional woodland-shrub': 0.5,
        'nodata':1e-3,
        'Non-irrigated arable land':0.5,
        'Broad-leaved forest': 3,
        'Water courses': 1e-3,
        'Continuous urban fabric': 1e-3,
        'Discontinuous urban fabric': 1e-3,
        'Industrial or commercial units': 1e-3,
        'Dump sites': 1e-3,
        }
    
    return CLC_root_depth


def clip_rioxarray(ET_filelist,
                   ET_0_filelist,
                   rain_filelist,
                   majadas_aoi):
        
    for m in ET_filelist:
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
        
        clipped_etai['time']=utils.extract_filedate(m)
        clipped_etai.rio.to_raster('../prepro/Majadas/' + m.name)
        
    
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
        clipped_etrefi['time']=utils.extract_filedate(m)
        clipped_etrefi.rio.to_raster('../prepro/Majadas/' + m.name)
        
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
    
        clipped_raini['time']=utils.extract_filedate(m)
        clipped_raini.rio.to_raster('../prepro/Majadas/' + m.name)
        
    return clipped_etai, clipped_etrefi, clipped_raini
        
def export_tif2netcdf(pathTif2read='../prepro/Majadas/',fieldsite='Majadas'):
    
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
        ETafi['time']=utils.extract_filedate(m)
        ETa_l.append(ETafi)
        ETa_dates.append(ETafi['time'])
    
    
    ETp_l = []
    ETp_dates = []
    for m in ET_0_clipped_filelist:
        ETpfi = rxr.open_rasterio(m)
        ETpfi['time']=utils.extract_filedate(m)
        ETp_l.append(ETpfi)
        ETp_dates.append(ETpfi['time'])
    
    rain_l = []
    rain_dates = []
    for m in rain_clipped_filelist:
        rainfi = rxr.open_rasterio(m)
        rainfi['time']=utils.extract_filedate(m)
        rain_l.append(rainfi)
        rain_dates.append(rainfi['time'])
    
    ETp = xr.concat(ETp_l,dim='time')
    ETp.to_netcdf(f'../prepro/Majadas/ETp_{fieldsite}.netcdf')
    RAIN = xr.concat(rain_l,dim='time')
    RAIN.to_netcdf(f'../prepro/Majadas/RAIN_{fieldsite}.netcdf')
    ETa = xr.concat(ETa_l,dim='time')
    ETa.to_netcdf(f'../prepro/Majadas/ETa_{fieldsite}.netcdf')



def spatial_ET_animated(simu,fig,ax):
    # Plot the initial frame
    
    df_fort777 = out_CT.read_fort777(os.path.join(simu.workdir,
                                                  simu.project_name,
                                                  'fort.777'),
                                     )
    cax = simu.show('spatialET',
                             ax=ax, 
                             ti=0,
                             clim=[0,5e-9],
    
                       )
    ti = df_fort777['time'].unique()[1]
    df_fort777_select_t_xr = df_fort777.set_index(['time','x','y']).to_xarray()
    df_fort777_select_t_xr = df_fort777_select_t_xr.rio.set_spatial_dims('x','y')
    
    # Next we need to create a function that updates the values for the colormesh, as well as the title.
    def animate(frame):
        vi = df_fort777_select_t_xr.isel(time=frame)['ACT. ETRA'].values
        cax.set_array(vi)
        ax.set_title("Time = " + str(df_fort777_select_t_xr.coords['time'].values[frame])[:13])
    
    # Finally, we use the animation module to create the animation.
    ani = FuncAnimation(
        fig,             # figure
        animate,         # name of the function above
        frames=50,       # Could also be iterable or list
        interval=50     # ms between frames
    )
    # To save the animation using Pillow as a gif
    writer = animation.PillowWriter(fps=15,
                                    metadata=dict(artist='Me'),
                                    bitrate=1800)
    return ani, writer


def plot_time_serie_ETa_CATHY_TSEB(key_cover,
                                   df_fort777_select_t_xr,
                                   ds_analysis_EO,
                                   xPOI, yPOI,
                                   ax,
                                   ):
    
    # xPOI, yPOI =  gdf_AOI_POI_Majadas.set_index('POI/AOI').loc[key_cover].geometry.coords[0]
    
    ETa_poi = df_fort777_select_t_xr.sel(x=xPOI,
                                         y=yPOI, 
                                         method="nearest"
                                        )
    
    ETa_poi_datetimes = ds_analysis_EO.time.isel(time=0).values + ETa_poi.time.values
    
    ax.plot(ETa_poi_datetimes, 
            ETa_poi['ACT. ETRA'].values*1000*86400,
            linestyle='--',
            label='CATHY'
            )
    
    # ETa_TSEB = xr.open_dataset('../prepro/ETa_Majadas.netcdf')
    # ETa_TSEB = ETa_TSEB.rename({"__xarray_dataarray_variable__": "ETa_TSEB"})
    # ETa_TSEB = ETa_TSEB.to_dataarray().isel(variable=0,band=0).sortby('time')
    
    ETa_TSEB_poi = ds_analysis_EO.sel(x=xPOI,
                                    y=xPOI, 
                                    method="nearest"
                                    )
    ax.plot(ETa_TSEB_poi.time,
            ETa_TSEB_poi.values,
            label='TSEB'
            )
    ax.set_xlabel('Date')
    ax.set_ylabel('ETa (mm/day)')
    plt.legend()
    plt.title(key_cover)




def get_irr_center_coords(irrigation_map):
    nb_irr_areas = len(np.unique(irrigation_map)) -1
    patch_centers = {}
    patch_centers_CATHY = {}
    for irr_area_i in range(2,nb_irr_areas+2):
        mask = irrigation_map == irr_area_i
        labeled_array, num_features = label(mask)
        centers = center_of_mass(labeled_array) 
        center_x =  float(irrigation_map.x[int(centers[0])].values)
        center_y =  float(irrigation_map.y[int(centers[1])].values)
        patch_centers[irr_area_i] = [center_x,center_y]
        
        # I dont know why but values in the CATHY mesh are flipped!
        center_x_CATHY =  float(np.flipud(irrigation_map.x)[int(centers[0])])
        patch_centers_CATHY[irr_area_i] = [center_x_CATHY,center_y]
       
    return patch_centers, patch_centers_CATHY

def get_irr_time_trigger(grid_xr,irr_patch_center):
    x_center = irr_patch_center[0]
    y_center = irr_patch_center[1]
    irr_time_series = grid_xr['irr_daily'].sel(x=x_center, 
                                                y=y_center
                                                )   
    non_zero_indices = np.nonzero(irr_time_series.values)[0]
    if non_zero_indices.size > 0:
        first_non_zero_value = irr_time_series.values[non_zero_indices[0]]
        first_non_zero_time = irr_time_series.time[non_zero_indices[0]].values
    else:
        print("No non-zero values found in the time series at this location.")
    return non_zero_indices, first_non_zero_time, first_non_zero_value
        


def irrigation_delineation(ds_analysis,
                           threshold_local=0.25,
                           threshold_regional=0.25,
                           time_window=10,
                           ):
    decision_ds = utils.compute_ratio_ETap_local(ds_analysis,time_window=time_window)
    decision_ds = utils.compute_ratio_ETap_regional(decision_ds,time_window=time_window)

    # Create a boolean that check changes in ratioETap "ratio_ETap_local_diff"
    # -------------------------------------------------------------------------
    decision_ds = compute_bool_threshold_decision_local(ds_analysis,
                                                        threshold_local
                                                        )

    # Create a boolean that check regional changes in ratioETap 
    # -------------------------------------------------------------------------
    decision_ds = compute_bool_threshold_decision_regional(decision_ds,
                                                           threshold_regional
                                                           )
    
    # Create a dataset event_type of dim x,y,times
    # -------------------------------------------------------------------------
    # event_type = xr.DataArray(0, 
    #                           coords=decision_ds.coords, 
    #                           dims=decision_ds.dims
    #                           )
    # event_type = event_type.where(time_mask,drop=True)

    # Drop time 0 as the analysis is conducted on values differences (ti - t0)
    # -------------------------------------------------------------------------
    # time_mask = decision_ds['time'] > np.timedelta64(time_window, 'D')
    time_mask = decision_ds['time'] > np.timedelta64(0, 'D')
    decision_ds = decision_ds.where(time_mask, drop=True)

    # Apply rules of delineation
    # -------------------------------------------------------------------------
    decision_ds = apply_rules_rain(
                     decision_ds, 
                     # event_type
                    )
    
    decision_ds = apply_rules_irrigation(
                            decision_ds, 
                            # event_type
                        )
    # Classify based on rules
    # -------------------------------------------------------------------------
    event_type = classify_event(decision_ds)
    return decision_ds, event_type



def apply_rules_rain(decision_ds, 
                     # event_type
                    ):
    
    # There is input of water into the soil but due to rainfall 
    # (e.g. increase in regional ETa/p is over a threshold and
     # larger or similar to increase in local Eta/p)
     

    decision_ds['condRain1'] = decision_ds['threshold_regional']==1
    decision_ds['condRain2'] = abs(decision_ds['ratio_ETap_regional_spatial_avg_time_avg']) >= abs(decision_ds['ratio_ETap_local_time_avg'])
    decision_ds['condRain'] = decision_ds['condRain1'] & decision_ds['condRain2']
    # event_type = xr.where(decision_ds['condRain'] == True, 2, 0)

    
    # fig, ax = plt.subplots()
    # plot_analysis(decision_ds,prop='condRain1',ax=ax)
    # plt.savefig(os.path.join(figpath,f'condRain1.png'),
    #             dpi=300,
    #             )
    # fig, ax = plt.subplots()
    # plot_analysis(decision_ds,prop='condRain2',ax=ax)
    # plt.savefig(os.path.join(figpath,f'condRain2.png'),
    #             dpi=300,
    #             )
    
    # fig, ax = plt.subplots()
    # plot_analysis(decision_ds,prop='condRain',ax=ax)
    # plt.savefig(os.path.join(figpath,f'condRain.png'),
    #             dpi=300,
    #             )
    return decision_ds
    
    
def apply_rules_irrigation(decision_ds, 
                           # event_type
                           ):
    #  There is input of water to the soil due to irrigation (e.g. increase in local ETa/p is over a
    # threshold and significantly larger than increase in regional ETa/p)
    
    decision_ds['condIrrigation1'] = decision_ds['threshold_local']==1
    a = abs(decision_ds['ratio_ETap_local_time_avg'])
    b = abs(1.5*decision_ds['ratio_ETap_regional_spatial_avg_time_avg'])
    decision_ds['condIrrigation2'] = a > b
    decision_ds['condIrrigation'] = decision_ds['condIrrigation1'] & decision_ds['condIrrigation2']
    

    # fig, ax = plt.subplots()
    # plot_analysis(decision_ds,prop='condIrrigation1',ax=ax)
    # plt.savefig(os.path.join(figpath,f'condIrrigation1.png'),
    #             dpi=300,
    #             )
    
    # fig, ax = plt.subplots()
    # plot_analysis(decision_ds,prop='condIrrigation2',ax=ax)
    # plt.savefig(os.path.join(figpath,f'condIrrigation2.png'),
    #             dpi=300,
    #             )
    # fig, ax = plt.subplots()
    # plot_analysis(decision_ds,prop='condIrrigation',ax=ax)
    # plt.savefig(os.path.join(figpath,f'condIrrigation.png'),
    #             dpi=300,
    #             )
    return decision_ds

def classify_event(decision_ds):
    
    event_type = xr.where(decision_ds['condIrrigation'] == True, 1, 
                          xr.where(decision_ds['condRain'] == True, 2, 0)
                          )
    return event_type
    
    
def plot_irrigation_schedule(event_type,time_steps,fig,axes):
        
    axes = axes.flatten()  # Flatten to easily iterate over
    # Custom colormap with discrete colors corresponding to 'No input', 'irrigation', 'rain'
    cmap = plt.cm.colors.ListedColormap(['white', 
                                         'red', 
                                         'blue'
                                         ])
    x_values = event_type['x'].values
    y_values = event_type['y'].values
    extent = [x_values.min(), x_values.max(), y_values.min(), y_values.max()]
    for i, ax in enumerate(axes):
        if i < time_steps:  # Only plot if there is corresponding data
            data = event_type.isel(time=i).values  # or event_type.sel(time=...) if using labels
            img = ax.imshow(data, 
                            cmap=cmap, 
                            vmin=0, 
                            vmax=2, 
                            extent=extent,
                            origin='lower'
                            )
            
            # Set the title with the time step
            event_type['days'] = event_type['time'] / np.timedelta64(1, 'D')
    
            ax.set_title(f'Day {np.round(event_type.days.values[i],1)}')
            ax.set_xlabel('x')  # Label for the x-axis
            ax.set_ylabel('y')  # Label for the y-axis
        else:
            ax.axis('off')  # Turn off empty subplots
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, 
                                              norm=plt.Normalize(vmin=0, vmax=2)
                                              ), 
                        ax=axes, 
                        orientation='horizontal', 
                        fraction=0.02, pad=0.04)  # Adjust placement
    cbar.set_ticks([0, 1, 2])
    cbar.set_ticklabels(['No input', 'irrigation', 'rain'])
    
    pass

def plot_analysis(ds_analysis, prop='ratio_ETap_local',ax=None):
    facetgrid = ds_analysis[prop].plot.imshow(x="x",
                                                y="y", 
                                                col="time",
                                                col_wrap=4
                                                )
    
    time_in_days = ds_analysis.time.dt.days
    for ax, time_value in zip(facetgrid.axes.flat, 
                              time_in_days.values
                              ):
        ax.set_title(f"Time: {time_value}")


def plot_accounting_summary_analysis(axs,
                                     irr_patch_centers,
                                     patch_centers_CATHY,
                                     netIrr,
                                     simu_with_IRR,
                                     maxDEM,
                                     out_with_IRR,
                                     out_baseline,
                                     ETp,
                                     grid_xr_EO,
                                     ):
    for i, j in enumerate(irr_patch_centers):
        
        Net_irr_IN_1D = netIrr.sel(x=patch_centers_CATHY[j][1], 
                                   y=patch_centers_CATHY[j][0], 
                                   method='nearest'
                                   )
        df_Net_irr_IN_1D = Net_irr_IN_1D.to_dataframe(name='netIrr').reset_index()
        df_Net_irr_IN_1D.netIrr = df_Net_irr_IN_1D.netIrr*86400*1000
        
        
        node_index, _ = simu_with_IRR.find_nearest_node([patch_centers_CATHY[j][1],
                                                         patch_centers_CATHY[j][0],
                                                         maxDEM
                                                         ]
                                                        )
        
        
        ETa1d_index = np.where(out_with_IRR['ETa']['SURFACE NODE']==node_index[0])[0]
        ETa1d_with_IRR = out_with_IRR['ETa']['ACT. ETRA'].iloc[ETa1d_index[1:]]
        ETa1d_baseline = out_baseline['ETa']['ACT. ETRA'].iloc[ETa1d_index[1:]]
        ETa1d_net = ETa1d_with_IRR - ETa1d_baseline
        # axs[0,i].set_xlabel('')
    
        indexplot = (3)
        axs[0,i].plot(out_baseline['ETa'].time_sec.unique()[1:]/86400,
                            ETa1d_with_IRR*86400*1000,
                            label='Irr',
                            color='blue',
                            marker='*'
                            )
        axs[0,i].plot(out_baseline['ETa'].time_sec.unique()[1:]/86400,
                            ETa1d_baseline*86400*1000,
                            label='baseline',
                            color='red',
                            marker='*'
                            )
        
        if len(ETp.values)>1:
            axs[0,i].scatter(x=out_baseline['ETa'].time_sec.unique()[1:]/86400, 
                             y=ETp.values, 
                             color='k', 
                             linestyle='--', 
                             label='ETp'
                            )
        else:
            axs[0,i].axhline(y=abs(ETp.values), 
                                color='k', 
                                linestyle='--', 
                                label='ETp'
                                )
    
        # Creating a second y-axis
        ax2 = axs[0, i].twinx()
        ax2.plot(out_baseline['ETa'].time_sec.unique()[1:] / 86400,
                 ETa1d_net,
                 label='ETa1D_net',
                 color='green',
                 marker='o')
        
        # Setting label for the second y-axis
        if i == len(irr_patch_centers):
            ax2.set_ylabel('ETa1D_net (m/s)', color='green')
        
        # Optional: Adjust the color of the y-axis label to match the line color
        ax2.tick_params(axis='y', labelcolor='green')
        
        # Optional: Add legends from both y-axes
        lines_1, labels_1 = axs[0, i].get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        axs[0, -1].legend(lines_1 + lines_2, labels_1 + labels_2, loc='center left', bbox_to_anchor=(1, 0.5))
    
       
        utils.plot_in_subplot(
                                axs[1,i],
                                node_index,
                                out_with_IRR,
                                out_baseline,
                                prop='sw',
                                # datetime,
                                )
        if i == 0:
           axs[0,i].set_ylabel('ETa (m/s)')
           ax2.set_yticklabels([])
        else:
           axs[0,i].set_yticklabels([])
           ax2.set_yticklabels([])
           axs[1,i].set_ylabel('')
           axs[1,i].set_yticklabels('')
          
        axs[2,i].bar(df_Net_irr_IN_1D['time'].dt.days, 
               df_Net_irr_IN_1D['netIrr'], 
               color='skyblue', 
               edgecolor='k', 
               alpha=0.7
               )
        
        ax3 = axs[2,i].twinx()

        cumsum_netIrr = df_Net_irr_IN_1D.netIrr.cumsum()
        ax3.plot(netIrr['time'].dt.days[:], 
                cumsum_netIrr, 
                color='k', 
                linestyle='--'
                   )
        
        ax3.plot(grid_xr_EO['irr_daily'].time.dt.days.values, 
                grid_xr_EO['irr_daily'].sel(
                                            x=irr_patch_centers[j][0], 
                                            y=irr_patch_centers[j][1], 
                                            ).cumsum().values*86400*1000, 
                color='k', 
                linestyle='-'
                   )
        
        (non_zero_indices, 
         first_non_zero_time_days, 
         first_non_zero_value) = utils.get_irr_time_trigger(grid_xr_EO,
                                                             irr_patch_centers[j]
                                                              )
        # t_irr = first_non_zero_time_days
        t_irr = first_non_zero_time_days.astype('timedelta64[D]').astype(int)

        axs[2,i].axvline(x=t_irr, 
                        color='r', 
                        linestyle='--', 
                        label='Start Irr.')
        for i, j in enumerate(irr_patch_centers):
            axs[0, i].set_title(f'Irr{j}')
            
            
    axs[0,-1].plot(out_baseline['ETa'].time_sec.unique()[1:]/86400,
                    out_baseline['ETa'].groupby('time').mean()['ACT. ETRA'][1:].values*86400*1000,
                    label='baseline',
                    color='red',
                    marker='*'
                        )
    axs[0,-1].plot(out_with_IRR['ETa'].time_sec.unique()[1:]/86400,
                    out_with_IRR['ETa'].groupby('time').mean()['ACT. ETRA'][1:]*86400*1000,
                        label='Irr',
                        color='blue',
                        marker='*'
                        )
    axs[0, -1].set_title('ALL')
    
    
    axs[1, -1].plot(
                    out_with_IRR['times']/86400,
                    out_with_IRR['sw'].mean(axis=1).values, #sw2plot_with_IRR,
                    # label='Irrigated',
                    marker='.',
                    color='blue'
                    )
    axs[1, -1].plot(
                    out_baseline['times']/86400,
                    out_baseline['sw'].mean(axis=1).values, #sw2plot_with_IRR,
                    # label='Irrigated',
                    marker='.',
                    color='red'
                    )
    
       
    axs[2,0].set_xlabel('Days')
    axs[2,0].set_ylabel('net Irr. (mm)')
    
    axs[2,-1].bar(netIrr['time'].dt.days, 
               netIrr.mean(dim=['x','y'])*86400*1000, 
               color='skyblue', 
               edgecolor='k', 
               alpha=0.7
               )
    
    cumsum_mean_netIrr = netIrr.mean(dim=['x','y']).cumsum()*86400*1000

    ax3 = axs[2,-1].twinx()

    ax3.plot(netIrr['time'].dt.days[:], 
            cumsum_mean_netIrr, 
            color='k', 
            linestyle='--'
               )
    
    pass 



def plot_july_rain_irr(datetime, grid_xr, mask_IN, axs=None):
    
    july.heatmap(datetime, 
                 grid_xr['irr_daily'].sum(['x','y']), 
                 title='Irrigation (mm/h)',
                 cmap="github",
                 colorbar=True,
                 ax=axs[0]
                 )
    july.heatmap(datetime, 
                 grid_xr['rain_daily'].sum(['x','y']), 
                 title='Rain (mm/h)',
                 cmap="golden",
                 colorbar=True,
                 ax=axs[1]
                 )
    pass
    


def get_mask_IN_patch_i(irrigation_map_xr,patchid=0):
    mask_IN = irrigation_map_xr==patchid
    return mask_IN

def get_mask_OUT(irrigation_map_xr,patchid=0):
    mask_OUT = irrigation_map_xr==1
    return mask_OUT



def plot_patches_irrigated_states(irr_patch_centers,
                                  patch_centers_CATHY,
                                  simu_with_IRR,
                                  maxDEM,
                                  grid_xr_with_IRR,
                                  sc,
                                  axs,
                                  out_with_IRR,
                                  out_baseline,
                                   dates,
                                  ):
    for i, j in enumerate(irr_patch_centers):
        node_index, _ = simu_with_IRR.find_nearest_node([patch_centers_CATHY[j][1],
                                                         patch_centers_CATHY[j][0],
                                                         maxDEM
                                                         ]
                                                        )   
        (non_zero_indices, 
         first_non_zero_time_days, 
         first_non_zero_value) = utils.get_irr_time_trigger(grid_xr_with_IRR,
                                                             irr_patch_centers[j]
                                                              )
        t_irr = first_non_zero_time_days.astype('timedelta64[s]').astype(int)
        mask_IN = get_mask_IN_patch_i(grid_xr_with_IRR['irrigation_map'],patchid=j)
        ETp_node_IN = grid_xr_with_IRR['ETp_daily'].where(mask_IN, drop=True).mean(['x','y'])
    
        utils.plot_1d_evol(
                            simu_with_IRR,
                            node_index,
                            out_with_IRR,
                            out_baseline,
                            ETp_node_IN.values,
                            axs,
                            scenario=sc,
                            dates = dates
                            # timeIrr_sec = t_irr,
                        )


def plot_patches_NOirrgation_states(simu_with_IRR,
                                    out_irr,
                                    maxDEM,
                                    grid_xr_with_IRR,
                                    out_with_IRR,
                                    out_baseline,
                                    axs,
                                    sc,
                                    dates,
                                   ):

    node_index_OUT, _ = simu_with_IRR.find_nearest_node([out_irr[0][0],
                                                         out_irr[1][0],
                                                         maxDEM
                                                         ]
                                                        )

    # ETp_node_IN = grid_xr_with_IRR['ETp_daily'][node_index]
    mask_OUT = grid_xr_with_IRR['irrigation_map']==1
    ETp_node_OUT = grid_xr_with_IRR['ETp_daily'].where(mask_OUT, drop=True).mean(['x','y'])
    
    utils.plot_1d_evol(
                        simu_with_IRR,
                        node_index_OUT,
                        out_with_IRR,
                        out_baseline,
                        ETp_node_OUT,
                        axs,
                        scenario=sc,
                        dates = dates
                    )

def custum_axis_patches_states(axs,
                               irr_patch_centers,
                               ):
    # Assuming `axes` is your array of Axes objects (like the one you provided)
    n_rows, n_cols = axs.shape
    
    for i, j in enumerate(irr_patch_centers):
        # Set title for the top subplot in each column
        axs[0, i].set_title(f'Irr{j}')
        
    for i in range(n_rows):
        for j in range(n_cols):
            ax = axs[i, j]
            
            # Hide x-axis labels and ticks for all but the last row
            if i < n_rows - 1:
                ax.set_xticklabels([])
                ax.set_xlabel('')
                ax.set_xticks([])
            
            # Hide y-axis labels and ticks for all but the first column
            if j > 0:
                ax.set_yticklabels([])
                ax.set_ylabel('')
                ax.set_yticks([])
    
    # Format the x-axis with datetime labels
    axs[-1,0].xaxis.set_major_locator(mdates.AutoDateLocator())  # Automatically set major ticks
    axs[-1,0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Set date format
    
    axs[0,0].xaxis.set_major_locator(mdates.AutoDateLocator())  # Automatically set major ticks
    axs[0,0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Set date format
    
    # Rotate the x-axis labels for better readability
    axs[-1, 0].tick_params(axis='x', rotation=45)
    axs[-1, 1].tick_params(axis='x', rotation=45)



def apply_EO_rules(ds_analysis_EO,sc_EO):
    
    ds_analysis_EO_ruled = ds_analysis_EO
    if 'EO_resolution' in sc_EO:
        # Assume sc_EO['EO_resolution'] is the desired new resolution (e.g., 30 meters)
        new_resolution_x = sc_EO['EO_resolution']
        new_resolution_y = sc_EO['EO_resolution']
        
        # Calculate the scale factors (coarsening factors) based on the current resolution and desired resolution
        scale_factor_x = int(new_resolution_x / ds_analysis_EO.rio.resolution()[0])
        scale_factor_y = int(new_resolution_y / ds_analysis_EO.rio.resolution()[1])
        
        # Resample the DataArray to the new resolution
        ds_analysis_EO_ruled = ds_analysis_EO_ruled.coarsen(
            x=scale_factor_x, 
            y=scale_factor_y, 
            boundary="trim"
        ).mean()
        
        # fig, axs = plt.subplots(1,2)
        # ds_analysis_EO['ACT. ETRA'].isel(time=5).plot.imshow(ax=axs[0])
        # ds_analysis_EO_ruled['ACT. ETRA'].isel(time=5).plot.imshow(ax=axs[1])
    if 'EO_freq_days' in sc_EO:
        new_frequency = sc_EO['EO_freq_days']
        
        # Create a boolean mask where True indicates the days to keep
        mask = ds_analysis_EO_ruled.time.to_index() % new_frequency == 0
        
        # Convert the mask to a DataArray
        mask_da = xr.DataArray(mask, coords=[ds_analysis_EO_ruled.time], 
                               dims=['time']
                               )
        
        # Apply the mask to set values to NaN where the mask is False
        ds_analysis_EO_ruled = ds_analysis_EO_ruled.where(mask_da, np.nan)


    return ds_analysis_EO_ruled

def check_and_tune_E0_dict(sc):
    '''
    Add EO criteria (resolution, frequency, type, ...)
    '''
    sc_EO = {}
    if sc.get('microwaweMesh'):
        sc_EO.update({'maxdepth': 0.05})
    if sc.get('EO_freq'):
        sc_EO.update({'EO_freq': sc.get('EO_freq')})
    if sc.get('EO_resolution'):
        sc_EO.update({'EO_resolution': sc.get('EO_resolution')})
    if sc.get('PERMX'):
        sc_EO.update({'SOIL_PERMX': sc.get('PERMX')})
        
    return sc_EO

#%% Data Assimilation

     
def backup_simulog_DA(args,filename='DAlog.csv'):
    results_df = pd.read_csv(filename,index_col=0)
    now = datetime.now()
    results_df_cols = vars(args).keys()
    results_df_new = pd.DataFrame([vars(args)])
    cols2check = list(vars(args).keys())
    
    values = results_df_new[cols2check].values
    matching_index = results_df.index[(results_df[cols2check] == values).all(axis=1)].tolist()
    if matching_index:
        now = datetime.now()
        results_df.loc[matching_index, 'datetime'] = now
        matching_index = matching_index[0]
    else:
        results_df_new['datetime']=now
        results_df = pd.concat([results_df,results_df_new],ignore_index=True)
        matching_index = len(results_df)-1
    results_df.to_csv(filename)
    return results_df, matching_index

#%%

def prep_AQUACROP_inputs(wdf,args):
    sim_start = wdf.Date.iloc[0].strftime('%Y/%m/%d')
    sim_end = wdf.Date.iloc[365].strftime('%Y/%m/%d')
    soil= Soil('SandyLoam')
    crop = Crop('Maize',
                planting_date='07/06'
                )
    initWC = InitialWaterContent(value=['FC'])
    labels=[]
    outputs=[]
    smt = args.SMT
    crop.Name = str(smt) # add helpfull label
    labels.append(str(smt))
    
    # SMT (list):  Soil moisture targets (%taw) to maintain in each growth stage (only used if irrigation method is equal to 1)
    irr_mngt = IrrigationManagement(irrigation_method=1,
                                    SMT=[smt]*4 # same for each developement growth stages 
                                    ) # specify irrigation management
    return sim_start, sim_end, soil, crop, initWC, irr_mngt

def prep_ERA5_reanalysis_data_SPAIN(dataPath):
    # ERA5ds = xr.open_dataset(dataPath / 'test/data_stream-oper.nc')
    ERA5ds = xr.open_dataset(dataPath /'data_SPAIN_ERA5_singlelevel_hourly.nc')
    ERA5ds = ERA5ds.rio.write_crs("EPSG:4326")
    
    # Central point in lat/lon
    central_lat = 39.978757
    central_lon = -5.81843
    
    # Calculate the degree distance for 7.5 km
    delta_lat = 100 / 111  # Approximate change in degrees latitude
    delta_lon = 100 / (111 * np.cos(np.radians(central_lat)))  # Change in degrees longitude
    
    # Define the bounding box in lat/lon
    min_lat = central_lat - delta_lat
    max_lat = central_lat + delta_lat
    min_lon = central_lon - delta_lon
    max_lon = central_lon + delta_lon
    
    # Crop the dataset using the bounding box
    cropped_ERA5ds = ERA5ds.sel(latitude=slice(max_lat, min_lat), longitude=slice(min_lon, max_lon))
    #%%
    fig, ax = plt.subplots()
    ERA5ds.pev.isel(valid_time=0).plot.imshow(ax=ax,
                                            add_colorbar=False
                                            )
    ax.set_aspect('equal')
    # cx.add_basemap(ax, 
    #                 crs=cropped_ERA5ds.rio.crs,
    #                 alpha=0.4,
    #                 # credit=False
    #                 )
    # Resample to daily frequency and calculate the mean for each day
    # daily_ERA5ds = cropped_ERA5ds.resample(valid_time='1D').mean().mean(['latitude','longitude'])
    # cropped_ERA5ds['mint2m'] = cropped_ERA5ds.resample(valid_time='1D').min('t2m').mean(['latitude','longitude'])
    maxt2m = cropped_ERA5ds.resample(valid_time='1D').max().mean(['latitude','longitude'])['t2m']
    mint2m  = cropped_ERA5ds.resample(valid_time='1D').min().mean(['latitude','longitude'])['t2m']
    sumtp = cropped_ERA5ds.resample(valid_time='1D').sum().mean(['latitude','longitude'])['tp']
    sumpev  = cropped_ERA5ds.resample(valid_time='1D').sum().mean(['latitude','longitude'])['pev']
    # Create a new Dataset with the calculated variables
    analysis_xr = xr.Dataset({
        'maxt2m': maxt2m - 273.15,
        'mint2m': mint2m - 273.15,
        'sumtp': sumtp*1000,
        'sumpev': abs(sumpev)*1000
    })
    analysis_xr.attrs['description'] = 'Daily aggregated statistics from the cropped ERA5 dataset'
    return analysis_xr

def create_scenario_ERA5(analysis_xr,args,dataPath):
    scenario_analysis = analysis_xr.copy()
    if args.weather_scenario == 'plus20p_tp':
        scenario_analysis = analysis_xr.copy()
        scenario_analysis['sumtp'] = scenario_analysis['sumtp'] * 1.20
    elif args.weather_scenario =='minus20p_tp':
        # Scenario 2: -20% precipitation
        scenario_analysis = analysis_xr.copy()
        scenario_analysis['sumtp'] = scenario_analysis['sumtp'] * 0.80
    elif args.weather_scenario =='plus25p_t2m':
        # Scenario 3: +25% air temperature
        scenario_analysis = analysis_xr.copy()
        scenario_analysis['maxt2m'] = scenario_analysis['maxt2m'] * 1.25
        scenario_analysis['mint2m'] = scenario_analysis['mint2m'] * 1.25
    
    # dataPath = Path('../data/Spain/Spain_ETp_Copernicus_CDS/')
    # Save the scenario datasets to new NetCDF files
    # analysis_xr.to_netcdf(dataPath/'era5_scenario_ref.nc')
    # scenario_analysis.to_netcdf(f'{dataPath}/era5_scenario{args.scenario_nb}_weather_{args.weather_scenario}.nc')
    scenario_analysis.to_netcdf(f'{dataPath}/era5_scenario{args.scenario_nb}_weather_{args.weather_scenario}.nc')
    # scenario_analysis
    
    # scenario2.to_netcdf(dataPath/'era5_scenario2_precipitation_minus20.nc')
    # scenario3.to_netcdf(dataPath/'era5_scenario3_temperature_plus25.nc')
    # scenario1.to_netcdf(dataPath/'era5_scenario1_precipitation_plus20.nc')
    # scenario2.to_netcdf(dataPath/'era5_scenario2_precipitation_minus20.nc')
    # scenario3.to_netcdf(dataPath/'era5_scenario3_temperature_plus25.nc')
    
    wdf = analysis_xr.to_dataframe()
    wdf = wdf.reset_index()
    wdf = wdf.rename(columns={
        'valid_time': 'Date', 
        'maxt2m': 'MaxTemp', 
        'mint2m': 'MinTemp', 
        'sumpev': 'ReferenceET', 
        'sumtp': 'Precipitation'
    })
    wdf = wdf[['MinTemp','MaxTemp','Precipitation','ReferenceET','Date']]
    return wdf, scenario_analysis

def plot_weather_ET_timeserie(analysis_xr,
                              scenario_analysis,
                              axs):
    analysis_xr.plot.scatter(x='valid_time',
                                y='sumpev',
                                ax=axs[0],
                                color='k',
                                s=2
                                )
    scenario_analysis.plot.scatter(x='valid_time',
                                y='sumpev',
                                ax=axs[0],
                                color='red',
                                s=2
                                )
    analysis_xr.plot.scatter(x='valid_time',
                                y='maxt2m',
                                ax=axs[1],
                                color='k',
                                s=2
                                )
    analysis_xr.plot.scatter(x='valid_time',
                                y='mint2m',
                                ax=axs[1],
                                color='k',
                                s=2
                                )
    scenario_analysis.plot.scatter(x='valid_time',
                                y='mint2m',
                                ax=axs[1],
                                color='r',
                                s=2
                                )
    scenario_analysis.plot.scatter(x='valid_time',
                                y='maxt2m',
                                ax=axs[1],
                                color='r',
                                s=2
                                )
    scenario_analysis.plot.scatter(x='valid_time',
                            y='sumtp',
                            ax=axs[2],
                            color='r',
                            s=2
                            )
    analysis_xr.plot.scatter(x='valid_time',
                            y='sumtp',
                            ax=axs[2],
                            color='k',
                            s=2
                            )
    axs[0].set_title('')
    axs[1].set_title('')
    axs[2].set_title('')
    axs[0].set_xlabel('')
    axs[1].set_xlabel('')
    
    

def plot_atmbc_rain_irr_events(ax,
                               dates,
                               mean_irr_daily,
                               mean_rain_daily, 
                               perc_detection = None,
                               colors='blue',
                               ):
    
    # Plot the bar chart
    ax.bar(dates, 
           mean_irr_daily*(1e3*86400), 
           color=colors)
   
    ax.set_ylim([0,100])
    
    # Create a second y-axis for rainfall
    ax2 = ax.twinx()
    # ax2.spines['right'].set_position(('axes', 1.0))  # Shift ax3 to the right by 0.05 from the default position
    ax2.bar(dates, mean_rain_daily*(1e3*86400), color='blue', alpha=0.5)
    ax2.set_ylim([0,100])
    ax2.invert_yaxis()
    
    # Format the x-axis with datetime labels
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())  # Automatically set major ticks
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # Set date format
    
    # Rotate the x-axis labels for better readability
    plt.xticks(rotation=45)
    
    plt.xlabel('Date')
    ax.set_ylabel('Irrigation Daily Mean \n (mm/day)')
    ax2.set_ylabel('Rain Daily Mean \n (mm/day)',color='blue')
    plt.title(f'%of detected irr events={perc_detection}%')
    
    return ax2