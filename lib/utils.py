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
from shapely.geometry import mapping
from scipy import stats
from scipy.ndimage import label, center_of_mass


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

    
def compute_ratio_ETap_local(ds_analysis,
                             ETa_name='ACT. ETRA',
                             ETp_name='ETp',
                             ):
    ds_analysis["ratio_ETap_local"] = ds_analysis[ETa_name]/ds_analysis[ETp_name]
    
    #
    # calculating the change in **ETa/p** between the time on which irrigation
    # is to be detect and most recent previous time on which ET estimates are available.
    #
    ds_analysis["ratio_ETap_local_diff"] = ds_analysis["ratio_ETap_local"].diff(dim='time')
    
    return ds_analysis

def compute_regional_ETap(ds_analysis,
                          ETa_name='ACT. ETRA',
                          ETp_name='ETp',
                          window_size_x=10
                          ):
    # (i.e. as an average change in all agricultural pixels within 10 km window)
    # Compute the mean on X and Y dimensions for the ETp variable
    for pp in [ETa_name, ETp_name]:
        mean = ds_analysis[pp].mean()
        mean_dataarray = xr.full_like(ds_analysis[pp], 
                                      fill_value=mean
                                      )
        ds_analysis[pp+'_mean'] = mean_dataarray
    return ds_analysis

def compute_ratio_ETap_regional(ds_analysis,
                                ETa_name='ACT. ETRA',
                                ETp_name='ETp',
                                stat = 'mean'
                                ):
    ds_analysis["ratio_ETap_regional"] = ds_analysis[f'{ETa_name}']/ds_analysis[f'{ETp_name}']
    #
    # calculating the change in **ETa/p** between the time on which irrigation
    # is to be detect and most recent previous time on which ET estimates are available.
    #
    if stat == 'mean':
        # ds_analysis = compute_regional_ETap(ds_analysis,ETa_name,ETp_name)
        # ds_analysis = compute_regional_ETap(ds_analysis,ETa_name,ETp_name)
        mean = ds_analysis["ratio_ETap_regional"].mean(dim=['x','y'])
        mean_dataarray = xr.full_like(ds_analysis['ratio_ETap_regional'], 
                                  fill_value=0
                                  )
        for i, m in enumerate(mean.values):
            timei = mean_dataarray.time[i]
            mean_dataarray.loc[{'time': timei}] = m
            
        ds_analysis["ratio_ETap_regional_mean"] = mean_dataarray
        ds_analysis["ratio_ETap_regional_diff"] = ds_analysis["ratio_ETap_regional_mean"].diff(dim='time')
    
    return ds_analysis


def compute_bool_threshold_decision_local(ds_analysis,
                                          threshold_local=-0.25
                                          ):
        # Initialize 'threshold_local' with False values
    ds_analysis["threshold_local"] = xr.DataArray(False, 
                                                  coords=ds_analysis.coords, 
                                                  dims=ds_analysis.dims
                                                     )
    # Set 'threshold_local' to True where condition is met
    checkon = ds_analysis["ratio_ETap_local_diff"]
    ds_analysis["threshold_local"] = xr.where(checkon <= threshold_local, True, False)

    return ds_analysis

def compute_bool_threshold_decision_regional(ds_analysis,
                                             threshold_regional=-0.25
                                             ):
    
    ds_analysis["threshold_regional"] = xr.DataArray(False, 
                                                  coords=ds_analysis.coords, 
                                                  dims=ds_analysis.dims
                                                     )
    checkon = ds_analysis["ratio_ETap_regional_diff"]
    ds_analysis["threshold_regional"] = xr.where(checkon <= threshold_regional, True, False)
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
                    ):

    sw2plot_with_IRR =  out_with_IRR[prop].iloc[:,node_index].values
    sw2plot_baseline =  out_baseline[prop].iloc[:,node_index].values
    
    ax.plot(
            out_with_IRR['times']/86400,
            sw2plot_with_IRR,
            # label='Irrigated',
            marker='.',
            color='blue'
            )
    ax.plot(
            out_with_IRR['times']/86400,
            sw2plot_baseline,
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
    
def plot_1d_evol(simu,
                 node_index,
                 out_with_IRR,
                 out_baseline,
                 ETp,
                 axs,
                 **kwargs
                 ):
    
   
    # simu.show_input('atmbc',ax=axs[0],
    #                 units='days'
    #                 )
    
    df_atmbc = simu.read_inputs('atmbc')
    df_atmbc['idnode'] = np.tile(np.arange(0,int(len(df_atmbc)/len(np.unique(df_atmbc.time)))),
                                 len(np.unique(df_atmbc.time))
                                 )
                                 
    mask_node = df_atmbc['idnode']==node_index[0]
    df_atmbc[mask_node]
    # df_atmbc_mean = df_atmbc.groupby('time').mean()
    # hydro_Majadas
    # df_atmbc = hydro_Majadas.read_inputs('atmbc')

    axs[0].bar(df_atmbc[mask_node].time/86400,
               df_atmbc[mask_node].value.values
                )
    # axs[0].scatter(df_atmbc_mean.index/86400,
    #             df_atmbc_mean.value.values
    #             )
    # axs[0].set_title('')
    # simu_with_IRR.show_input('atmbc',ax=axs[0])
    
    timeIrr_sec = None
    if 'timeIrr_sec' in kwargs:
        timeIrr_sec = kwargs.pop('timeIrr_sec')

    scenario = None
    if 'scenario' in kwargs:
        scenario = kwargs.pop('scenario')


    utils.plot_in_subplot(
                            axs[1],
                            node_index,
                            out_with_IRR,
                            out_baseline,
                            prop='sw',
                            )
    
    
    utils.plot_in_subplot(
                            axs[2],
                            node_index,
                            out_with_IRR,
                            out_baseline,
                            prop='psi',
                            )
    
    # ETa1d_index, ETa1d_with_IRR, ETa1d_baseline = find_irr_surface_node(index,
    #                                                               out_with_IRR,
    #                                                               out_baseline
    #                                                               )
    
    ETa1d_index = np.where(out_with_IRR['ETa']['SURFACE NODE']==node_index[0])[0]
    ETa1d_with_IRR = out_with_IRR['ETa']['ACT. ETRA'].iloc[ETa1d_index[1:]]
    ETa1d_baseline = out_baseline['ETa']['ACT. ETRA'].iloc[ETa1d_index[1:]]
    axs[0].set_xlabel('')
    axs[1].set_xlabel('')

    indexplot = (3)
    axs[indexplot].plot(out_baseline['ETa'].time_sec.unique()[1:]/86400,
                        ETa1d_with_IRR,
                        label='Irr',
                        color='blue',
                        marker='*'
                        )
    axs[indexplot].plot(out_baseline['ETa'].time_sec.unique()[1:]/86400,
                        ETa1d_baseline,
                        label='baseline',
                        color='red',
                        marker='*'
                        )
    
    axs[indexplot].axhline(y=abs(ETp), 
                           color='k', 
                           linestyle='--', 
                           label='ETp')
    # axs[indexplot].legend()
    axs[indexplot].set_xlabel('time')
    axs[indexplot].set_ylabel('ETa (m/s)')

    if timeIrr_sec is not None:
        axs[indexplot].axvline(x=timeIrr_sec/86400, 
                               color='r', 
                               linestyle='--', 
                               label='Start Irr.')
        # axs[indexplot].axvline(x=timeIrr_sec + scenario['irr_length'], 
        #                        color='r', 
        #                        linestyle='--', 
        #                        label='End Irr.')
    
    # Get current tick positions in seconds
    # xticks = out_baseline['ETa'].time_sec.unique()[1:]

    # # Convert tick positions to days
    # xtick_labels = seconds_to_days(xticks)
    # axs[-1].set_xticks(xtick_labels)

    # # Update x-axis tick labels
    # axs[-1].set_xticklabels([f'{int(day)}' for day in xtick_labels])
    
    # # Set x-axis label
    # axs[-1].set_xlabel('Time (days)')
    
    
 
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
        'Discontinuous urban fabric':1e-3,
        'nodata':1e-3,
        'Non-irrigated arable land':0.5,
        'Broad-leaved forest': 3,
        'Water courses': 1e-3,
        }
    
    return CLC_root_depth


def clip_rioxarray(ET_filelist,ET_0_filelist,rain_filelist,
                   majadas_aoi):
        
    for m in ET_filelist:
        etai = rxr.open_rasterio(m)
        clipped_etai = etai.rio.clip_box(
                                          minx=majadas_aoi.bounds['minx'],
                                          miny=majadas_aoi.bounds['miny'],
                                          maxx=majadas_aoi.bounds['maxx'],
                                          maxy=majadas_aoi.bounds['maxy'],
                                        crs=majadas_aoi.crs,
                                        )   
        
        # clipped_etai = etai.rio.clip(
        #                                 majadas_aoi.geometry,
        #                                 crs=majadas_aoi.crs,
        #                                 )  
        
        clipped_etai['time']=utils.extract_filedate(m)
        clipped_etai.rio.to_raster('../prepro/Majadas/' + m.name)
        
    
    for m in ET_0_filelist:
        etrefi = rxr.open_rasterio(m)
        clipped_etrefi = etrefi.rio.clip_box(
                                              minx=majadas_aoi.bounds['minx'],
                                              miny=majadas_aoi.bounds['miny'],
                                              maxx=majadas_aoi.bounds['maxx'],
                                              maxy=majadas_aoi.bounds['maxy'],
                                            crs=majadas_aoi.crs,
                                            )   
        # clipped_etrefi = etrefi.rio.clip(
        #                                     majadas_aoi.geometry,
        #                                     crs=majadas_aoi.crs,
        #                                     )   
        clipped_etrefi['time']=utils.extract_filedate(m)
        clipped_etrefi.rio.to_raster('../prepro/Majadas/' + m.name)
        
    for m in rain_filelist:
        raini = rxr.open_rasterio(m)
        clipped_raini = raini.rio.clip_box(
                                              minx=majadas_aoi.bounds['minx'],
                                              miny=majadas_aoi.bounds['miny'],
                                              maxx=majadas_aoi.bounds['maxx'],
                                              maxy=majadas_aoi.bounds['maxy'],
                                            crs=majadas_aoi.crs,
                                            )   
        # clipped_raini = raini.rio.clip(
        #                                     majadas_aoi.geometry,
        #                                     crs=majadas_aoi.crs,
        #                                     )   
    
        clipped_raini['time']=utils.extract_filedate(m)
        clipped_raini.rio.to_raster('../prepro/Majadas/' + m.name)
        
def export_tif2netcdf(pathTif2read='../prepro/Majadas/'):
    
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
    
    rain = []
    rain_dates = []
    for m in rain_clipped_filelist:
        rainfi = rxr.open_rasterio(m)
        rainfi['time']=utils.extract_filedate(m)
        rain.append(rainfi)
        rain_dates.append(rainfi['time'])
    
    ETp = xr.concat(ETp_l,dim='time')
    ETp.to_netcdf('../prepro/Majadas/ETp_Majadas.netcdf')
    RAIN = xr.concat(rain,dim='time')
    RAIN.to_netcdf('../prepro/Majadas/RAIN_Majadas.netcdf')
    ETa = xr.concat(ETa_l,dim='time')
    ETa.to_netcdf('../prepro/Majadas/ETa_Majadas.netcdf')

def read_prepo_EO_datasets(fieldsite='Majadas'):
    ETa_ds = xr.open_dataset(f'../prepro/Majadas/ETa_{fieldsite}.netcdf')
    ETa_ds = ETa_ds.rename({"__xarray_dataarray_variable__": "ETa"})
    ETp_ds = xr.open_dataset(f'../prepro/Majadas/ETp_{fieldsite}.netcdf')
    ETp_ds = ETp_ds.rename({"__xarray_dataarray_variable__": "ETp"})
    RAIN_ds = xr.open_dataset(f'../prepro/Majadas/RAIN_{fieldsite}.netcdf')
    RAIN_ds = RAIN_ds.rename({"__xarray_dataarray_variable__": "RAIN"})
    CLC_ds = xr.open_dataset(f'../prepro/Majadas/CLCover_{fieldsite}.netcdf')

    ds_analysis_EO = ETa_ds.to_dataarray().isel(variable=0,band=0)
    ds_analysis_EO['ETa'] = ETa_ds.to_dataarray().isel(variable=0,band=0)
    ds_analysis_EO['ETp'] = ETp_ds.to_dataarray().isel(variable=0,band=0)
    ds_analysis_EO['RAIN'] = RAIN_ds.to_dataarray().isel(variable=0,band=0)
    ds_analysis_EO = ds_analysis_EO.drop_vars('spatial_ref', errors='ignore')

    CLC_ds = CLC_ds.drop_vars('spatial_ref', errors='ignore')
    ds_analysis_EO['CLC_code18'] = CLC_ds.Code_18
    # ds_analysis_EO.to_netcdf('../prepro/ds_analysis_EO.netcdf')
    ds_analysis_EO = ds_analysis_EO.sortby('time')
    
    nulltimeETa = np.where(ds_analysis_EO.ETa.isel(x=0,y=0).isnull())[0]
    valid_mask = ~ds_analysis_EO.time.isin(ds_analysis_EO.time[nulltimeETa])
    
    if len(nulltimeETa)>1:
        print('times with null ETa values!!')
    ds_analysis_EO = ds_analysis_EO.isel(time=valid_mask)
    
    print('Errrrrorrr in rain evaluation in the input!')
    # data_array = data_array.where((data_array <= 300) & (data_array > 0), other=np.nan)
    ds_analysis_EO['RAIN'] = ds_analysis_EO['RAIN'].where((ds_analysis_EO['RAIN'] <= 300) & (ds_analysis_EO['RAIN'] > 0), 
                                                          other=0)
    
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
        first_non_zero_time = irr_time_series.time_days[non_zero_indices[0]].values
    else:
        print("No non-zero values found in the time series at this location.")
    return non_zero_indices, first_non_zero_time, first_non_zero_value
        


def irrigation_delineation(decision_ds,
                           threshold_local=-0.25,
                           threshold_regional=-0.25,
                           ):
    decision_ds = compute_ratio_ETap_local(decision_ds)
    decision_ds = utils.compute_ratio_ETap_regional(decision_ds)

    # Create a boolean that check changes in ratioETap "ratio_ETap_local_diff"
    # -------------------------------------------------------------------------
    decision_ds = compute_bool_threshold_decision_local(decision_ds,
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
    return event_type



def apply_rules_rain(decision_ds, 
                     # event_type
                    ):
    
    # There is input of water into the soil but due to rainfall 
    # (e.g. increase in regional ETa/p is over a threshold and
     # larger or similar to increase in local Eta/p)
     
    
    
    decision_ds['condRain1'] = decision_ds['threshold_regional']==1
    decision_ds['condRain2'] = abs(decision_ds['ratio_ETap_regional_diff']) >= abs(decision_ds['ratio_ETap_local_diff'])
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
    decision_ds['condIrrigation2'] = abs(decision_ds['ratio_ETap_local_diff']) > abs(1.5*decision_ds['ratio_ETap_regional_diff'])
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
        
        (non_zero_indices, 
         first_non_zero_time_days, 
         first_non_zero_value) = utils.get_irr_time_trigger(grid_xr_EO,
                                                             irr_patch_centers[j]
                                                              )
        t_irr = first_non_zero_time_days
        
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
    pass 