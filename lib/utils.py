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


def extract_filedate(file_path):
    file_name = file_path.name
    date_str = file_name.split('_')[0]
    return datetime.strptime(date_str, '%Y%m%d')

#%% Function to build the analysis xarray dataset

def get_analysis_ds(out_with_IRR):
    ds_multiindex = out_with_IRR.reset_index()
    ds_multiindex = ds_multiindex.set_index(['time', 'X', 'Y'])
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
    ds_analysis["ETp"] = (("time", "X", "Y"), [padded_ETp]*len(ds_analysis.time))
    return ds_analysis

    
def compute_ratio_ETap_local(ds_analysis,
                             ETa_name='ACT. ETRA',
                             ETp_name='ETp',
                             ):
    ds_analysis["ratio_ETap_local"] = ds_analysis[ETa_name]/ds_analysis[ETp_name]
    return ds_analysis

def compute_regional_ETap(ds_analysis,
                          ETa_name='ACT. ETRA',
                          ETp_name='ETp',
                          window_size_x=10
                          ):
    # (i.e. as an average change in all agricultural pixels within 10 km window)
    # Compute local ratio to check: 
    # b) There is input of water into the soil but due to rainfall (e.g. increase in regional ETa/p is over a
    # threshold and larger or similar to increase in local Eta/p)
    # c) There is input of water to the soil due to irrigation (e.g. increase in local ETa/p is over a
    # threshold and significantly larger than increase in regional ETa/p)
    # Compute the rolling mean on X and Y dimensions for the ETp variable
    for pp in [ETa_name, ETp_name]:
        rolling_mean = ds_analysis[pp].rolling(x=window_size_x, 
                                               y=window_size_x, 
                                               center=True,
                                              ).mean()
        ds_analysis[pp+'_rolling_mean'] = rolling_mean
    return ds_analysis

def compute_ratio_ETap_regional(ds_analysis,
                                ETa_name='ACT. ETRA',
                                ETp_name='ETp',
                                ):
    ds_analysis["ratio_ETap_rolling_regional"] = ds_analysis[f'{ETa_name}_rolling_mean']/ds_analysis[f'{ETp_name}_rolling_mean']
    return ds_analysis


def compute_threshold_decision_local(ds_analysis,threshold=0.6):
    # Initialize 'threshold_local' with False values
    ds_analysis["threshold_local"] = xr.DataArray(False, 
                                                  coords=ds_analysis.coords, 
                                                  dims=ds_analysis.dims
                                                     )
    # Set 'threshold_local' to True where condition is met
    ds_analysis["threshold_local"] = ds_analysis["threshold_local"].where(abs(ds_analysis['ratio_ETap_local']) <= threshold, True)

    return ds_analysis

def compute_threshold_decision_regional(ds_analysis,threshold=0.6):
    ds_analysis["threshold_regional"] = xr.DataArray(False, 
                                                  coords=ds_analysis.coords, 
                                                  dims=ds_analysis.dims
                                                     )
    ds_analysis["threshold_regional"] = ds_analysis["threshold_regional"].where(abs(ds_analysis['ratio_ETap_rolling_regional']) <= threshold, True)
    return ds_analysis


def define_decision_thresholds(ds_analysis):
    pass
    

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
                    index,
                    out_with_IRR,
                    out_baseline,
                    prop='sw',
                    ):

    sw2plot_with_IRR =  out_with_IRR[prop].iloc[:,index].values
    sw2plot_baseline =  out_baseline[prop].iloc[:,index].values
    
    ax.plot(
            out_with_IRR['times'],
            sw2plot_with_IRR,
            label='Irrigated',
            marker='.',
            color='blue'
            )
    ax.plot(
            out_with_IRR['times'],
            sw2plot_baseline,
            label='Baseline',
            marker='.',
            color='red',
            )   
    
    ax.legend()
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
                 index,
                 out_with_IRR,
                 out_baseline,
                 ETp,
                 axs,
                 **kwargs
                 ):
    
   
    simu.show_input('atmbc',ax=axs[0])
    # simu_with_IRR.show_input('atmbc',ax=axs[0])
    
    timeIrr_sec = None
    if 'timeIrr_sec' in kwargs:
        timeIrr_sec = kwargs.pop('timeIrr_sec')

    scenario = None
    if 'scenario' in kwargs:
        scenario = kwargs.pop('scenario')


    utils.plot_in_subplot(
                            axs[1],
                            index,
                            out_with_IRR,
                            out_baseline,
                            prop='sw',
                            )
    
    
    utils.plot_in_subplot(
                            axs[2],
                            index,
                            out_with_IRR,
                            out_baseline,
                            prop='psi',
                            )
    
    # ETa1d_index, ETa1d_with_IRR, ETa1d_baseline = find_irr_surface_node(index,
    #                                                               out_with_IRR,
    #                                                               out_baseline
    #                                                               )
    
    ETa1d_index = np.where(out_with_IRR['ETa']['SURFACE NODE']==index[0])[0]
    ETa1d_with_IRR = out_with_IRR['ETa']['ACT. ETRA'].iloc[ETa1d_index[1:]]
    ETa1d_baseline = out_baseline['ETa']['ACT. ETRA'].iloc[ETa1d_index[1:]]
    axs[0].set_xlabel('')
    axs[1].set_xlabel('')

    indexplot = (3)
    axs[indexplot].plot(out_baseline['ETa'].time_sec.unique()[1:],
                        ETa1d_with_IRR,
                        label='Irr',
                        color='blue',
                        marker='*'
                        )
    axs[indexplot].plot(out_baseline['ETa'].time_sec.unique()[1:],
                        ETa1d_baseline,
                        label='baseline',
                        color='red',
                        marker='*'
                        )
    
    axs[indexplot].axhline(y=abs(ETp), 
                           color='k', 
                           linestyle='--', 
                           label='ETp')
    axs[indexplot].legend()
    axs[indexplot].set_xlabel('time')
    axs[indexplot].set_ylabel('ETa (m/s)')

    if timeIrr_sec is not None:
        axs[indexplot].axvline(x=timeIrr_sec, 
                               color='r', 
                               linestyle='--', 
                               label='Start Irr.')
        axs[indexplot].axvline(x=timeIrr_sec + scenario['irr_length'], 
                               color='r', 
                               linestyle='--', 
                               label='End Irr.')
    
    # Get current tick positions in seconds
    xticks = axs[-1].get_xticks()
    
    # Convert tick positions to days
    xtick_labels = seconds_to_days(xticks)
    
    # Update x-axis tick labels
    axs[-1].set_xticklabels([f'{int(day)}' for day in xtick_labels])
    
    # Set x-axis label
    axs[-1].set_xlabel('Time (days)')
    
    
 
def set_index_IN_OUT_irr_area(simu,maxDEM=1):
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
        
def export_tif2netcdf(pathTif2read='../prepro/Majadas'):
    
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
    ETp.to_netcdf('../prepro/ETp_Majadas.netcdf')
    RAIN = xr.concat(rain,dim='time')
    RAIN.to_netcdf('../prepro/RAIN_Majadas.netcdf')
    ETa = xr.concat(ETa_l,dim='time')
    ETa.to_netcdf('../prepro/ETa_Majadas.netcdf')

def read_prepo_EO_datasets(fieldsite='Majadas'):
    ETa_ds = xr.open_dataset(f'../prepro/ETa_{fieldsite}.netcdf')
    ETa_ds = ETa_ds.rename({"__xarray_dataarray_variable__": "ETa"})
    ETp_ds = xr.open_dataset(f'../prepro/ETp_{fieldsite}.netcdf')
    ETp_ds = ETp_ds.rename({"__xarray_dataarray_variable__": "ETp"})
    RAIN_ds = xr.open_dataset(f'../prepro/RAIN_{fieldsite}.netcdf')
    RAIN_ds = RAIN_ds.rename({"__xarray_dataarray_variable__": "RAIN"})
    CLC_ds = xr.open_dataset(f'../prepro/CLCover_{fieldsite}.netcdf')

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
    df_fort777_select_t_xr = df_fort777.set_index(['time','X','Y']).to_xarray()
    df_fort777_select_t_xr = df_fort777_select_t_xr.rio.set_spatial_dims('X','Y')
    
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
