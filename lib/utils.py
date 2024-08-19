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


import os
import pyvista as pv
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr


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

    
def compute_ratio_ETap_local(ds_analysis):
    ds_analysis["ratio_ETap_local"] = ds_analysis["ACT. ETRA"]/ds_analysis["ETp"]
    return ds_analysis

def compute_regional_ETap(ds_analysis,window_size_x=10):
    # (i.e. as an average change in all agricultural pixels within 10 km window)
    # Compute local ratio to check: 
    # b) There is input of water into the soil but due to rainfall (e.g. increase in regional ETa/p is over a
    # threshold and larger or similar to increase in local Eta/p)
    # c) There is input of water to the soil due to irrigation (e.g. increase in local ETa/p is over a
    # threshold and significantly larger than increase in regional ETa/p)
    # Compute the rolling mean on X and Y dimensions for the ETp variable
    for pp in ['ETp', 'ACT. ETRA']:
        rolling_mean = ds_analysis[pp].rolling(X=window_size_x, 
                                               Y=window_size_x, 
                                               center=True,
                                              ).mean()
        ds_analysis[pp+'_rolling_mean'] = rolling_mean
    return ds_analysis

def compute_ratio_ETap_regional(ds_analysis):
    ds_analysis["ratio_ETap_rolling_regional"] = ds_analysis['ACT. ETRA_rolling_mean']/ds_analysis["ETp_rolling_mean"]
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