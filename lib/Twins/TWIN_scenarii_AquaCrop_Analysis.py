#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 12:38:06 2024
"""

# from aquacrop.utils import prepare_weather, get_filepath
# from aquacrop import AquaCropModel, Soil, Crop, InitialWaterContent, IrrigationManagement
# from aquacrop.utils import prepare_weather, get_filepath

import xarray as xr
import rioxarray as rxr
from pathlib import Path
import Majadas_utils
import matplotlib.pyplot as plt
import pandas as pd

dataPath = Path('../data/Spain/Spain_ETp_Copernicus_CDS/')
from TWIN_scenarii import load_scenario
import scenarii2pyCATHY
import argparse
import utils
import numpy as np
import os
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import datetime
from shapely.geometry import box

#%%
def get_cmd():
    parse = argparse.ArgumentParser()
    process_param = parse.add_argument_group('process_param')
    process_param.add_argument('-run_process', 
                               type=int, 
                               help='run_process',
                               default=0, 
                               required=False
                               ) 
    process_param.add_argument('-scenario_nb', 
                               type=int, 
                               help='scenario_nb',
                               default=0, # only 1 patch of irrigation
                               required=False
                               ) 
    process_param.add_argument('-weather_scenario', 
                               type=str, 
                               help='weather_scenario',
                                default='reference', 
                               # default='plus20p_tp', 
                               required=False
                               )     
    process_param.add_argument('-SMT', 
                               type=int, 
                               help='SMT %TAW (total available water',
                               default=70, # only 1 patch of irrigation
                               required=False
                               )     
    args = parse.parse_args()
    return(args)    

args = get_cmd() 
figpath = Path(f'../figures/scenario_AquaCrop_sc{args.scenario_nb}_weather_{args.weather_scenario}')
figpath.mkdir(parents=True, exist_ok=True)
#%%
# run_process = True # don't rerun the hydro model
# scenario_nb = 3
sc = load_scenario(args.scenario_nb)
prj_name = f'EOMAJI_AquaCrop_sc{args.scenario_nb}_weather_{args.weather_scenario}_SMT_{args.SMT}' 
sc['figpath'] = figpath


#%% RELOAD results

grid_xr_with_IRR = xr.open_dataset(f'../prepro/grid_xr_EO_{args.scenario_nb}.netcdf')
grid_xr_baseline = xr.open_dataset(f'../prepro/grid_xr_baseline_{args.scenario_nb}.netcdf')
ds_analysis_EO = xr.open_dataset(f'../prepro/ds_analysis_EO_{args.scenario_nb}.netcdf')
ds_analysis_baseline = xr.open_dataset(f'../prepro/ds_analysis_baseline_{args.scenario_nb}.netcdf')

#%%

time_sel = np.arange(0,len(grid_xr_with_IRR.time),10)
grid_xr_with_IRR['irr_daily'].isel(time=time_sel).plot.imshow(x="x", y="y", 
                                                              col="time", 
                                                              col_wrap=4
                                                              )
# plt.title('ETp EO')
plt.savefig(os.path.join(figpath,'irr_daily_aquacrop.png'),
            dpi=300,
            )
# grid_xr_with_IRR['irr_daily'].sum()
# grid_xr_with_IRR['rain_daily'].sum()

# time_sel = np.arange(0,len(grid_xr_with_IRR.time),10)
grid_xr_with_IRR['rain_daily'].plot.imshow(x="x", y="y", 
                                            col="time", 
                                            col_wrap=4
                                            )
# plt.title('ETp EO')
plt.savefig(os.path.join(figpath,'rain_daily_aquacrop.png'),
            dpi=300,
            )



# IRRIGATION DELIMITATION
# --------------------------------------------------
#%% Read and plot outputs (scenario with irrigation)
# --------------------------------------------------
print('Read outputs - can take a while')

from pyCATHY import CATHY
simu_with_IRR = CATHY(dirName='../WB_twinModels/', 
                      prj_name=prj_name
                    )
           
out_with_IRR = utils.read_outputs(simu_with_IRR)
out_baseline = utils.read_outputs(simu_baseline)

#%%


#%%
mask_IN = utils.get_mask_IN_patch_i(grid_xr_with_IRR['irrigation_map'],
                              patchid=2
                              )
mask_OUT = utils.get_mask_OUT(grid_xr_with_IRR['irrigation_map'],
                              )

fig, axs = plt.subplots(2,1,sharex=True,figsize=(12,9))
utils.plot_july_rain_irr(sc['datetime'], 
                         grid_xr_with_IRR, 
                         mask_IN,
                         axs=axs
                         )
fig.savefig(os.path.join(figpath,
                         'july_rain_irr.png'
                         ),
            dpi=300,
            )


#%% Plot evolution of ETa, SW and PSI INSIDE and OUTSIDE of the irrigation area 
# 1D + time
atmbc_df = simu_with_IRR.read_inputs('atmbc')
nb_irr_areas = len(np.unique(grid_xr_with_IRR.irrigation_map)) -1
(irr_patch_centers, 
 patch_centers_CATHY) = utils.get_irr_center_coords(irrigation_map=grid_xr_with_IRR['irrigation_map'])   
maxDEM = simu_with_IRR.grid3d['mesh3d_nodes'][:,2].max()
out_irr = np.where(grid_xr_with_IRR.irrigation_map==1)

start_date = sc['datetime'].values[0]




time_deltas = np.array([np.timedelta64(i, 'D') for i in range(len(grid_xr_with_IRR.time))])  # Example data
dates = [start_date + delta for delta in time_deltas]
dates_pd = pd.to_datetime(dates) 

fig, axs = plt.subplots(4,nb_irr_areas,
                        sharex=True,
                        sharey=False,
                        figsize=(7,5)
                        )

utils.plot_patches_irrigated_states(
                                    irr_patch_centers,
                                    patch_centers_CATHY,
                                    simu_with_IRR,
                                    maxDEM,
                                    grid_xr_with_IRR,
                                    sc,
                                    axs,
                                    out_with_IRR,
                                    out_baseline,
                                    dates_pd,
                                  )


fig.savefig(os.path.join(figpath,
                         'plot_1d_evol_irrArea.png'
                         ),
            dpi=300,
            )

#%%
fig, axs = plt.subplots(4,nb_irr_areas,
                        sharex=True,
                        sharey=False,
                        figsize=(15,5)
                        )

utils.plot_patches_NOirrgation_states(
                                        simu_with_IRR,
                                        out_irr,
                                        maxDEM,
                                        grid_xr_with_IRR,
                                        out_with_IRR,
                                        out_baseline,
                                        axs,
                                        sc,
                                        dates
                                    )

# utils.custum_axis_patches_states(axs,
#                                  irr_patch_centers,
#                                  )
                
# axs[0].set_title('WITHIN Irrigation Area')
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

fig.savefig(os.path.join(figpath,
                         'plot_1d_evol_OUTirrArea.png'
                         ),
            dpi=300,
            )

#%% Apply EO rules

# ds_analysis_EO_ruled = utils.apply_EO_rules(ds_analysis_EO,sc_EO)
ds_analysis_EO_ruled = utils.apply_EO_rules(ds_analysis_EO,
                                            sc_EO
                                            )

#%% irrigation_delineation

event_type = utils.irrigation_delineation(ds_analysis_EO_ruled) 
   
#%% Plot timeline 
# ncols = 4
# time_steps = event_type.time.size
# nrows = int(np.ceil(time_steps / ncols))  # Number of rows needed
# fig, axes = plt.subplots(nrows=nrows, ncols=ncols, 
#                          figsize=(15, nrows * 3)
#                          )
# utils.plot_irrigation_schedule(event_type,time_steps,fig,axes)
# plt.savefig(os.path.join(figpath, 'classify_events.png'))


event_type_node_IN = event_type.where(mask_IN, drop=True).mean(['x','y'])
event_type_node_OUT = event_type.where(mask_OUT, drop=True).mean(['x','y'])

#%%
import july
dates_dt = [pd.to_datetime(date).to_pydatetime() for date in ds_analysis_EO_ruled.datetime.values]
fig, axs = plt.subplots(2,1,sharex=True,figsize=(12,9))

im = july.heatmap(dates_dt, 
              event_type_node_IN.values, 
              title='Irrigation detected',
              # cmap=white_cmap,
              ax=axs[0],
              linewidth=1, 
              value_label=True,
              )

im = july.heatmap(dates_dt, 
              event_type_node_IN.values, 
              title='Irrigation detected',
              # cmap=white_cmap,
              ax=axs[1],
              linewidth=1, 
              value_label=True,
              )

utils.plot_july_rain_irr(sc['datetime'], 
                         grid_xr_with_IRR, 
                         mask_IN,
                         axs=axs,
                         )                    
fig.savefig(os.path.join(figpath,
                          'july_rain_irr.png'
                          ),
            dpi=300,
            )

#%% Vheck detection quality



mask_irr_solution = (grid_xr_with_IRR['irr_daily'].where(mask_IN, drop=True).mean(['x','y']) > 0).values
mask_irr_detection = (event_type_node_IN == 1).values
common_mask = mask_irr_solution & mask_irr_detection
nbIrr_detection =  np.sum(common_mask)
nbIrr_solution = np.sum(mask_irr_solution)
perc_detection = (nbIrr_detection/nbIrr_solution)*100

dates = sc['datetime']
mean_irr_daily = grid_xr_with_IRR['irr_daily'].where(mask_IN, drop=True).mean(['x', 'y']).values
mean_rain_daily = grid_xr_with_IRR['rain_daily'].mean(['x','y'])


# Create a list of colors based on the common_mask
colors = ['red' if is_common else 'blue' for is_common in common_mask]

# Plot the bar chart
fig, ax = plt.subplots()
ax.bar(dates, 
       mean_irr_daily, 
       color=colors)

ax.bar(dates, 
        mean_rain_daily, 
        color='g')

# Format the x-axis with datetime labels
ax.xaxis.set_major_locator(mdates.AutoDateLocator())  # Automatically set major ticks
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Set date format

# Rotate the x-axis labels for better readability
plt.xticks(rotation=45)

plt.xlabel('Date')
plt.ylabel('Irrigation Daily Mean')
plt.title(f'%of detected irr events={perc_detection}%')


# Add legend for red and blue bars
red_patch = mpatches.Patch(color='red', label='Detected')
blue_patch = mpatches.Patch(color='blue', label='Not Detected')
ax.legend(handles=[red_patch, blue_patch])

# Optional: Auto-adjust layout to prevent overlap
plt.tight_layout()

fig.savefig(os.path.join(figpath,
                          'Irrigation_detection.png'
                          ),
            dpi=300,
            )


# common_true_indices = common_mask.values[common_mask]

# ds_analysis_EO['event_type_node_IN'] = event_type_node_IN


# ds_analysis_quality = ds_analysis_EO
# ds_analysis_quality



# quality_check



#%%

# Detected irrigation events are further split into low, medium and high probability based on another set
# of thresholds. Since irrigation is normally applied on a larger area, the raster map with per-pixel
# irrigation events is cleaned up by removing isolated pixels in which irrigation was detected.

def set_probability_levels():
    print('to implement')
    pass
    
#%%

grid_xr_with_IRR.attrs = {}
grid_xr_with_IRR.to_netcdf(f'../prepro/grid_xr_EO_{args.scenario_nb}.netcdf')
grid_xr_baseline.attrs = {}
grid_xr_baseline.to_netcdf(f'../prepro/grid_xr_baseline_{args.scenario_nb}.netcdf')

ds_analysis_EO['time'] = ds_analysis_EO['time'].astype('timedelta64[D]')
ds_analysis_EO.to_netcdf(f'../prepro/ds_analysis_EO_{args.scenario_nb}.netcdf')

ds_analysis_baseline['time'] = ds_analysis_EO['time'].astype('timedelta64[D]')
ds_analysis_baseline.to_netcdf(f'../prepro/ds_analysis_baseline_{args.scenario_nb}.netcdf')



