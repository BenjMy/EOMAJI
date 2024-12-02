#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 12:38:06 2024
"""

# from aquacrop.utils import prepare_weather, get_filepath
from aquacrop import AquaCropModel, Soil, Crop, InitialWaterContent, IrrigationManagement
from aquacrop.utils import prepare_weather, get_filepath

from pyCATHY import CATHY

import xarray as xr
import rioxarray as rio
from pathlib import Path
import Majadas_utils
import contextily as cx
import matplotlib.pyplot as plt
import pandas as pd

dataPath = Path('../data/Spain/Spain_ETp_Copernicus_CDS/')
from DigTWIN_scenarii import load_scenario
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
                               default=7, # only 1 patch of irrigation
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
    process_param.add_argument('-ApplyEOcons', 
                               type=int, 
                               help='Applying EO cons',
                               default=None, # only 1 patch of irrigation
                               required=False
                               )   
    args = parse.parse_args()
    return(args)    

args = get_cmd() 
figpath = Path(f'../figures/scenario_AquaCrop_sc{args.scenario_nb}_weather_{args.weather_scenario}')
figpath.mkdir(parents=True, exist_ok=True)



#%% Read and plot outputs (scenario with irrigation)
# --------------------------------------------------
print('Read outputs - can take a while')
prj_name = f'EOMAJI_AquaCrop_sc{args.scenario_nb}_weather_{args.weather_scenario}_SMT_{args.SMT}_EOcons_{args.ApplyEOcons}' 

grid_xr_with_IRR = xr.open_dataset(f'../prepro/grid_xr_EO_AquaCrop_sc{args.scenario_nb}_weather_{args.weather_scenario}.netcdf')
grid_xr_baseline = xr.open_dataset(f'../prepro/grid_xr_baseline_AquaCrop_sc{args.scenario_nb}_weather_{args.weather_scenario}.netcdf')


simu_with_IRR = CATHY(dirName='../WB_twinModels/AQUACROP/', 
                     prj_name=prj_name + '_withIRR'
                     )
out_with_IRR = utils.read_outputs(simu_with_IRR)

simu_baseline = CATHY(dirName='../WB_twinModels/AQUACROP/', 
                     prj_name=prj_name
                     )
out_baseline = utils.read_outputs(simu_baseline)

#%%
# run_process = True # don't rerun the hydro model
# scenario_nb = 3
sc, _ = load_scenario(args.scenario_nb)
sc_EO = utils.check_and_tune_E0_dict(sc)

#%% Paths
sc['figpath'] = figpath
# sc['nb_days'] = len(out_with_IRR['ETa'].time.unique())

# (grid_xr_with_IRR, layers) = scenarii2pyCATHY.prepare_scenario( 
#                                                                 sc,
#                                                                 with_irrigation=True
#                                                                 )
# grid_xr_with_IRR['irr_daily'].sum(['x','y'])
# grid_xr_with_IRR['rain_daily'].sum(['x','y'])
 
# simu_with_IRR, grid_xr_with_IRR = scenarii2pyCATHY.setup_cathy_simulation(
#                                             prj_name=prj_name,
#                                             scenario=sc,
#                                             # ETp = ETp,
#                                             with_irrigation=True,
#                                             sc_EO=sc_EO,
#                                             # irr_time_index = 5,
#                                             # irr_flow = 5e-7 #m/s
#                                             )

# plt.close('all')
#%%

import os
import matplotlib.pyplot as plt

# start_date = sc['datetime'][0]
date_string = '2022-01-01 08:00:00.00000'
start_date = pd.to_datetime(date_string)

meanxy_irr_solution = grid_xr_with_IRR['irr_daily'].mean(['x', 'y'])
time_in_datetime = start_date + meanxy_irr_solution.time.data
fig, ax = plt.subplots()

ax.bar(time_in_datetime, 
       meanxy_irr_solution * (1e3 * 86400),  # Convert units to mm/day
       label='Daily Irrigation'
       )

# Add labels to the axes
ax.set_xlabel('Time')
ax.set_ylabel('Irrigation (mm/day)')

# Create a second y-axis to plot the cumulative sum
ax2 = ax.twinx()

# Plot the cumulative sum (cumsum) on the second y-axis
cumsum_irr = (meanxy_irr_solution * (1e3 * 86400)).cumsum()
ax2.plot(time_in_datetime, cumsum_irr, color='red', label='Cumulative Irrigation')

# Set the label for the cumulative sum axis
ax2.set_ylabel('Cumulative Irrigation (mm)')

# Save the figure
fig.savefig(os.path.join(figpath, 'plot_1d_net_irrArea.png'), dpi=300)

plt.show()

#%% Simulate with NO irrigation 
# -----------------------------

# (grid_xr_basline, layers) = scenarii2pyCATHY.prepare_scenario( 
#                                                             sc,
#                                                             with_irrigation=False
#                                                             )
#%%
# IRRIGATION DELIMITATION
# --------------------------------------------------

#%%

#%% 
# ETp = np.ones(np.shape(simu_with_IRR.DEM))*sc['ETp']
ds_analysis_EO = utils.get_analysis_ds(out_with_IRR['ETa'])
ds_analysis_baseline = utils.get_analysis_ds(out_baseline['ETa'])
# grid_xr_with_IRR = grid_xr_with_IRR.rename({'time_days': 'time'})
grid_xr_with_IRR = grid_xr_with_IRR.assign_coords(x=grid_xr_with_IRR['x'].astype('float64'),
                                                  y=grid_xr_with_IRR['y'].astype('float64'),
                                                  # time=pd.to_timedelta(grid_xr_with_IRR['time'], unit='D')
                                                  )
grid_xr_with_IRR_interp = grid_xr_with_IRR['ETp_daily'].interp_like(ds_analysis_EO, 
                                                                    method="linear"
                                                                    )
ds_analysis_EO["ETp"] = grid_xr_with_IRR_interp 
ds_analysis_EO["datetime"] = pd.to_datetime('01/01/2022') + ds_analysis_EO["time"]
# sc['datetime'] = pd.to_datetime(ds_analysis_EO["datetime"].values)

sc['datetime'] = pd.to_datetime(ds_analysis_EO["datetime"].dt.strftime('%d/%m/%Y'),
                                format='%d/%m/%Y').unique().sort_values()

# ds_analysis_EO = utils.add_ETp2ds(ETp,ds_analysis_EO)
# ds_analysis_baseline = utils.add_ETp2ds(ETp,ds_analysis_baseline)

time_sel = np.arange(0,len(ds_analysis_EO.time),50)
ds_analysis_EO['ETp'].isel(time=time_sel).plot.imshow(x="x", y="y", 
                                                      col="time", 
                                                      col_wrap=4
                                                      )
plt.title('ETp EO')
plt.savefig(os.path.join(figpath,'ETp.png'),
            dpi=300,
            )

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
# grid_xr_with_IRR['rain_daily'].sum(['x','y'])

#%% Plot evolution of ETa, SW and PSI INSIDE and OUTSIDE of the irrigation area 
# 1D + time
atmbc_df = simu_with_IRR.read_inputs('atmbc')
nb_irr_areas = len(np.unique(grid_xr_with_IRR.irrigation_map)) -1
(irr_patch_centers, 
 patch_centers_CATHY) = utils.get_irr_center_coords(irrigation_map=grid_xr_with_IRR['irrigation_map'])   
maxDEM = simu_with_IRR.grid3d['mesh3d_nodes'][:,2].max()
out_irr = np.where(grid_xr_with_IRR.irrigation_map==1)

start_date = sc['datetime'].values[0]


fig, axs = plt.subplots(4,nb_irr_areas,
                        sharex=True,
                        sharey=False,
                        figsize=(15,5)
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
                                    sc['datetime'],
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
                                    sc['datetime'],
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
                                        sc['datetime'],
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
#%%


# ds_analysis_EO_ruled['ratio_ETap_local_diff']

# ds_analysis_EO_ruled['ratio_ETap']


#%% irrigation_delineation
threshold_local=0.25
threshold_regional=0.10

# s
decision_ds, event_type = utils.irrigation_delineation(ds_analysis_EO_ruled,
                                                       threshold_local=threshold_local,
                                                       threshold_regional=threshold_regional,
                                                       time_window=5,
                                                       ) 

event_type_node_IN = event_type.where(mask_IN, drop=True).mean(['x','y'])
event_type_node_OUT = event_type.where(mask_OUT, drop=True).mean(['x','y'])

#%% Plot timeline 
# ncols = 4
# time_steps = event_type.time.size
# nrows = int(np.ceil(time_steps / ncols))  # Number of rows needed
# fig, axes = plt.subplots(nrows=nrows, ncols=ncols, 
#                          figsize=(15, nrows * 3)
#                          )
# utils.plot_irrigation_schedule(event_type,time_steps,fig,axes)
# plt.savefig(os.path.join(figpath, 'classify_events.png'))
dates_dt = [pd.to_datetime(date).to_pydatetime() for date in ds_analysis_EO_ruled.datetime.values]

mask_irr_solution = (grid_xr_with_IRR['irr_daily'].where(mask_IN, drop=True).mean(['x','y']) > 0).values
mask_rain_solution = (grid_xr_with_IRR['rain_daily'].where(mask_OUT, drop=True).mean(['x','y']) > 1e-8).values


non_zero_count = mask_rain_solution.sum()

mask_irr_detection = (event_type_node_IN == 1).values
common_mask = mask_irr_solution & mask_irr_detection
nbIrr_detection =  np.sum(common_mask)
nbIrr_solution = np.sum(mask_irr_solution)
perc_detection = (nbIrr_detection/nbIrr_solution)*100


decision_ds_ruled_node_IN = decision_ds.where(mask_IN, drop=True).mean(['x','y'])
decision_ds_ruled_node_OUT = decision_ds.where(mask_OUT, drop=True).mean(['x','y'])
decision_ds_ruled_node_IN = decision_ds_ruled_node_IN.assign_coords(datetime=sc['datetime'])
decision_ds_ruled_node_OUT = decision_ds_ruled_node_OUT.assign_coords(datetime=sc['datetime'])

dates = sc['datetime']
mean_irr_daily = grid_xr_with_IRR['irr_daily'].where(mask_IN, drop=True).mean(['x', 'y']).values
mean_rain_daily = grid_xr_with_IRR['rain_daily'].mean(['x','y'])

# # Create a list of colors based on the common_mask
colors = ['green' if is_common else 'red' for is_common in common_mask]


#%%


def plot_local_regional_time_serie(axs, ax2):
    
    axs[1].plot(decision_ds_ruled_node_IN.datetime,
        decision_ds_ruled_node_IN.ratio_ETap_local_time_avg,
          color='k', 
          linestyle='-',
          marker='.'
        )
    ax2.plot(decision_ds_ruled_node_IN.datetime,
            decision_ds_ruled_node_IN.ratio_ETap_regional_spatial_avg_time_avg,
              color='grey', 
              linestyle='-',
              marker='.'
            )
    
    
    ax2.scatter(decision_ds_ruled_node_IN.datetime[(decision_ds_ruled_node_IN.condRain>0).values],
            decision_ds_ruled_node_IN.ratio_ETap_local_time_avg[(decision_ds_ruled_node_IN.condRain>0).values],
              color='orange', 
              marker='v',
              s=100
            )
    
    ax2.scatter(decision_ds_ruled_node_IN.datetime[mask_rain_solution],
            decision_ds_ruled_node_IN.ratio_ETap_local_time_avg[mask_rain_solution],
              color='red', 
              marker='*'
            )
    
    ax2.scatter(decision_ds_ruled_node_IN.datetime[(decision_ds_ruled_node_IN.condIrrigation>0).values],
            decision_ds_ruled_node_IN.ratio_ETap_local_time_avg[(decision_ds_ruled_node_IN.condIrrigation>0).values],
              color='lightgreen', 
              marker='v',
              s=100,
              label='Irr. Detected'
            )
    ax2.scatter(decision_ds_ruled_node_IN.datetime[mask_irr_solution],
                decision_ds_ruled_node_IN.ratio_ETap_local_time_avg[mask_irr_solution],
                color='darkgreen', 
                marker='*',
                label='Irr. TRUE'
            )
    
#%%
# Plot the bar chart
fig, axs = plt.subplots(2,1,
                        figsize=(15,5),
                        sharex=True
                        )

utils.plot_atmbc_rain_irr_events(axs[0],
                                dates,
                                mean_irr_daily,
                                mean_rain_daily, 
                                perc_detection,
                                colors,
                                )


ax2 = axs[1].twinx()

plot_local_regional_time_serie(axs, ax2)

                                                    
# Adding horizontal lines for the thresholds
ax2.axhline(y=threshold_local, color='green', linestyle='--', label='threshold_local')
ax2.axhline(y=threshold_regional, color='red', linestyle='--', label='threshold_regional')

# Set labels for each y-axis
axs[1].set_ylabel('Local ETap \n Ratio Difference', color='green')
ax2.set_ylabel('Regional ETap \n Ratio Difference', color='red')

# Add legends for both plots
axs[1].legend(loc='upper left')
# ax2.legend(loc='upper right')
ax2.legend(loc='upper left', bbox_to_anchor=(1.15, 0.5), borderaxespad=0.)

# Annotation to explain the event of water input due to rainfall
fig.text(0.72, 0.5, 
         '''
             Rainfall=  \n d(regional ETa/p) > threshold \n d(regional ETa/p)>= d(local Eta/p)  \n  \n 
             Irr=  \n d(local ETa/p) > threshold \n d(local ETa/p)>> d(regional ETa/p)  \n  \n 
         ''',
         # ha='left', va='center', 
         fontsize=10, color='k'
         )
# Adjust plot to ensure the annotation is visible
plt.subplots_adjust(right=0.65)  # Make space for the annotation


# ax3.set_ylabel('ratio_ETap_local_diff', color='black')
# ax3.set_ylim([0,10])
# decision_ds_ruled_node_IN.variables
fig.savefig(os.path.join(figpath,
                          'irrigation_detection.png'
                          ),
            dpi=300,
            )



#%%
import july
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

#%%
plt.close('all')

# # Detected irrigation events are further split into low, medium and high probability based on another set
# # of thresholds. Since irrigation is normally applied on a larger area, the raster map with per-pixel
# # irrigation events is cleaned up by removing isolated pixels in which irrigation was detected.

# def set_probability_levels():
#     print('to implement')
#     pass


# IRRIGATION QUANTIFICATION
#%% Quantify volume applied
# -------------------------
ds_analysis_baseline = ds_analysis_baseline.assign_coords(time=ds_analysis_EO['time'])
netIrr = abs(ds_analysis_EO['ACT. ETRA']) - abs(ds_analysis_baseline['ACT. ETRA'])
netIrr = netIrr.rename('netIrr (m/s)')
netIrr_cumsum = netIrr.cumsum('time')
netIrr_cumsum = netIrr_cumsum.rename('netIrr cumsum (m/s)')
# netIrr.plot.imshow(x="x", y="y", col="time", col_wrap=4)
plt.savefig(os.path.join(figpath,'netIrr_spatial_plot.png'))

#%% Plot histogram of daily net irrigation/rain versus real
plt.close('all')
maxDEM = simu_with_IRR.grid3d['mesh3d_nodes'][:,2].max()
nb_irr_areas = 1
# ETp = np.ones(np.shape(simu_with_IRR.DEM))*sc['ETp']

ETp = grid_xr_baseline['ETp_daily'].mean(dim=['x','y'])
# ETp = ds_analysis_EO["ETp"].mean(dim=['x','y'])
fig, axs = plt.subplots(3,nb_irr_areas+1,
                        sharex=True,
                        sharey=False,
                        figsize=(16,6)
                        )
utils.plot_accounting_summary_analysis(
                                        axs,
                                         irr_patch_centers,
                                         patch_centers_CATHY,
                                         netIrr,
                                         simu_with_IRR,
                                         maxDEM,
                                         out_with_IRR,
                                         out_baseline,
                                         ETp,
                                         grid_xr_with_IRR,
                                    )
fig.savefig(os.path.join(figpath,
                         'plot_accounting_summary_analysis.png'
                         ),
            dpi=300,
            )
#%%

def irrigation_accounting(ds_analysis_EO,
                          ds_analysis_baseline
                          ):
    netIrr = abs(ds_analysis_EO['ACT. ETRA']) - abs(ds_analysis_baseline['ACT. ETRA'])
    netIrr_cumsum = netIrr.cumsum('time')
    netIrr_cumsum = netIrr_cumsum.rename('netIrr cumsum (m/s)')




#%%

mask_irr_solution


meanxy_netIrr_estimated = netIrr.mean(dim=['x','y'])

max_cum_sum_Year = grid_xr_with_IRR['irr_daily'].cumsum('time').sum(dim=['x','y']).max()
total_net_Year = netIrr_cumsum.sum(dim=['x','y']).max()

time_net_datetime = start_date + netIrr.time.data


# fig, axs = plt.subplots(2,1,
#                         figsize=(15,5),
#                         sharex=True
#                         )
# ax2 = axs[1].twinx()
# plot_local_regional_time_serie(axs, axs[0])
fig, ax = plt.subplots(figsize=(12,4))


ax.bar(time_in_datetime, 
       meanxy_irr_solution * (1e3 * 86400),  # Convert units to mm/day
       label='Daily Irrigation',
       color='lightblue'
       )
ax.bar(time_net_datetime, 
       meanxy_netIrr_estimated * (1e3 * 86400),  # Convert units to mm/day
       label='Estimated Irrigation',
       color='orange'
       )
# Add labels to the axes
ax.set_xlabel('Time')
ax.set_ylabel('Irrigation (mm/day)')

# Create a second y-axis to plot the cumulative sum
ax2 = ax.twinx()

# Plot the cumulative sum (cumsum) on the second y-axis
cumsum_irr_solution = (meanxy_irr_solution * (1e3 * 86400)).cumsum()
cumsum_irr_estimated = (meanxy_netIrr_estimated * (1e3 * 86400)).cumsum()
ax2.plot(time_in_datetime, cumsum_irr, color='darkblue', 
         label='Cumulative Irrigation')
ax2.plot(time_net_datetime, cumsum_irr_estimated, color='red',
         label='Cumulative Irrigation')

# Set the label for the cumulative sum axis
ax2.set_ylabel('Cumulative Irrigation (mm)')

# Save the figure
fig.savefig(os.path.join(figpath, 'plot_1d_net_irrArea.png'), dpi=300)

plt.show()



#%% Plot runoff
plt.close('all')
# simu_with_IRR.show(prop="hgraph")
# simu_with_IRR.show(prop="cumflowvol")

fig, ax = plt.subplots()
simu_with_IRR.show(prop="cumflowvol",ax=ax)
simu_baseline.show(prop="cumflowvol",ax=ax,
                   color='red')




#%%
# args.scenario_nb=0
# args.weather_scenario=0

ds_analysis_EO['time'] = ds_analysis_EO['time'].astype('timedelta64[D]')
ds_analysis_EO.to_netcdf(f'../prepro/EO_scenario_AquaCrop_sc{args.scenario_nb}_weather_{args.weather_scenario}.netcdf')

ds_analysis_baseline['time'] = ds_analysis_EO['time'].astype('timedelta64[D]')
ds_analysis_baseline.to_netcdf(f'../prepro/baseline_scenario_AquaCrop_sc{args.scenario_nb}_weather_{args.weather_scenario}.netcdf')

