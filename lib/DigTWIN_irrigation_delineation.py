#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:50:23 2023
@author: ben

Project EOMAJI (Earth Observation system to Manage Africa’s food systems by Joint-knowledge of crop production and Irrigation digitization) 
ET-Based algoritms for net irrigation estimation. 

- Identifying irrigation events through monitoring of changes in ET. 

The ratio of actual to potential ET (ETa/p) should be used in order to avoid changes in ET
due to changes in weather (e.g. increased wind speed) or crop cover (e.g. quick development of
leaves) being attributed to irrigation. This ratio is closely related to root-zone water availability and
therefore is mainly influenced by irrigation or rainfall events.

This is achieved by first calculating the change in ETa/p between the time on which irrigation is to be detect and most recent previous time on which ET estimates are available. This change is calculated both locally (i.e. at individual pixel level) and regionally (i.e. as an average change in all agricultural pixels within 10 km window). 

The local and regional changes are then compared to a number of thresholds to try to detect if:
a) There is no input of water into the soil (e.g. local ETa/p does not increase above a threshold)
b) There is input of water into the soil but due to rainfall (e.g. increase in regional ETa/p is over a
threshold and larger or similar to increase in local Eta/p)
c) There is input of water to the soil due to irrigation (e.g. increase in local ETa/p is over a
threshold and significantly larger than increase in regional ETa/p)

Detected irrigation events are further split into low, medium and high probability based on another set
of thresholds. Since irrigation is normally applied on a larger area, the raster map with per-pixel
irrigation events is cleaned up by removing isolated pixels in which irrigation was detected.

""" 


import DigTWIN_scenarii
import pyCATHY
import numpy as np
from pyCATHY import CATHY
from pyCATHY.plotters import cathy_plots as cplt
from pyCATHY.importers import cathy_outputs as out_CT
import pyCATHY.meshtools as msh_CT
import utils
import scenarii2pyCATHY
from DigTWIN_scenarii import load_scenario

import os
import pyvista as pv
import matplotlib.pyplot as plt
import pandas as pd

plt.close('all')

from pathlib import Path
import xarray as xr
import argparse
import copy

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
                               default=1, 
                               required=False
                               ) 
    # process_param.add_argument('-WTD', type=float, help='WT height',
    #                     # default=100, required=False) 
    #                     # default=4, required=False) 
    #                     default=99, required=False) 
    args = parse.parse_args()
    return(args)    

args = get_cmd() 
#%%
# run_process = True # don't rerun the hydro model
# scenario_nb = 3
sc = load_scenario(args.scenario_nb)

#%%
sc_df = pd.DataFrame.from_dict(sc,orient='index').T
sc_df
sc_df.to_csv('EOMAJI_synthetic_log.csv',index=False)
# sc_df.index.rename = 'Scenario'
sc_df = pd.read_csv('EOMAJI_synthetic_log.csv',index_col=False)

#%% Paths
prj_name = 'EOMAJI_' + str(args.scenario_nb)
# figpath = os.path.join('../figures/',prj_name)
figpath = Path('../figures/scenario' + str(args.scenario_nb))

# Create the directory if it doesn't exist
figpath.mkdir(parents=True, exist_ok=True)
sc['figpath'] = figpath
#%% Simulate with irrigation atmospheric boundary conditions
# ----------------------------------------------------------

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
        
    return sc_EO

sc_EO = check_and_tune_E0_dict(sc)

simu_with_IRR, grid_xr_with_IRR = scenarii2pyCATHY.setup_cathy_simulation(
                                            prj_name=prj_name,
                                            scenario=sc,
                                            # ETp = ETp,
                                            with_irrigation=True,
                                            sc_EO=sc_EO,
                                            # irr_time_index = 5,
                                            # irr_flow = 5e-7 #m/s
                                            )
if args.run_process:
    simu_with_IRR.run_processor(
                                IPRT1=2,
                                verbose=True,
                                DTMIN=1,
                                DTMAX=1e3,
                                DELTAT=1e1,
                            )
# ee
plt.close('all')
#%% Simulate with NO irrigation 
# -----------------------------
simu_baseline, grid_xr_baseline = scenarii2pyCATHY.setup_cathy_simulation(
                                                     prj_name=prj_name, 
                                                     scenario=sc,
                                                     with_irrigation=False,
                                                )


if args.run_process:
    simu_baseline.run_processor(
                                IPRT1=2,
                                verbose=True,
                                DTMIN=1,
                                DTMAX=1e3,
                                DELTAT=1e2,
                            )

plt.close('all')

# IRRIGATION DELIMITATION
# --------------------------------------------------
#%% Read and plot outputs (scenario with irrigation)
# --------------------------------------------------

out_with_IRR = utils.read_outputs(simu_with_IRR)
out_baseline = utils.read_outputs(simu_baseline)


#%% 
ETp = np.ones(np.shape(simu_with_IRR.DEM))*sc['ETp']
ds_analysis_EO = utils.get_analysis_ds(out_with_IRR['ETa'])
ds_analysis_baseline = utils.get_analysis_ds(out_baseline['ETa'])

ds_analysis_EO = utils.add_ETp2ds(ETp,ds_analysis_EO)
ds_analysis_baseline = utils.add_ETp2ds(ETp,ds_analysis_baseline)

# ds_analysis_baseline['ETp'].plot.imshow(x="x", y="y", 
#                                         col="time", 
#                                         col_wrap=4
#                                         )
# plt.title('ETp baseline')
# plt.savefig(os.path.join(figpath,'ETp_baseline.png'),
#             dpi=300,
#             )


#%% Plot evolution of ETa, SW and PSI INSIDE and OUTSIDE of the irrigation area 
# 1D + time
atmbc_df = simu_with_IRR.read_inputs('atmbc')
nb_irr_areas = len(np.unique(grid_xr_with_IRR.irrigation_map)) -1
(irr_patch_centers, 
 patch_centers_CATHY) = utils.get_irr_center_coords(irrigation_map=grid_xr_with_IRR['irrigation_map'])   
# grid_xr_with_IRR.x 
maxDEM = simu_with_IRR.grid3d['mesh3d_nodes'][:,2].max()
out_irr = np.where(grid_xr_with_IRR.irrigation_map==1)


fig, axs = plt.subplots(4,nb_irr_areas+1,
                        sharex=True,
                        sharey=False,
                        )
    

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
    t_irr = first_non_zero_time_days*86400
    
    axs_idi = axs[:,i]
    utils.plot_1d_evol(
                        simu_with_IRR,
                        node_index,
                        out_with_IRR,
                        out_baseline,
                        np.mean(ETp),
                        axs_idi,
                        scenario=sc,
                        timeIrr_sec = t_irr,
                    )

node_index_OUT, _ = simu_with_IRR.find_nearest_node([out_irr[0][0],
                                                     out_irr[1][0],
                                                     maxDEM
                                                     ]
                                                    )
utils.plot_1d_evol(
                    simu_with_IRR,
                    node_index_OUT,
                    out_with_IRR,
                    out_baseline,
                    np.mean(ETp),
                    axs[:,-1],
                    scenario=sc,
                    timeIrr_sec = t_irr,
                )


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
    
    
            
# axs[0].set_title('WITHIN Irrigation Area')
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

fig.savefig(os.path.join(figpath,
                         'plot_1d_evol_irrArea.png'
                         ),
            dpi=300,
            )

      
#%% irrigation_delineation

event_type = utils.irrigation_delineation(ds_analysis_EO) 
   
#%% Plot timeline 
ncols = 4
time_steps = event_type.time.size
nrows = int(np.ceil(time_steps / ncols))  # Number of rows needed
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, 
                         figsize=(15, nrows * 3)
                         )
utils.plot_irrigation_schedule(event_type,time_steps,fig,axes)
plt.savefig(os.path.join(figpath, 'classify_events.png'))

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


#%% 
# ds_analysis_EO['time'] = ds_analysis_EO.time.astype('int32')


# # Specify the encoding
# encoding = {
#     'time': {
#         'dtype': np.int64,
#         # 'calendar': 'standard'  # Optional, depends on your data
#     }
# }

# When saving the dataset
# ds_analysis_baseline.to_netcdf('output_file.nc', encoding=encoding)

# ds_analysis_EO.to_netcdf(f'../WB_twinModels/scenario{args.scenario_nb}.netcdf')
# w

#%%

# Use July
# import July

# grid_xr, layers  = scenarii2pyCATHY.prepare_scenario(sc)


# (center_zones, 
#   start_local_idx, 
#   end_local_idx, 
#   start_local_idy, 
#   end_local_idy) =  scenarii2pyCATHY.get_irr_coordinates(local_domain,
#                                                       region_domain,
#                                                       dem,
#                                                       )
                                                       
                                                       
# raster_irr_zones, hd_irr_zones = simu_with_IRR.read_inputs('zone')
# np.unique(raster_irr_zones)

# padded_zones, padded_zones_1d = scenarii2pyCATHY.pad_zone_2mesh(raster_irr_zones)
# # len(padded_zones_1d)

# # plt.imshow(padded_zones)
# # grid3d = simu_with_IRR.grid3d['mesh3d_nodes']
# # grid2d_surf = grid3d[0:int(simu_with_IRR.grid3d['nnod'])]

# # event_type.values.shape
                                                      
# import numpy as np
# import matplotlib.pyplot as plt
# import july
# from july.utils import date_range
# # import warnings!
# # warnings.simplefilter("ignore", category=DeprecationWarning)

# condx_irr_area = event_type.X[start_local_idx:end_local_idx]
# condy_irr_area = event_type.X[start_local_idx:end_local_idx]

# condx_irr_area_bool = event_type.X.isin(condx_irr_area)
# condy_irr_area_bool = event_type.Y.isin(condy_irr_area)

# irr_area_1 = event_type.where(condx_irr_area_bool & condy_irr_area_bool, 
#                         drop=False
#                         )
# irr_area_1_mean_event_class = irr_area_1.mean(dim=["x", "y"]).round()


# dates = date_range("2020-01-01", "2020-12-31")
# # data = np.random.randint(0, 14, len(dates))
# # aa
# # july.heatmap(dates, 
# #              irr_area_1_mean_event_class.values, 
# #              title='Event Classification', 
# #              cmap='jet',
# #              )

# # test = july.calendar_plot(dates, 
# #              irr_area_1_mean_event_class.values, 
# #              title='Event Classification', 
# #              cmap='jet',
# #              # year_label=True,
# #              # colorbar=False,
# #              fontfamily="monospace",
# #              fontsize=12,
# #              )

# july.heatmap(dates=dates, 
#              data=irr_area_1_mean_event_class.values, 
#              cmap='Pastel1',
#              month_grid=True, 
#              horizontal=True,
#              value_label=False,
#              date_label=False,
#              weekday_label=True,
#              month_label=True, 
#              year_label=True,
#              colorbar=True,
#              fontfamily="monospace",
#              fontsize=12,
#              title=None,
#              titlesize='large',
#              dpi=300
#              )


# # test = july.calendar_plot(dates, 
# #              irr_area_1_mean_event_class.values, 
# #              title='Event Classification', 
# #              cmap='jet',
# #              # year_label=True,
# #              # colorbar=False,
# #              fontfamily="monospace",
# #              fontsize=12,
# #              )


# # plt.color
# plt.savefig(os.path.join(figpath,'events_calendar.png'))







# if 


# ds_analysis["threshold"] = (("time", "X", "Y"),
#                             np.ones(np.shape(ds_analysis['index']))*False
#                             )


# threshold3D = np.ones(np.shape(ds_analysis_EO['index']))*False
# dims = ('time', 'X', 'Y')
# threshold_xr = xr.DataArray(threshold3D, dims=dims)
# ds_analysis_EO['threshold'] = threshold_xr
# ds_analysis_EO.loc[abs(ds_analysis_EO['ratio_ETap_local']) < 0.6, 'threshold'] = True
# ds_analysis_EO['threshold_numeric'] = ds_analysis_EO['threshold'].astype(int)

