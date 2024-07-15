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


import scenarii
import pyCATHY
import numpy as np
from pyCATHY import CATHY
from pyCATHY.plotters import cathy_plots as cplt
from pyCATHY.importers import cathy_outputs as out_CT
import pyCATHY.meshtools as msh_CT
import utils
import scenarii2pyCATHY
from scenarii import load_scenario

import os
import pyvista as pv
import matplotlib.pyplot as plt
import pandas as pd

plt.close('all')

from pathlib import Path
import xarray as xr

#%%
run_process = True # don't rerun the hydro model
scenario_nb = 0
sc = load_scenario(scenario_nb)

#%%
sc_df = pd.DataFrame.from_dict(sc,orient='index').T
sc_df
sc_df.to_csv('EOMAJI_synthetic_log.csv',index=False)
# sc_df.index.rename = 'Scenario'
sc_df = pd.read_csv('EOMAJI_synthetic_log.csv',index_col=False)

#%% Paths
prj_name = 'EOMAJI_' + str(scenario_nb)
# figpath = os.path.join('../figures/',prj_name)
figpath = Path('../figures/scenario' + str(scenario_nb))

# Create the directory if it doesn't exist
figpath.mkdir(parents=True, exist_ok=True)
sc['figpath'] = figpath
#%% Simulate with irrigation atmospheric boundary conditions
# ----------------------------------------------------------
simu_with_IRR, t_irr = scenarii2pyCATHY.setup_cathy_simulation(
                                            prj_name=prj_name,
                                            scenario=sc,
                                            # ETp = ETp,
                                            with_irrigation=True,
                                            # irr_time_index = 5,
                                            # irr_flow = 5e-7 #m/s
                                            )

if run_process:
    simu_with_IRR.run_processor(
                                IPRT1=2,
                                verbose=True,
                                DTMIN=1,
                                DTMAX=1e3,
                                DELTAT=1e2,
                            )
    
#%% Simulate with NO irrigation 
# -----------------------------
simu_baseline, _ = scenarii2pyCATHY.setup_cathy_simulation(
                                            prj_name=prj_name, 
                                            scenario=sc,
                                            with_irrigation=False,
                                       )


if run_process:
    simu_baseline.run_processor(
                                IPRT1=2,
                                verbose=True,
                                DTMIN=1,
                                DTMAX=1e3,
                                DELTAT=1e2,
                            )

# IRRIGATION DELIMITATION
# --------------------------------------------------
#%% Read and plot outputs (scenario with irrigation)
# --------------------------------------------------

out_with_IRR = utils.read_outputs(simu_with_IRR)
out_baseline = utils.read_outputs(simu_baseline)


#%% 

ETp = np.ones(np.shape(simu_with_IRR.DEM))*sc['ETp']
# fig, ax = plt.subplots()
# ax.imshow(ETp)


ds_analysis_EO = utils.get_analysis_ds(out_with_IRR['ETa'])
ds_analysis_baseline = utils.get_analysis_ds(out_baseline['ETa'])

ds_analysis_EO = utils.add_ETp2ds(ETp,ds_analysis_EO)
ds_analysis_baseline = utils.add_ETp2ds(ETp,ds_analysis_baseline)

ds_analysis_EO['ETp'].plot.imshow(x="X", y="Y", 
                                  col="time", 
                                  col_wrap=4,
                                  )
plt.suptitle('ETp with irr.')
plt.savefig(os.path.join(figpath,'ETp_spatial_plot.png'),
            dpi=300,
            )

ds_analysis_baseline['ETp'].plot.imshow(x="X", y="Y", 
                                        col="time", 
                                        col_wrap=4
                                        )
plt.title('ETp baseline')
plt.savefig(os.path.join(figpath,'ETp_baseline_spatial_plot.png'),
            dpi=300,
            )


#%% Plot evolution of ETa, SW and PSI INSIDE and OUTSIDE of the irrigation area 
# 1D + time


(
 index_irrArea, 
 index_out_irrArea,
 ) = utils.set_index_IN_OUT_irr_area(simu_with_IRR)

# simu_with_IRR.show_input('atmbc')
atmbc_df = simu_with_IRR.read_inputs('atmbc')
len(atmbc_df.time.unique())


fig, axs = plt.subplots(4,1,
                        sharex=True,
                        )
utils.plot_1d_evol(
                    simu_with_IRR,
                    index_irrArea,
                    out_with_IRR,
                    out_baseline,
                    np.mean(ETp),
                    axs,
                    scenario=sc,
                    timeIrr_sec = t_irr,
                )
axs[0].set_title('WITHIN Irrigation Area')
fig.savefig(os.path.join(figpath,
                         'plot_1d_evol_irrArea.png'
                         ),
            dpi=300,
            )

fig, axs = plt.subplots(4,1,
                        sharex=True
                        )
utils.plot_1d_evol(
                    simu_with_IRR,
                    index_out_irrArea,
                    out_with_IRR,
                    out_baseline,
                    np.mean(ETp),
                    axs,
                )
axs[0].set_title('OUTSIDE Irrigation Area')
fig.savefig(os.path.join(figpath,
                         'plot_1d_evol_outArea.png'
                         ),
            dpi=300,
            )


# path_results_withIRR = os.path.join(simu_with_IRR.workdir,
#                             simu_with_IRR.project_name,
#                             'vtk'
#                             )
# utils.plot_3d_SatPre(path_results_withIRR)




#%% Find the time when the irrigation is triggered 
# create the ratio between ETa and ETp
# np.shape(out_with_IRR['ETa']['ACT. ETRA'])


#%% Create an xarray dataset with all the necessery variables ETp, ETa, ...

# xr_analysis = ET_from_EO_xr.copy()
# xr_analysis["ETp"] = (("time", "X", "Y"), [padded_ETp]*len(xr_analysis.time))

# Compute local ratio to check: 
# a) There is no input of water into the soil (e.g. local ETa/p does not increase above a threshold)


ds_analysis_EO = utils.compute_ratio_ETap_local(ds_analysis_EO)
ds_analysis_baseline = utils.compute_ratio_ETap_local(ds_analysis_baseline)


ds_analysis_EO['ratio_ETap_local'].plot.imshow(x="X", y="Y", col="time", col_wrap=4)
plt.savefig(os.path.join(figpath,'ratioETap_withIRR_spatial_plot.png'),
            dpi=300,
            )

ds_analysis_baseline['ratio_ETap_local'].plot.imshow(x="X", y="Y", col="time", col_wrap=4)
plt.savefig(os.path.join(figpath,'ratioETap_baseline_spatial_plot.png'),
            dpi=300,
            )


#%% 


ds_analysis_EO = utils.compute_regional_ETap(ds_analysis_EO,
                                       window_size_x=sc['ETp_window_size_x']
                                       )
ds_analysis_baseline = utils.compute_regional_ETap(ds_analysis_baseline,
                                             window_size_x=sc['ETp_window_size_x']
                                             )



    
ds_analysis_EO = utils.compute_ratio_ETap_regional(ds_analysis_EO)
ds_analysis_baseline = utils.compute_ratio_ETap_regional(ds_analysis_baseline)



ds_analysis_EO['ratio_ETap_rolling_regional'].plot.imshow(x="X", y="Y", col="time", col_wrap=4)
plt.savefig(os.path.join(figpath,'ratioETap_regional_withIRR_spatial_plot.png'),
            dpi=300,
            )

ds_analysis_baseline['ratio_ETap_rolling_regional'].plot.imshow(x="X", y="Y", col="time", col_wrap=4)
plt.savefig(os.path.join(figpath,'ratioETap_regional_baseline_spatial_plot.png'),
            dpi=300,
            )


    
#%%
# a) There is no input of water into the soil (e.g. local ETa/p does not increase above a threshold)


# ds_analysis_EO
# def compute_ratio_ETap_regional(ds_analysis):
#     ds_analysis["ratio_ETap_rolling_regional"] = ds_analysis['ACT. ETRA_rolling_mean']/ds_analysis["ETp_rolling_mean"]
#     return ds_analysis


# ds_analysis_EO['ratio_ETap_local'].plot.imshow(x="X", y="Y", col="time", col_wrap=4)

 
ds_analysis_EO = utils.compute_threshold_decision_local(ds_analysis_EO,
                                                        threshold=sc['threshold_localETap']
                                                        # threshold=0.1
                                                        )

ds_analysis_EO = utils.compute_threshold_decision_regional(ds_analysis_EO,
                                                        threshold=sc['threshold_regionalETap']
                                                        )

# ds_analysis_EO['threshold_numeric'] = ds_analysis_EO['threshold'].astype(int)
# ds_analysis_EO['threshold_local'].plot.imshow(x="X", y="Y", col="time", col_wrap=4)
# ds_analysis_EO['ratio_ETap_local'].plot.imshow(x="X", y="Y", col="time", col_wrap=4)


event_type = xr.DataArray(0, 
                          coords=ds_analysis_EO.coords, 
                          dims=ds_analysis_EO.dims
                          )

condRain1 = ds_analysis_EO['threshold_regional']==True
condRain2 = ds_analysis_EO['ratio_ETap_rolling_regional'] >= ds_analysis_EO['ratio_ETap_local']
condRain = condRain1 & condRain2
event_type = event_type.where(~condRain, 1)

condIrrigation1 = ds_analysis_EO['threshold_local']==True
np.sum(condIrrigation1)
condIrrigation2 = ds_analysis_EO['ratio_ETap_local'] > 1.5*ds_analysis_EO['ratio_ETap_rolling_regional']
condIrrigation = condIrrigation1 & condIrrigation2

event_type = event_type.where(~condIrrigation, 2)

#%%
mapping = {'rain': 2, 'irrigation': 1, 'No input': 0}
# Custom colormap with discrete colors
# Custom colormap with discrete colors
cmap = plt.cm.colors.ListedColormap(['white', 'blue', 'red'])

# Plot using imshow with the custom colormap
plot = event_type.plot.imshow(x="X", y="Y", 
                              col="time", col_wrap=4, 
                              cmap=cmap)

# Get the current axes
ax = plt.gca()

# Create a colorbar with ticks and labels corresponding to mapping
cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), 
                    ax=ax, 
                    ticks=[
                            mapping['No input'], 
                            mapping['irrigation'], 
                            mapping['rain']
                            ]
                    )
cbar.ax.set_yticklabels(['No input', 'irrigation', 'rain'])

plt.savefig(os.path.join(figpath,'classify_events.png'))

#%%

# Use July
# import July

(region_domain, 
 local_domain, 
 dem, 
 layers, 
 zones, 
 ETp_scenario) = scenarii2pyCATHY.prepare_scenario(sc)


(center_zones, 
  start_local_idx, 
  end_local_idx, 
  start_local_idy, 
  end_local_idy) =  scenarii2pyCATHY.get_irr_coordinates(local_domain,
                                                      region_domain,
                                                      dem,
                                                      )
                                                       
                                                       
raster_irr_zones, hd_irr_zones = simu_with_IRR.read_inputs('zone')
np.unique(raster_irr_zones)

padded_zones, padded_zones_1d = scenarii2pyCATHY.pad_zone_2mesh(raster_irr_zones)
# len(padded_zones_1d)

# plt.imshow(padded_zones)
# grid3d = simu_with_IRR.grid3d['mesh3d_nodes']
# grid2d_surf = grid3d[0:int(simu_with_IRR.grid3d['nnod'])]

# event_type.values.shape
                                                      
import numpy as np
import matplotlib.pyplot as plt
import july
from july.utils import date_range
# import warnings!
# warnings.simplefilter("ignore", category=DeprecationWarning)

condx_irr_area = event_type.X[start_local_idx:end_local_idx]
condy_irr_area = event_type.X[start_local_idx:end_local_idx]

condx_irr_area_bool = event_type.X.isin(condx_irr_area)
condy_irr_area_bool = event_type.Y.isin(condy_irr_area)

irr_area_1 = event_type.where(condx_irr_area_bool & condy_irr_area_bool, 
                        drop=False
                        )
irr_area_1_mean_event_class = irr_area_1.mean(dim=["X", "Y"]).round()


dates = date_range("2020-01-01", "2020-12-31")
# data = np.random.randint(0, 14, len(dates))
# aa
# july.heatmap(dates, 
#              irr_area_1_mean_event_class.values, 
#              title='Event Classification', 
#              cmap='jet',
#              )

# test = july.calendar_plot(dates, 
#              irr_area_1_mean_event_class.values, 
#              title='Event Classification', 
#              cmap='jet',
#              # year_label=True,
#              # colorbar=False,
#              fontfamily="monospace",
#              fontsize=12,
#              )

july.heatmap(dates=dates, 
             data=irr_area_1_mean_event_class.values, 
             cmap='Pastel1',
             month_grid=True, 
             horizontal=True,
             value_label=False,
             date_label=False,
             weekday_label=True,
             month_label=True, 
             year_label=True,
             colorbar=True,
             fontfamily="monospace",
             fontsize=12,
             title=None,
             titlesize='large',
             dpi=300
             )


# test = july.calendar_plot(dates, 
#              irr_area_1_mean_event_class.values, 
#              title='Event Classification', 
#              cmap='jet',
#              # year_label=True,
#              # colorbar=False,
#              fontfamily="monospace",
#              fontsize=12,
#              )


# plt.color
plt.savefig(os.path.join(figpath,'events_calendar.png'))


#%%
# Custom colormap with white, blue, and red
cmap = plt.cm.colors.ListedColormap(['white', 'blue', 'red'])

# Create subplots for each time slice
fig, axs = plt.subplots(nrows=int(event_type.shape[0]/4), ncols=4, figsize=(8, 6),
                        sharex=True,
                        sharey=True)


time_values = event_type.time.values

# Calculate elapsed days and hours
elapsed_days = (time_values / np.timedelta64(1, 'D')).astype(int)
remaining_hours = ((time_values % np.timedelta64(1, 'D')) / np.timedelta64(1, 'h')).astype(int)

# Combine elapsed days and hours into a list
time_list = [f"{day} days, {hour} hours" for day, hour in zip(elapsed_days, remaining_hours)]


axs = axs.ravel()
for i, ax in enumerate(axs):
    # Plot each time slice using imshow
    ax.imshow(event_type[i].values, cmap=cmap)
    ax.set_title(f'{time_list[i]}',fontsize=6)
    if i == len(axs)-1:
        ax.set_xlabel('x [m]')
        ax.set_xlabel('y [m]')

    # ax.set_yticks([])

# Position the colorbar to the right of the subplots
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Adjust the position and size as needed
# cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), 
#                     cax=cbar_ax, 
#                     ticks=[mapping['No input'], 
#                            mapping['rain'], 
#                            mapping['irrigation']
#                            ]
#                     )
cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), 
                    cax=cbar_ax, 
                    ticks=[0,1,2]
                    )
cbar.ax.set_yticklabels(['No input', 'irrigation', 'rain'])

# plt.tight_layout()
plt.show()
    




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


#%% 
def compare_local_regional_ratios(ds_analysis):
    pass
# differenciation between rain and irrigation events
# --------------------------------------------------
# compute local vs regional ETa/ETp

def compute_rolling_time_mean(ds_analysis):
    ds_analysis.rolling(time=3).mean()
    return ds_analysis

#Since irrigation is normally applied on a larger area, 
# the raster map with per-pixel irrigation events 
# is cleaned up by removing isolated pixels in which irrigation was detected.




# fig, axs = plt.subplots(2,1, sharex=True)
# # plot and detect when irrigation has been trigerred
# xr_analysis.groupby('time').max().plot(y='ratio_ETap_local',ax=axs[0])
# axs[0].axhline(y= -0.6, 
#                color='k', 
#                linestyle='--', 
#                label='Threshold0.6'
#                )


# Plot threshold on the second subplot
# Since 'threshold' is boolean, we'll use a step plot to represent it
# xr_analysis.groupby('time').max().plot(y='threshold_numeric', 
#                                       ax=axs[1], 
#                                       drawstyle='steps-post', 
#                                       marker='s'
#                                       )

# axs[0].set_ylabel('ETa/ETp (m/s)')
# axs[0].set_xlabel('')
# axs[1].yaxis.set_ticklabels(['True', 'False'])
# fig.savefig(os.path.join(figpath,'Irrigation detection.png'))


#%%




