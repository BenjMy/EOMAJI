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


import numpy as np
# from pyCATHY.plotters import cathy_plots as cplt
# from pyCATHY.importers import cathy_outputs as out_CT
# import pyCATHY.meshtools as msh_CT
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
import rioxarray as rxr

#%%
scenario_nb = 0
sc = load_scenario(scenario_nb)

#%% Paths
prj_name = 'EOMAJI_' + str(scenario_nb)
# figpath = os.path.join('../figures/',prj_name)
figpath = Path('../figures/scenario' + str(scenario_nb))

# Create the directory if it doesn't exist
figpath.mkdir(parents=True, exist_ok=True)
sc['figpath'] = figpath
#%% Simulate with irrigation atmospheric boundary conditions
# ----------------------------------------------------------
simu_with_IRR, sc_out = scenarii2pyCATHY.setup_cathy_simulation(
                                            prj_name=prj_name,
                                            scenario=sc,
                                            with_irrigation=True,
                                            )



sc['irr_time_index']
sc['irr_length']
sc['irr_flow']




#%% Simulate with NO irrigation 
# -----------------------------
simu_baseline, sc_out_baseline = scenarii2pyCATHY.setup_cathy_simulation(
                                                        prj_name=prj_name, 
                                                        scenario=sc,
                                                        with_irrigation=False,
                                                   )

#%%
# atmbc_df = simu_with_IRR.read_inputs('atmbc')
sc['irr_time_index']


net_irr_solution = sc['irr_flow']*sc['irr_length']

#%% find mesh index IN and OUT irrigation area

index_irrArea, index_out_irrArea = utils.set_index_IN_OUT_irr_area(simu_with_IRR)

(region_domain, 
 local_domain, 
 dem, 
 layers, 
 zones, 
 ETp_scenario) = scenarii2pyCATHY.prepare_scenario(sc)


(center_zones, 
 start_local_x, 
 end_local_x, 
 start_local_y, 
 end_local_y) =  scenarii2pyCATHY.get_irr_coordinates(local_domain,
                                                     region_domain,
                                                     dem,
                                                     )
# np.shape(dem)
                                                      
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

#%%
# ff
ds_analysis_EO['ACT. ETRA'].plot.imshow(x="X", y="Y", 
                                              col="time", 
                                              col_wrap=4,
                                              )
plt.suptitle('ETa with irr.')
plt.savefig(os.path.join(figpath,'ETa_withIRR_spatial_plot.png'),
            dpi=300,
            )

#%%
ds_analysis_baseline['ACT. ETRA'].plot.imshow(x="X", y="Y", 
                                              col="time", 
                                              col_wrap=4,
                                              )
plt.suptitle('ETa baseline')
plt.savefig(os.path.join(figpath,'ETa_baseline_spatial_plot.png'),
            dpi=300,
            )


# ds_analysis_baseline['ACT. ETRA'].time

#%% Create an xarray dataset with all the necessery variables ETp, ETa, ...

ds_analysis_EO = utils.compute_ratio_ETap_local(ds_analysis_EO)
ds_analysis_baseline = utils.compute_ratio_ETap_local(ds_analysis_baseline)

#%% 

ds_analysis_EO = utils.compute_regional_ETap(ds_analysis_EO,
                                       window_size_x=sc['ETp_window_size_x']
                                       )
ds_analysis_baseline = utils.compute_regional_ETap(ds_analysis_baseline,
                                             window_size_x=sc['ETp_window_size_x']
                                             )    
ds_analysis_EO = utils.compute_ratio_ETap_regional(ds_analysis_EO)
ds_analysis_baseline = utils.compute_ratio_ETap_regional(ds_analysis_baseline)
    
#%%
ds_analysis_EO = utils.compute_threshold_decision_local(ds_analysis_EO)

# IRRIGATION QUANTIFICATION
#%% Quantify volume applied
# -------------------------

# act_etra_variable.name
# ss
ds_analysis_baseline = ds_analysis_baseline.assign_coords(time=ds_analysis_EO['time'])
netIrr = abs(ds_analysis_EO['ACT. ETRA']) - abs(ds_analysis_baseline['ACT. ETRA'])
# netIrr = ds_analysis_baseline['ACT. ETRA']
# netIrr = netIrr.rename({netIrr.name: 'net Irrigation (m/s)'})
netIrr = netIrr.rename('netIrr (m/s)')

netIrr_cumsum = netIrr.cumsum('time')
netIrr_cumsum = netIrr_cumsum.rename('netIrr cumsum (m/s)')

netIrr.plot.imshow(x="X", y="Y", col="time", col_wrap=4)
plt.savefig(os.path.join(figpath,'netIrr_spatial_plot.png'))

#%% 

netIrr_cumsum = netIrr.cumsum('time')
netIrr_cumsum = netIrr_cumsum.rename('netIrr cumsum (m/s)')

netIrr_cumsum.plot.imshow(x="X", y="Y", col="time", col_wrap=4)
plt.savefig(os.path.join(figpath,'netIrr_cumsum_spatial_plot.png'))

#%% Plot histogram of daily net irrigation/rain versus real


Net_irr_IN_1D = netIrr.sel(X=500, 
                           Y=500, 
                           method='nearest'
                           )

df_Net_irr_IN_1D = Net_irr_IN_1D.to_dataframe(name='netIrr').reset_index()

fig, ax = plt.subplots()
# Plotting histogram
ax.bar(df_Net_irr_IN_1D['time'].dt.days, 
       df_Net_irr_IN_1D['netIrr'], 
       color='skyblue', 
       edgecolor='k', 
       alpha=0.7
       )



    
if 't_irr_start' in sc:
    
    # ax.axhline(y=sc['irr_flow'], 
    #             color='r', 
    #             linestyle='--', 
    #             label='irr_flow'
    #             )
 
    ax.axvline(x=sc['t_irr_start']/86000, 
                color='r', 
                linestyle='--', 
                label='Start Irr.'
            )
    ax.axvline(x=sc['t_irr_stop']/86000, 
                color='r', 
                linestyle='--', 
                label='End Irr.'
            )
    
    
# ss
ax.set_ylabel('netIrr (m/s)')
ax.set_xlabel('Day')
ax.set_title('Histogram of netIrr values over time')
plt.savefig(os.path.join(figpath,'netIrr_bar_plot.png'))

#%% Plot histogram of daily cumsum net irrigation/rain versus real


Net_irr_IN_1D = netIrr.sel(X=500, 
                           Y=500, 
                           method='nearest'
                           )

df_Net_irr_IN_1D = Net_irr_IN_1D.to_dataframe(name='netIrr').reset_index()

fig, ax = plt.subplots()
# Plotting histogram
ax.bar(df_Net_irr_IN_1D['time'].dt.days, 
       df_Net_irr_IN_1D['netIrr'].cumsum(), 
       color='skyblue', 
       edgecolor='k', 
       alpha=0.7
       )

df_Net_irr_IN_1D['netIrr'].cumsum().max()

ax.set_ylabel('netIrr (m/s)')
ax.set_xlabel('Day')
ax.set_title('Histogram of netIrr values over time')
plt.savefig(os.path.join(figpath,'netIrr_cumsum_bar_plot.png'))

#%%

df_atmbc = simu_with_IRR.read_inputs('atmbc')
raster_irr_zones, hd_irr_zones = simu_with_IRR.read_inputs('zone')
np.unique(raster_irr_zones)

padded_zones, padded_zones_1d = scenarii2pyCATHY.pad_zone_2mesh(raster_irr_zones)
# len(padded_zones_1d)

grid3d = simu_with_IRR.grid3d['mesh3d_nodes']
grid2d_surf = grid3d[0:int(simu_with_IRR.grid3d['nnod'])]
# len(np.unique(grid2d_surf[:,0]))


df_atmbc_t0 = df_atmbc[df_atmbc.time==0]

nnod = int(simu_with_IRR.grid3d['nnod'])
ntimes = len(df_atmbc.time.unique())
nodeIds = np.hstack([np.arange(0,nnod)]*ntimes)
zones_irr_bool = (padded_zones_1d == 2)
np.sum(zones_irr_bool)
zones_irr_bool_all_times = np.hstack([zones_irr_bool]*ntimes)
df_atmbc.insert(2,
                'nodeId',
                nodeIds
                )
df_atmbc.insert(3,
                'Irr_bool',
                zones_irr_bool_all_times
                )
np.sum(zones_irr_bool_all_times)
df_atmbc_irr = df_atmbc[df_atmbc['Irr_bool']==True]

# Net_irr_IN_1D.time
df_atmbc_irr['timeDelta'] = pd.to_timedelta(df_atmbc_irr['time'], 
                                            unit='s'
                                            )
df_atmbc['timeDelta'] = pd.to_timedelta(df_atmbc['time'], 
                                            unit='s'
                                            )

df_atmbc_irr_daily = df_atmbc_irr.set_index('timeDelta').resample('D').sum()
df_atmbc_daily = df_atmbc.set_index('timeDelta').resample('D').sum()
# df_atmbc = df_atmbc + 1e-9
# ax.bar(df_atmbc_irr_daily.index.days, 
#        df_atmbc_irr_daily['value'], 
#        color='red', 
#        edgecolor='k', 
#        alpha=0.1,
#        )
ax.bar(df_atmbc_daily.index.days, 
       df_atmbc_daily['value'], 
       color='b', 
       edgecolor='k', 
       alpha=0.1,
       )

sc['irr_length']
sc['irr_flow']



#%%


fig.savefig(os.path.join(figpath,'bar_plot_netIrr.png'))
