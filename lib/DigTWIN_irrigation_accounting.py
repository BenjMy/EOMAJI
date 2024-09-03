#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:50:23 2023
@author: ben

ET-based algorithms
The advantages of using ET over soil moisture is that ET is directly linked to plant transpiration
reacting to irrigation whereas soil moisture produces an indirect estimate, especially since satellite MW
systems only penetrate the topsoil (few cm) . In addition, the spatial resolution of EO optical sensors
is typically higher, with some orders of magnitude, than passive microwave.
Net irrigation is estimated based on the systematic evapotranspiration (ET) residuals between a
remote sensing ‐ based model and a calibrated hydrologic or SWB model that does not include an
irrigation scheme 47,49
I net ( t )=ET EO − ET baseline

""" 
import numpy as np
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
import argparse
from pyCATHY import CATHY


#%%
def get_cmd():
    parse = argparse.ArgumentParser()
    process_param = parse.add_argument_group('process_param')
    process_param.add_argument('-scenario_nb', 
                               type=int, 
                               help='scenario_nb',
                               default=1, 
                               required=False
                               ) 
    args = parse.parse_args()
    return(args)    

args = get_cmd() 
sc = load_scenario(args.scenario_nb)

#%% Paths
prj_name = 'EOMAJI_' + str(args.scenario_nb)
figpath = Path('../figures/scenario' + str(args.scenario_nb))

#%%
os.getcwd()
ds_analysis_EO = xr.open_dataset(f'../prepro/ds_analysis_EO_{args.scenario_nb}.netcdf',engine='scipy')
ds_analysis_baseline = xr.open_dataset(f'../prepro/ds_analysis_baseline_{args.scenario_nb}.netcdf',engine='scipy')
grid_xr_EO = xr.open_dataset(f'../prepro/grid_xr_EO_{args.scenario_nb}.netcdf',engine='scipy')
grid_xr_baseline = xr.open_dataset(f'../prepro/grid_xr_baseline_{args.scenario_nb}.netcdf',engine='scipy')

#%% Paths
prj_name = 'EOMAJI_' + str(args.scenario_nb)
figpath = Path('../figures/scenario' + str(args.scenario_nb))
sc['figpath'] = figpath

#%% 
simu_with_IRR = CATHY(dirName='../WB_twinModels/', 
                        prj_name=prj_name + '_withIRR' 
                         )
simu_baseline = CATHY(dirName='../WB_twinModels/', 
                        prj_name=prj_name
                         )
#%% find mesh index IN and OUT irrigation area
(irr_patch_centers, 
 patch_centers_CATHY) = utils.get_irr_center_coords(irrigation_map=grid_xr_EO['irrigation_map'])
                                                    
# IRRIGATION DELIMITATION
# --------------------------------------------------
#%% Read and plot outputs (scenario with irrigation)
# --------------------------------------------------
out_with_IRR = utils.read_outputs(simu_with_IRR)
out_baseline = utils.read_outputs(simu_baseline)

#%%  get_analysis_ds
ds_analysis_EO = utils.get_analysis_ds(out_with_IRR['ETa'])
ds_analysis_baseline = utils.get_analysis_ds(out_baseline['ETa'])
#%%
# ff
ds_analysis_EO['ACT. ETRA'].plot.imshow(x="x", y="y", 
                                        col="time", 
                                        col_wrap=4,
                                        )
plt.suptitle('ETa with irr.')
plt.savefig(os.path.join(figpath,'ETa_withIRR_spatial_plot.png'),
            dpi=300,
            )

#%%
ds_analysis_baseline['ACT. ETRA'].plot.imshow(x="x", y="y", 
                                              col="time", 
                                              col_wrap=4,
                                              )
plt.suptitle('ETa baseline')
plt.savefig(os.path.join(figpath,'ETa_baseline_spatial_plot.png'),
            dpi=300,
            )

# IRRIGATION QUANTIFICATION
#%% Quantify volume applied
# -------------------------
ds_analysis_baseline = ds_analysis_baseline.assign_coords(time=ds_analysis_EO['time'])
netIrr = abs(ds_analysis_EO['ACT. ETRA']) - abs(ds_analysis_baseline['ACT. ETRA'])
netIrr = netIrr.rename('netIrr (m/s)')
netIrr_cumsum = netIrr.cumsum('time')
netIrr_cumsum = netIrr_cumsum.rename('netIrr cumsum (m/s)')
netIrr.plot.imshow(x="x", y="y", col="time", col_wrap=4)
plt.savefig(os.path.join(figpath,'netIrr_spatial_plot.png'))

#%% 

netIrr_cumsum = netIrr.cumsum('time')
netIrr_cumsum = netIrr_cumsum.rename('netIrr cumsum (m/s)')

netIrr_cumsum.plot.imshow(x="x", y="y", col="time", col_wrap=4)
plt.savefig(os.path.join(figpath,'netIrr_cumsum_spatial_plot.png'))

#%% Plot histogram of daily net irrigation/rain versus real
plt.close('all')
maxDEM = simu_with_IRR.grid3d['mesh3d_nodes'][:,2].max()
nb_irr_areas = len(np.unique(grid_xr_EO.irrigation_map)) -1
# ETp = np.ones(np.shape(simu_with_IRR.DEM))*sc['ETp']

# ETp = grid_xr_EO['ETp_daily'].mean(dim=['x','y'])
ETp = grid_xr_EO['ETp_daily'].mean()
fig, axs = plt.subplots(3,nb_irr_areas+1,
                        sharex=True,
                        sharey=False,
                        figsize=(16,6)
                        )
grid_xr_EO['irr_daily']
# 3e-7*86400*1000

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
                                         grid_xr_EO,
                                    )
fig.savefig(os.path.join(figpath,
                         'plot_accounting_summary_analysis.png'
                         ),
            dpi=300,
            )

#%% Plot runoff
plt.close('all')
# simu_with_IRR.show(prop="hgraph")
# simu_with_IRR.show(prop="cumflowvol")

fig, ax = plt.subplots()
simu_with_IRR.show(prop="cumflowvol",ax=ax)
simu_baseline.show(prop="cumflowvol",ax=ax,
                   color='red')


#%%

# df_atmbc = simu_with_IRR.read_inputs('atmbc')
# raster_irr_zones, hd_irr_zones = simu_with_IRR.read_inputs('zone')
# np.unique(raster_irr_zones)

# padded_zones, padded_zones_1d = scenarii2pyCATHY.pad_zone_2mesh(raster_irr_zones)
# # len(padded_zones_1d)

# grid3d = simu_with_IRR.grid3d['mesh3d_nodes']
# grid2d_surf = grid3d[0:int(simu_with_IRR.grid3d['nnod'])]
# # len(np.unique(grid2d_surf[:,0]))


# df_atmbc_t0 = df_atmbc[df_atmbc.time==0]

# nnod = int(simu_with_IRR.grid3d['nnod'])
# ntimes = len(df_atmbc.time.unique())
# nodeIds = np.hstack([np.arange(0,nnod)]*ntimes)
# zones_irr_bool = (padded_zones_1d == 2)
# np.sum(zones_irr_bool)
# zones_irr_bool_all_times = np.hstack([zones_irr_bool]*ntimes)
# df_atmbc.insert(2,
#                 'nodeId',
#                 nodeIds
#                 )
# df_atmbc.insert(3,
#                 'Irr_bool',
#                 zones_irr_bool_all_times
#                 )
# np.sum(zones_irr_bool_all_times)
# df_atmbc_irr = df_atmbc[df_atmbc['Irr_bool']==True]

# # Net_irr_IN_1D.time
# df_atmbc_irr['timeDelta'] = pd.to_timedelta(df_atmbc_irr['time'], 
#                                             unit='s'
#                                             )
# df_atmbc['timeDelta'] = pd.to_timedelta(df_atmbc['time'], 
#                                             unit='s'
#                                             )

# df_atmbc_irr_daily = df_atmbc_irr.set_index('timeDelta').resample('D').sum()
# df_atmbc_daily = df_atmbc.set_index('timeDelta').resample('D').sum()
# # df_atmbc = df_atmbc + 1e-9
# # ax.bar(df_atmbc_irr_daily.index.days, 
# #        df_atmbc_irr_daily['value'], 
# #        color='red', 
# #        edgecolor='k', 
# #        alpha=0.1,
# #        )
# ax.bar(df_atmbc_daily.index.days, 
#        df_atmbc_daily['value'], 
#        color='b', 
#        edgecolor='k', 
#        alpha=0.1,
#        )

# sc['irr_length']
# sc['irr_flow']



# #%%


# fig.savefig(os.path.join(figpath,'bar_plot_netIrr.png'))
