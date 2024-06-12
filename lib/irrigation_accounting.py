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
simu_with_IRR, _ = scenarii2pyCATHY.setup_cathy_simulation(
                                            prj_name=prj_name,
                                            scenario=sc,
                                            # ETp = ETp,
                                            with_irrigation=True,
                                            # irr_time_index = 5,
                                            # irr_flow = 5e-7 #m/s
                                            )
#%% Simulate with NO irrigation 
# -----------------------------
simu_baseline, _ = scenarii2pyCATHY.setup_cathy_simulation(
                                            prj_name=prj_name, 
                                            scenario=sc,
                                            with_irrigation=False,
                                       )

#%%
# atmbc_df = simu_with_IRR.read_inputs('atmbc')
sc['irr_time_index']


net_irr_solution = sc['irr_flow']*sc['irr_length']

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

ds_analysis_baseline = ds_analysis_baseline.assign_coords(time=ds_analysis_EO['time'])
netIrr = ds_analysis_EO['ACT. ETRA'] - ds_analysis_baseline['ACT. ETRA']
# netIrr = netIrr.rename({netIrr.name: 'net Irrigation (m/s)'})
netIrr = netIrr.rename('netIrr (m/s)')

netIrr_cumsum = netIrr.cumsum('time')
netIrr_cumsum = netIrr_cumsum.rename('netIrr cumsum (m/s)')

netIrr.plot.imshow(x="X", y="Y", col="time", col_wrap=4)
plt.savefig(os.path.join(figpath,'netIrr_spatial_plot.png'))


netIrr_cumsum.plot.imshow(x="X", y="Y", col="time", col_wrap=4)
plt.savefig(os.path.join(figpath,'netIrr_spatial_plot.png'))



