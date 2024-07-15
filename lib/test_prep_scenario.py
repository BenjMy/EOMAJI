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
scenario_nb = 2
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

    
#%% Simulate with NO irrigation 
# -----------------------------
simu_baseline, _ = scenarii2pyCATHY.setup_cathy_simulation(
                                            prj_name=prj_name, 
                                            scenario=sc,
                                            with_irrigation=False,
                                       )

