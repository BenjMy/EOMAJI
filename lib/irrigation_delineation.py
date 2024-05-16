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


#%%
scenario_nb = 0
sc = load_scenario(scenario_nb)

#%% Paths
prj_name = 'test_EOMAJI'
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
fig, ax = plt.subplots()
ax.imshow(ETp)



#%% Plot evolution of ETa, SW and PSI INSIDE and OUTSIDE of the irrigation area 
# 1D + time
 
maxDEM=1
index_irrArea, closest = simu_with_IRR.find_nearest_node([500,500,maxDEM])
index_out_irrArea, closest_out_irrArea = simu_with_IRR.find_nearest_node([10,10,maxDEM])


# simu_with_IRR.show_input('atmbc')
atmbc_df = simu_with_IRR.read_inputs('atmbc')
len(atmbc_df.time.unique())


fig, axs = plt.subplots(4,1,
                        sharex=True,
                        )
utils.plot_1d_evol(
                    simu_with_IRR,
                    index_irrArea,
                    closest,
                    out_with_IRR,
                    out_baseline,
                    np.mean(ETp),
                    axs
                )
axs[0].set_title('WITHIN Irrigation Area')
fig.savefig(os.path.join(figpath,
                         'plot_1d_evol_irrArea.png'
                         )
            )

fig, axs = plt.subplots(4,1,
                        sharex=True
                        )
utils.plot_1d_evol(
                    simu_with_IRR,
                    index_out_irrArea,
                    closest,
                    out_with_IRR,
                    out_baseline,
                    np.mean(ETp),
                    axs
                )
axs[0].set_title('OUTSIDE Irrigation Area')
fig.savefig(os.path.join(figpath,
                         'plot_1d_evol_outArea.png'
                         )
            )


# path_results_withIRR = os.path.join(simu_with_IRR.workdir,
#                             simu_with_IRR.project_name,
#                             'vtk'
#                             )
# utils.plot_3d_SatPre(path_results_withIRR)
       
#%% Plot actual ET with time

ET_from_EO_multiindex = out_with_IRR['ETa'].reset_index()
ET_from_EO_multiindex = ET_from_EO_multiindex.set_index(['time', 'X', 'Y'])
ET_from_EO_xr = ET_from_EO_multiindex.to_xarray()
ET_from_EO_xr['ACT. ETRA'].plot.imshow(x="X", y="Y", col="time", col_wrap=3)
plt.savefig(os.path.join(figpath,'ETa_withIRR_spatial_plot.png'))


#%% Find the time when the irrigation is triggered 
# create the ratio between ETa and ETp
np.shape(out_with_IRR['ETa']['ACT. ETRA'])



# Raster zones to nodes
# ----------------------------------------------------------------------
pad_width = ((0, 1), (0, 1))  # Padding only the left and top
padded_ETp = np.pad(ETp, 
                       pad_width, 
                       mode='constant', 
                       constant_values=ETp.mean()
                       )    

#%% Create an xarray dataset with all the necessery variables ETp, ETa, ...

xr_analysis = ET_from_EO_xr.copy()
xr_analysis["ETp"] = (("time", "X", "Y"), [padded_ETp]*len(xr_analysis.time))
xr_analysis["ratio_ETalocal_ETplocal"] = xr_analysis["ACT. ETRA"]/xr_analysis["ETp"]

xr_analysis['ratio_ETalocal_ETplocal'].plot.imshow(x="X", y="Y", col="time", col_wrap=4)
plt.savefig(os.path.join(figpath,'ratioETap_withIRR_spatial_plot.png'))

#%%
# Define the window size for rolling mean
# window_size_x = 4
# window_size_y = 4
# Compute the rolling mean on X and Y dimensions for the ETp variable
rolling_mean_ETp = xr_analysis['ETp'].rolling(X=sc['window_size_x'], 
                                           Y=sc['window_size_y']).mean()
xr_analysis['ETp_rolling_mean'] = rolling_mean_ETp

xr_analysis["ratio_ETalocal_ETp_rollingmean"] = xr_analysis["ACT. ETRA"]/xr_analysis["ETp_rolling_mean"]
xr_analysis['ratio_ETalocal_ETp_rollingmean'].plot.imshow(x="X", y="Y", col="time", col_wrap=4)

# ET_2test.rolling(time=3).mean()


#%%



ratio_ETap = out_with_IRR['ETa'].copy()
ratio_ETap['ratioETap'] = (out_with_IRR['ETa']['ACT. ETRA']/ETp)
ratio_ETap['threshold'] = False
ratio_ETap.loc[abs(ratio_ETap['ratioETap']) < 0.6, 'threshold'] = True


fig, axs = plt.subplots(2,1, sharex=True)
# plot and detect when irrigation has been trigerred
ratio_ETap.groupby('time').max().plot(y='ratioETap',ax=axs[0])
axs[0].axhline(y= -0.6, 
               color='k', 
               linestyle='--', 
               label='Threshold0.6'
               )


ratio_ETap['threshold_numeric'] = ratio_ETap['threshold'].astype(int)

# Plot threshold on the second subplot
# Since 'threshold' is boolean, we'll use a step plot to represent it
ratio_ETap.groupby('time').max().plot(y='threshold_numeric', 
                                      ax=axs[1], 
                                      drawstyle='steps-post', 
                                      marker='s'
                                      )

axs[0].set_ylabel('ETa/ETp (m/s)')
axs[0].set_xlabel('')
axs[1].yaxis.set_ticklabels(['True', 'False'])
fig.savefig(os.path.join(figpath,'Irrigation detection.png'))


#%% 
# differenciation between rain and irrigation events
# --------------------------------------------------
# compute local vs regional ETa/ETp

ET_from_EO_xr.rolling(time=3).mean()

#Since irrigation is normally applied on a larger area, 
# the raster map with per-pixel irrigation events 
# is cleaned up by removing isolated pixels in which irrigation was detected.


