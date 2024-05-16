#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:50:23 2023
@author: ben

Project EOMAJI (Earth Observation system to Manage Africa’s food systems by Joint-knowledge of crop production and Irrigation digitization) 
ET-Based algoritms for net irrigation estimation. 

Net irrigation is estimated based on the systematic evapotranspiration (ET) residuals between a
remote sensing ‐ based model and a calibrated hydrologic or SWB model that does not include an
irrigation scheme.

I net(t)= ET_EO − ET_baseline

- Koch, J; Zhang, W; Martinsen, G; He, X; Stisen, S. Estimating net irrigation across the North China Plain through dual modeling of evapotranspiration. Water Resources Research, 2020, 56(12), p. e2020WR027413. https://doi.org/10.1029/2020WR027413
- Garrido-Rubio, J; Gonzalez-Piqueras, J; Campos, I; Osann, A; Gonzalez-Gomez, L; Calera, A. Remote sensing–based soil water balance for irrigation water accounting at plot and water user association management scale. Agricultural Water Management, 2020, 238, p.106236. https://doi.org/10.1016/j.agwat.2020.106236

""" 

import scenarii
import pyCATHY
import numpy as np
from pyCATHY import CATHY
from pyCATHY.plotters import cathy_plots as cplt
from pyCATHY.importers import cathy_outputs as out_CT
import pyCATHY.meshtools as msh_CT
import utils


import os
import pyvista as pv
import matplotlib.pyplot as plt
import pandas as pd

plt.close('all')

#%% Paths
prj_name = 'test_EOMAJI'
# figpath = os.path.join('../figures/',prj_name)
figpath = os.path.join('../figures/')

#%% CONSTANTS

ETp = -1e-7

#%% Simulate with irrigation atmospheric boundary conditions
# ----------------------------------------------------------
simu_with_IRR, _ = utils.setup_cathy_simulation(
                                            prj_name=prj_name,
                                            ETp = ETp,
                                            with_irrigation=True,
                                            irr_time_index = 5,
                                            irr_flow = 5e-7 #m/s
                                            )
# simu_with_IRR.run_processor(
#                             IPRT1=2,
#                             verbose=True,
#                             DTMIN=1,
#                             DTMAX=1e3,
#                             DELTAT=1e2,
#                         )
    
#%% Simulate with NO irrigation 
# -----------------------------
simu_baseline, _ = utils.setup_cathy_simulation(
                                            prj_name=prj_name, 
                                            ETp = ETp,
                                            with_irrigation=False,
                                       )


# simu_baseline.run_processor(
#                             IPRT1=2,
#                             verbose=True,
#                             DTMIN=1,
#                             DTMAX=1e3,
#                             DELTAT=1e2,
#                         )

    

#%% Read and plot outputs (scenario with irrigation)
# --------------------------------------------------

out_with_IRR = utils.read_outputs(simu_with_IRR)
out_baseline = utils.read_outputs(simu_baseline)


# IRRIGATION QUANTIFICATION
#%% Quantify volume applied
# -------------------------


# fig, ax = plt.subplots()
ET_baseline_multiindex = out_baseline['ETa'].reset_index()
ET_baseline_multiindex = ET_baseline_multiindex.set_index(['time', 'X', 'Y'])
ET_baseline_xr = ET_baseline_multiindex.to_xarray()
ET_baseline_xr['ACT. ETRA'].plot.imshow(x="X", y="Y", col="time", col_wrap=3)
plt.savefig(os.path.join(figpath,'ET_baseline_spatial_plot.png'))



fig, axs = plt.subplots(2)

simu_with_IRR.show_input('atmbc',
                         ax=axs[0]
                         )
atmbc_df = simu_with_IRR.read_inputs('atmbc')
irr = ETp - atmbc_df.values

def estimate_volume_from_atmbc():
    pass

# plt.title


#%% Read and plot outputsNO irrigation

# path_results = os.path.join(simu_baseline.workdir,
#                             simu_baseline.project_name,
#                             'vtk'
#                             )
# cplt.show_vtk(
#                 unit="pressure",
#                 path=path_results,
#                 savefig=True,
#                 timeStep=1,
#                 )

# ET_baseline = os.path.join(simu_baseline.workdir,
#                         simu_baseline.project_name,
#                         'fort.777'
#                         )
# ET_baseline = out_CT.read_fort777(fort777)
# ET_baseline.set_index('time',inplace=True)
# ET_baseline.drop_duplicates(inplace=True)

# ET_baseline.plot(y='ACT. ETRA')
# cplt.show_spatialET(ET_baseline,
#                     ti=1,
#                     )
# ET_baseline_multiindex = ET_baseline.reset_index()
# ET_baseline_multiindex = ET_baseline_multiindex.set_index(['time', 'X', 'Y'])
# ET_baseline_xr = ET_baseline_multiindex.to_xarray()
# ET_baseline_xr['ACT. ETRA'].plot.imshow(x="X", y="Y", col="time", col_wrap=3)




# ETa_from_EO.index

#%% Compute ratio between ETp and ETa
# This is achieved by first calculating the change in ETa/p between the time on 
# which irrigation is to be detect and most recent previous time on which ET 
# estimates are available. This change is calculated both locally 
# (i.e. at individual pixel level) and regionally (i.e. as an average change
#                                                  in all agricultural pixels 
# #                                                  within 10 km window). 






    
#%%



    
    
    
# EOMAJI_simu.update_zone(zones)

# EOMAJI_simu.show_input('zone')


#%%
# EOMAJI_simu.create_mesh_vtk()
# EOMAJI_simu.run_preprocessor(verbose=True)

#%%
# t_atmbc = [0,1e2,5e2]
# ETp = -1e-7
# v_atmbc = zones + ETp

# EOMAJI_simu.update_atmbc(
#                         HSPATM=0,
#                         IETO=0,
#                         time=t_atmbc,
#                         netValue=[-1e-7]*len(t_atmbc)
#                       )


# EOMAJI_simu.run_processor(IPRT1=1,
#                           verbose=True
#                           )

#%%
# path_results = os.path.join(EOMAJI_simu.workdir,
#                             EOMAJI_simu.project_name,
#                             'vtk'
#                            )

# cplt.show_vtk(
#                 unit="pressure",
#                 path=path_results,
#                 savefig=True,
#                 )
# pl = pv.Plotter()
# pl.add_mesh()
