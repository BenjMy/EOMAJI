#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 10:25:35 2024

@author: z0272571a
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
scenario_nb = 1
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

#%%

simu_with_IRR.soil


#%%
pl = pv.Plotter(notebook=False)
cplt.show_vtk(
                unit="saturation",
                 path = os.path.join(simu_with_IRR.workdir,
                                     simu_with_IRR.project_name,
                                     'vtk'
                                     ),
                 ax=pl,
                 # show_edges=True
             )
pl.set_scale(1,1,30)
actor = pl.show_grid(
    color='gray',
    location='outer',
    grid='back',
    ticks='outside',
    xtitle='x (m)',
    ytitle='y (m)',
    ztitle='Elevation (m)',
    font_size=5,
)
pl.show()
    
#%%
# pl = pv.Plotter(notebook=False)
cplt.show_vtk_TL(
                 unit="saturation",
                 path = os.path.join(simu_with_IRR.workdir,
                                     simu_with_IRR.project_name,
                                     'vtk'
                                     ),
                  # ax=pl,
                  show_edges=False,
                  x_units='days'
             )
# pl.set_scale(1,1,50)
# pl.show()
    






