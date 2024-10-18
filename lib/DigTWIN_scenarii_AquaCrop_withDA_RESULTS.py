#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 12:38:06 2024
"""

# from aquacrop.utils import prepare_weather, get_filepath
from aquacrop import AquaCropModel, Soil, Crop, InitialWaterContent, IrrigationManagement
from aquacrop.utils import prepare_weather, get_filepath

import xarray as xr
import rioxarray as rio
from pathlib import Path
import Majadas_utils
import contextily as cx
import matplotlib.pyplot as plt
import pandas as pd

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

from scenarii_AQUACROP_DA_ET import load_scenario 
from pyCATHY import cathy_tools
from pyCATHY.DA.cathy_DA import DA
import shutil
import glob
from pyCATHY.importers import cathy_outputs as out_CT
from pyCATHY.DA.observations import read_observations, prepare_observations, make_data_cov
from pyCATHY.DA.cathy_DA import dictObs_2pd
from pyCATHY.DA import perturbate

#%%
def get_cmd():
    parser = argparse.ArgumentParser(description='ProcessDA')
    # parser.add_argument('sc', type=int, help='scenario nb')
    #  #Feddes_irr, ref_scenario_irr_het_soil_f5, hetsoil_irr, Archie_Pert_scenarii, freq
    # -------------------------------------------------------------------------------------------------------
    parser.add_argument('-study','--study', type=str, 
                        help='study selection', 
                        required=False, 
                        default='ET_scenarii'
                        )  
    parser.add_argument('-sc','--sc', type=int,
                        help='scenario nb', 
                        required=False, 
                        default=0
                        )
    parser.add_argument('-weather_scenario', 
                            type=str, 
                            help='weather_scenario',
                             default='reference', 
                            # default='plus20p_tp', 
                            required=False
                               )    
    parser.add_argument('-nens','--nens', type=int, 
                        help='nb of ensemble', 
                        required=False,
                        default=24
                        )
    parser.add_argument('-DAtype',type=str, 
                        help='type of DA',
                        default='enkf_Evensen2009'
                        # default='enkf_Evensen2009_Sakov'
                        )
    parser.add_argument('-DAloc',type=int, 
                        help='DA localisation',
                        # default='enkf_Evensen2009'
                        default=1
                        )
    parser.add_argument('-damping',type=float, 
                        help='damping factor',
                        default=1
                        )
    parser.add_argument('-dataErr',type=float, 
                        help='error data',
                        # default=5e-2
                        default=5
                        )
    parser.add_argument('-parallel',type=int, 
                        help='parallel computing',
                        default=1
                        )
    parser.add_argument('-refModel',type=str, 
                        help='name of the ref model',
                        default='EOMAJI_AquaCrop_sc0_weather_reference_SMT_70'                       
                        ) #ZROOT_spatially_from_weill
    args = parser.parse_args()
    return(args)    

args = get_cmd() 
rootpath = Path(os.getcwd()).parent
figpath = Path(f'../figures/scenario_AquaCrop_DA{args.sc}')
figpath.mkdir(parents=True, exist_ok=True)
dataPath = rootpath / 'data/Spain/Spain_ETp_Copernicus_CDS/'

#%%
# ----------------------------------------------------------------------------
#  Build project name
# ----------------------------------------------------------------------------

args = get_cmd()
results_df, matching_index = utils.backup_simulog_DA(args,
                                                    filename='DA_ET_log.csv'
                                                    )
# prj_name = utils_Bousval.build_prj_name_DA(vars(args))
prj_name_DA = 'DA_ET_' + str(matching_index)



#%%
# ----------------------------------------------------------------------------
# Read DA scenario
# ----------------------------------------------------------------------------

scenarii = load_scenario(study=args.study)
scenario =  scenarii[list(scenarii)[args.sc]]

#%% 
# ----------------------------------------------------------------------------
# Init CATHY model: copy reference model input files into DA folder
# ----------------------------------------------------------------------------

path2prjSol = "../WB_twinModels/"  # add your local path here
path2prj = "../WB_twinModels/DA_ET/"  # add your local path here

simu = DA(dirName=path2prj, 
          prj_name=prj_name_DA
          )

#%%
# import rioxarray as rxr
# ds_analysis_EO['time'] = ds_analysis_EO['time'].astype('timedelta64[D]')
ds_analysis_baseline = xr.open_dataset(f'{rootpath}/prepro/ds_analysis_baseline_AquaCrop_sc0_weather_{args.weather_scenario}.netcdf')
ds_analysis_EO = xr.open_dataset(f'{rootpath}/prepro/ds_analysis_EO_AquaCrop_sc0_weather_{args.weather_scenario}.netcdf')
analysis_xr = xr.open_dataset(dataPath/'era5_scenario_ref.nc')

grid_xr_with_IRR = xr.open_dataset(f'{rootpath}/prepro/grid_xr_EO_AquaCrop_sc0_weather_{args.weather_scenario}.netcdf')

mask_IN = utils.get_mask_IN_patch_i(grid_xr_with_IRR['irrigation_map'],
                              patchid=2
                              )
mask_OUT = utils.get_mask_OUT(grid_xr_with_IRR['irrigation_map'],
                              )

#%% Shorten observation until May

selected_ds_analysis_EO = ds_analysis_EO.isel(time=slice(0, 20))



#%% Plot states_dyn psi  
fig, axs = plt.subplots(2,2, 
                        sharex=True,sharey=True,
                        # figsize=(5,7)
                        )
# simu.read_inputs('soil',MAXVEG=2)
# simu.update_dem_parameters()
# simu.update_veg_map()
# simu.update_soil()


ZROOTid = [0,0,1,1]
sw_datetimes = change_x2date(sw_times,start_date)
axs= axs.ravel()
for i, nn in enumerate(nodes2plots):
    axs[i].plot(sw_datetimes[1:],psi.iloc[1:,nn],'r',label='ref',linestyle='--',marker='.')
    pltCT.DA_plot_time_dynamic(results['df_DA'],
                                'psi',
                                nn,
                                savefig=False,
                                ax=axs[i],
                                start_date=start_date,
                                atmbc_times=atmbc_times
                                )  
    axs[i].legend().remove()
    axs[i].set_ylabel(r'$\psi_{soil}$ (m)')
    
    dd = simu.DEM.max() - np.round(simu.grid3d['mesh3d_nodes'][nn][0][2]*10)/10
    zr= str(FP['ZROOT'].iloc[ZROOTid[i]])

    axs[i].set_title('$Z_{root}$:' + str(zr) + 'm '
                     'SMC:' + str(dd) + 'm'
                     )

    if i == len(axs):
        axs[i].set_xlabel('Assimiliation time')
    # axs
    plt.legend(['solution','pred. ensemble'])
    plt.tight_layout()
fig.savefig(os.path.join(simu.workdir,simu.project_name,'states_dyn_psi.png'),dpi=300)
    
    
#%% Plot parameters dynamic
# -------------------------
veg_map, veg_map_hd = simu_ref.read_inputs('root_map')
df_SPP, df_FP = simu_ref.read_inputs('soil',MAXVEG=len(np.unique(veg_map)))
results['dict_parm_pert'].keys()
    
for kk in results['dict_parm_pert'].keys():
    # try:
        fig, ax = plt.subplots(figsize=[11,4])
        _, df = pltCT.DA_plot_parm_dynamic_scatter(parm = kk, 
                                            dict_parm_pert=results['dict_parm_pert'], 
                                            list_assimilation_times = assimilationTimes,
                                            ax=ax,
                                                  )
        zone_nb = int(''.join(filter(str.isdigit, kk)))
        nb_ass_times= len(results['dict_parm_pert'][kk]['ini_perturbation'])
        if 'ZROOT' in kk:
            ax.plot(np.arange(0,len(df.columns),1),[df_FP['ZROOT'].iloc[zone_nb]]*len(df.columns),
                    color='red',linestyle='--')
        elif 'PCREF' in kk:
            ax.plot(np.arange(0,len(df.columns),1),[df_FP['PCREF'].iloc[zone_nb]]*len(df.columns),
                    color='red',linestyle='--')
        fig.savefig( os.path.join(simu.workdir,simu.project_name,kk + '.png'),dpi=300)
    # except:
    #     pass
