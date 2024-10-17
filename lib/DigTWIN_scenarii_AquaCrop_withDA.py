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

print('Look for DA paper Kustas for error evaluation!')
expected_value = 1e-7  #expected mean value of ETa
percentage_error = args.dataErr 
absolute_error = expected_value * (percentage_error / 100)


#%% 
# ----------------------------------------------------------------------------
# Init CATHY model: copy reference model input files into DA folder
# ----------------------------------------------------------------------------

path2prjSol = "../WB_twinModels/"  # add your local path here
path2prj = "../WB_twinModels/DA_ET/"  # add your local path here

simu_ref = cathy_tools.CATHY(dirName=path2prjSol, 
                             prj_name=args.refModel
                             )
src_path = os.path.join(simu_ref.workdir,simu_ref.project_name)
dst_path = os.path.join(path2prj,prj_name_DA)

# Find .vtk files and output files to exclude
vtk_files = glob.glob(os.path.join(src_path, '**', '*.vtk'), recursive=True)
out_files = glob.glob(os.path.join(src_path, 'output', '**'), recursive=True)

# Combine both lists of files to exclude
exclude_files = vtk_files + out_files

# Define a function to ignore vtk and output files
def ignore_files(dir, files):
    excluded = []
    for file in files:
        # If the file's full path is in the exclude list, add it to the ignore list
        full_path = os.path.join(dir, file)
        if any(full_path.startswith(excl_file) for excl_file in exclude_files):
            excluded.append(file)
    return set(excluded)

# Copy the source directory to the destination, excluding vtk and output files
shutil.copytree(src_path, dst_path, dirs_exist_ok=True, ignore=ignore_files)

simu = DA(dirName=path2prj, 
          prj_name=prj_name_DA
          )



#%% 
# ----------------------------------------------------------------------------
# Update CATHY inputs
# ----------------------------------------------------------------------------
simu.run_preprocessor()
simu.run_processor(IPRT1=3)

DEM, _ = simu.read_inputs('dem')
grid3d = simu.read_outputs('grid3d')

#%% Plot actual ET
# --------------------
# dtcoupling = simu_ref.read_outputs('dtcoupling')
# ET_act_obs = dtcoupling[['Time','Atmact-vf']]

# df_fort777 = out_CT.read_fort777(os.path.join(simu_ref.workdir,
#                                               simu_ref.project_name,
#                                               'fort.777'),
#                                   )

# df_fort777 = df_fort777.set_index('time_sec')
# ET_nn = df_fort777.set_index('SURFACE NODE').index[0]
# df_fort777.set_index('SURFACE NODE').loc[ET_nn]['ACT. ETRA']

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

# event_type_node_IN = event_type.where(mask_IN, drop=True).mean(['x','y'])
# event_type_node_OUT = event_type.where(mask_OUT, drop=True).mean(['x','y'])

# Checking size:
ds_analysis_EO.ETp.shape
ds_analysis_EO['ACT. ETRA'].shape
np.shape(DEM)

#%% Shorten observation until May

selected_ds_analysis_EO = ds_analysis_EO.isel(time=slice(0, 20))

#%%
# d
df_atmbc = simu_ref.read_inputs('atmbc')
grouped = df_atmbc.groupby('time')['value'].apply(list).reset_index()
netValue = []
for values in grouped['value']:
    netValue.append(values)

simu.update_atmbc(
                    HSPATM = simu_ref.atmbc['HSPATM'],
                    IETO = simu_ref.atmbc['IETO'],
                    time = df_atmbc['time'].unique(),
                    netValue = netValue,
                    )
simu.show_input('atmbc')

#%% Read observations = actual ET
data_measure = {}

valid_time = analysis_xr.valid_time.values
elapsed_seconds = (analysis_xr.valid_time - analysis_xr.valid_time[0]).dt.total_seconds()

for i, tobsi in enumerate(selected_ds_analysis_EO.time):  
    for node_obsi in range(int(grid3d['nnod'])):
        data_measure = read_observations(
                                        data_measure,
                                        selected_ds_analysis_EO['ACT. ETRA'].isel(time=i).values.flatten()[node_obsi],  # Access the specific node value
                                        data_type = 'ETact', 
                                        data_err = absolute_error, # instrumental error
                                        show=True,
                                        tA=elapsed_seconds.to_numpy()[i],
                                        obs_cov_type='data_err', #data_err
                                        mesh_nodes=node_obsi, 
                                        datetime=pd.to_datetime(valid_time[i]),
                                        ) 
        

#%% Get Nodes of interest (1 in each zone)

# (nodes2plots, 
#  nodespos2plots, 
#  nodes2plots_surf, 
#  nodespos2plots_surf) = utils.get_NOIs(
#                                       simu=simu_ref,
#                                       depths = [0,1],
#                                       maxdem=DEM.max()
#                                     )
nodes2plots_surf = [1,100]
#%%
data_measure_df = dictObs_2pd(data_measure) 
data_measure_df.index
data_measure_df.iloc[0]

root_map, root_hd = simu_ref.read_inputs('root_map')
_, FP = simu.read_inputs('soil',MAXVEG=len(np.unique(root_map)))
simu.update_soil(PMIN=-1e+35,show=True)
# simu.update_zone()
# simu.soil_SPP
# simu.soil_FP
# simu.update_soil(PMIN=-1e+35,show=True)

fig, ax = plt.subplots(
                        # figsize=(6,2)
                        )
data_measure_df.xs(f'ETact{nodes2plots_surf[0]}', level=0).plot(
                                                                x='datetime',
                                                                y='data',
                                                              ax=ax,
                                                              marker='.',
                                                              color='r',
                                                              label='ZROOT:' + str(FP['ZROOT'].iloc[1])
                                                              )
data_measure_df.xs(f'ETact{nodes2plots_surf[1]}', level=0).plot(
                                                                x='datetime',
                                                                y='data',
                                                                ax=ax,marker='.',
                                                                color='b',
                                                                label='ZROOT:' + str(FP['ZROOT'].iloc[0])
                                                                )

ax.set_ylabel('ETobs (m/s)')
# data_measure_df.xs('ETact400', level=0)
fig.savefig(os.path.join(simu.workdir,simu.project_name,'atmbc_pernodes.png'),
            dpi=350,
            bbox_inches='tight'
            )

data_measure_df.columns

#%%

stacked_data_cov = []
for i in range(len(data_measure_df.index.get_level_values(1).unique())):
    matrix = np.zeros([ds_analysis_baseline.x.shape[0],ds_analysis_baseline.y.shape[0]])
    np.fill_diagonal(matrix, args.dataErr)
    stacked_data_cov.append(matrix)
    
simu.stacked_data_cov = stacked_data_cov

#%% perturbated variable 

list_pert = perturbate.perturbate(simu, 
                                  scenario, 
                                  args.nens
                                  )   

#%% Parameters perturbation
var_per_dict_stacked = {}
for dp in list_pert:
    
    if len(list_pert)>10:
        savefig = False
    else:
        savefig = os.path.join(
                                simu.workdir,
                                simu.project_name,
                                simu.project_name + dp['savefig']
                                )
    np.random.seed(1)
    # need to call perturbate_var as many times as variable to perturbate
    # return a dict merging all variable perturbate to parse into prepare_DA
    var_per_dict_stacked = perturbate.perturbate_parm(
                                                    var_per_dict_stacked,
                                                    parm=dp, 
                                                    type_parm = dp['type_parm'], # can also be VAN GENUCHTEN PARAMETERS
                                                    mean =  dp['mean'],
                                                    sd =  dp['sd'],
                                                    sampling_type =  dp['sampling_type'],
                                                    ensemble_size =  dp['ensemble_size'], # size of the ensemble
                                                    per_type= dp['per_type'],
                                                    savefig=savefig
                                                    )
    
#%%
try:
    utils.plot_atmbc_pert(simu,var_per_dict_stacked)
except:
    pass


#%% Run DA sequential

# DTMIN = 1e-2
# DELTAT = 1e-1

DTMIN = 1e-2
DELTAT = 100
DTMAX = 1e3

# stop
simu.run_DA_sequential(
                          VTKF=2,
                          TRAFLAG=0,
                          DTMIN=DTMIN,
                          DELTAT=DELTAT,
                          DTMAX=DTMAX,
                          parallel=True,    
                          dict_obs= data_measure,
                          list_assimilated_obs='all', # default
                          list_parm2update=scenarii[list(scenarii)[args.sc]]['listUpdateParm'],
                          DA_type=args.DAtype, #'pf_analysis', # default
                          dict_parm_pert=var_per_dict_stacked,
                          open_loop_run=False,
                          threshold_rejected=80,
                          damping=args.damping                    
                          )

# args.parallel
