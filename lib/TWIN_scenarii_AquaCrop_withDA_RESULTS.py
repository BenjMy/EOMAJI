#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 12:38:06 2024
"""

# from aquacrop.utils import prepare_weather, get_filepath

import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from DigTWIN_scenarii import load_scenario 
import argparse
import utils
import numpy as np
import os
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import datetime
from shapely.geometry import box

from scenarii_AQUACROP_DA_ET import load_scenario  as load_scenario_DA
from pyCATHY import cathy_tools
from pyCATHY.DA.cathy_DA import DA
import shutil
import glob
from pyCATHY.importers import cathy_outputs as out_CT
from pyCATHY.DA.observations import read_observations, prepare_observations, make_data_cov
from pyCATHY.DA.cathy_DA import dictObs_2pd
from pyCATHY.DA import perturbate

plt.close('all')
#%%
def get_cmd():
    parser = argparse.ArgumentParser(description='ProcessDA')
    # parser.add_argument('sc', type=int, help='scenario nb')
    #  #Feddes_irr, ref_scenario_irr_het_soil_f5, hetsoil_irr, Archie_Pert_scenarii, freq
    # -------------------------------------------------------------------------------------------------------
    parser.add_argument('-DA_ET_nb','--DA_ET_nb', type=int, 
                        help='DA output nb selection', 
                        required=False, 
                        default=7
                        )  
    args = parser.parse_args()
    return(args)    

args = get_cmd() 
rootpath = Path(os.getcwd()).parent
figpath = Path(f'../figures/scenario_AquaCrop_DA{args.DA_ET_nb}')
figpath.mkdir(parents=True, exist_ok=True)
dataPath = rootpath / 'data/Spain/Spain_ETp_Copernicus_CDS/'

#%%
# ----------------------------------------------------------------------------
#  Build project name
# ----------------------------------------------------------------------------
results_df = pd.read_csv('DA_ET_log.csv',
                         index_col=0)

# prj_name = utils_Bousval.build_prj_name_DA(vars(args))
prj_name_DA = 'DA_ET_' + str(args.DA_ET_nb)

#%% 
# ----------------------------------------------------------------------------
# Init CATHY model: copy reference model input files into DA folder
# ----------------------------------------------------------------------------

path2prjSol = rootpath / "WB_twinModels/AQUACROP/"  # add your local path here
path2prj = rootpath / "WB_twinModels/AQUACROP/DA_ET/"  # add your local path here

weather_scenario = results_df['weather_scenario'].iloc[args.DA_ET_nb]
refModel = results_df['refModel'].iloc[args.DA_ET_nb]
study = results_df['study'].iloc[args.DA_ET_nb]
sc_AQUACROP = results_df['sc_AQUACROP'].iloc[args.DA_ET_nb]
# sc = results_df['sc'].iloc[args.DA_ET_nb]

simu = DA(dirName=path2prj, 
          prj_name=prj_name_DA
          )

simu_ref = cathy_tools.CATHY(dirName=path2prjSol, 
                             prj_name=refModel
                             )
simu_ref_with_irr = cathy_tools.CATHY(dirName=path2prjSol, 
                                      prj_name=refModel + '_withIRR'
                                      )

#%% Import datasets (true model)

ds_analysis_baseline = xr.open_dataset(f'{rootpath}/prepro/ds_analysis_baseline_AquaCrop_sc0_weather_{weather_scenario}.netcdf')
ds_analysis_EO = xr.open_dataset(f'{rootpath}/prepro/ds_analysis_EO_AquaCrop_sc0_weather_{weather_scenario}.netcdf')
analysis_xr = xr.open_dataset(dataPath/f'era5_scenario{sc_AQUACROP}_weather_{weather_scenario}.nc')
grid_xr_with_IRR = xr.open_dataset(f'{rootpath}/prepro/grid_xr_EO_AquaCrop_sc{sc_AQUACROP}_weather_{weather_scenario}.netcdf')

mask_IN = utils.get_mask_IN_patch_i(grid_xr_with_IRR['irrigation_map'],
                              patchid=2
                              )
mask_OUT = utils.get_mask_OUT(grid_xr_with_IRR['irrigation_map'],
                              )

(irr_patch_centers, 
 patch_centers_CATHY) = utils.get_irr_center_coords(irrigation_map=grid_xr_with_IRR['irrigation_map'])   

grid3d = simu.read_outputs('grid3d')
maxDEM = grid3d['mesh3d_nodes'][:,2].max()

for j in patch_centers_CATHY:
    node_index_to_plot, test = simu.find_nearest_node([patch_centers_CATHY[j][1],
                                                    patch_centers_CATHY[j][0],
                                                    maxDEM
                                                    ]
                                                   )
# grid3d['mesh3d_nodes']
                                   
#%%
# ----------------------------------------------------------------------------
# Read DA scenario
# ----------------------------------------------------------------------------

mean_irr_daily = grid_xr_with_IRR['irr_daily'].where(mask_IN, drop=True).mean(['x', 'y']).values
mean_rain_daily = grid_xr_with_IRR['rain_daily'].mean(['x','y'])


# scenarii = load_scenario_DA(study=study)
# scenario =  scenarii[list(scenarii)[sc]]


            
#%% Shorten observation until May
selected_ds_analysis_EO = ds_analysis_EO.isel(time=slice(0, 20))


#%% Load simulation results
backupDA =  os.path.join(simu.workdir,prj_name_DA,prj_name_DA+ '_df.pkl')
results = simu.load_pickle_backup(backupDA)
dem, dem_hd= simu.read_inputs('dem')
test = results['df_DA'].set_index(['time','Ensemble_nb'])

date_string = '2022-01-01 08:00:00.00000'
start_date = pd.to_datetime(date_string)

#%% Plot state dynamic
# -------------------------
# read psi and sw 
# obs2plot_selec = observations.xs('ETact')[['data','data_err']]
psi_ref = simu_ref.read_outputs('psi')
sw_ref, sw_times_ref = simu_ref.read_outputs('sw')


psi_ref_with_irr = simu_ref_with_irr.read_outputs('psi')
sw_ref_with_irr, _ = simu_ref_with_irr.read_outputs('sw')


# psi = simu.read_outputs('psi')
# sw, sw_times = simu.read_outputs('sw')

tass = results['df_DA'].time.unique()


from pyCATHY.cathy_utils import change_x2date, MPa2m, kPa2m
from pyCATHY.plotters import cathy_plots as pltCT

df_atmbc = simu.read_inputs('atmbc')
atmbc_times = df_atmbc.time.unique()
nnod = len(df_atmbc)/len(atmbc_times)
ntimes = len(df_atmbc.time.unique())
nodenb = np.tile(np.arange(0,nnod),ntimes)
df_atmbc['nodeNb'] = nodenb
df_atmbc.set_index(['time','nodeNb'], inplace=True)

# _, FP = simu_ref.read_inputs('soil')

#%% Get point of interest (1 per zone)

                                      
sw_datetimes = change_x2date(sw_times_ref,start_date)
    
# import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

fig, axs = plt.subplots(2, 1, figsize=(15, 5), sharex=True)

# Plotting rain and irrigation events
utils.plot_atmbc_rain_irr_events(
    axs[0],
    sw_datetimes[1:],
    mean_irr_daily,
    mean_rain_daily,
    colors='blue',
)

# Plotting psi reference
axs[1].plot(
    sw_datetimes[1:],
    psi_ref.iloc[1:, node_index_to_plot],
    'r',
    label='ref',
    linestyle='--',
    marker='.'
)

# Plotting psi reference
axs[1].plot(
    sw_datetimes[1:],
    psi_ref_with_irr.iloc[1:, node_index_to_plot],
    'b',
    label='ref',
    linestyle='--',
    marker='.'
)


# Plotting DA time dynamic
pltCT.DA_plot_time_dynamic(
    results['df_DA'],
    'psi',
    node_index_to_plot,
    savefig=False,
    ax=axs[1],
    start_date=start_date,
    atmbc_times=atmbc_times
)
end_date = pd.to_datetime("2022-05-01")

# Add a semi-transparent red rectangle for the calibration period
axs[1].axvspan(
    start_date,
    end_date,
    color="red",
    alpha=0.1
)

# Create a red rectangle patch for the legend
calibration_patch = Rectangle((0, 0), 1, 1, 
                              color="red", 
                              alpha=0.1, 
                              label="Calibration period"
                              )

# Add legend entries
# axs[1].legend(handles=[calibration_patch], loc='upper left')
axs[1].legend().remove()
axs[1].set_ylabel(r'$\psi_{soil}$ (m)')
axs[1].set_xlabel('Assimiliation time')
plt.legend(['solution','pred. ensemble'])
plt.tight_layout()
fig.savefig(figpath/'states_dyn_psi.png',
            dpi=300)


#%% Plot parameters dynamic
# -------------------------
veg_map, veg_map_hd = simu_ref.read_inputs('root_map')
df_SPP, df_FP = simu_ref.read_inputs('soil',MAXVEG=len(np.unique(veg_map)))
results['dict_parm_pert'].keys()
    
for kk in list(results['dict_parm_pert'].keys())[1:]:
    # try:
        fig, ax = plt.subplots(figsize=[11,4])
        _, df = pltCT.DA_plot_parm_dynamic_scatter(parm = kk, 
                                            dict_parm_pert=results['dict_parm_pert'], 
                                            list_assimilation_times = sw_times_ref,
                                            ax=ax,
                                                  )
        zone_nb = int(''.join(filter(str.isdigit, kk)))
        nb_ass_times= len(results['dict_parm_pert'][kk]['ini_perturbation'])
        if 'ZROOT' in kk:
            ax.plot(np.arange(0,len(df.columns),1),[df_FP['ZROOT'].iloc[zone_nb]]*len(df.columns),
                    color='red',linestyle='--')
        elif 'ks' in kk:
            ax.plot(np.arange(0,len(df.columns),1),[df_SPP['PERMX'].iloc[zone_nb]]*len(df.columns),
                    color='red',linestyle='--')
        elif 'porosity' in kk:
            ax.plot(np.arange(0,len(df.columns),1),[df_SPP['POROS'].iloc[zone_nb]]*len(df.columns),
                    color='red',linestyle='--')
        fig.savefig(figpath/f'{kk}.png',dpi=300)
    # except:
    #     pass



#%% Plot RMS dynamic
# -------------------------
fig, ax = plt.subplots(2,figsize=[11,4])
pltCT.DA_RMS(results['df_performance'],'ETact',ax=ax)
pltCT.DA_RMS(results['df_performance'],'ETact10',ax=ax)
fig.savefig(figpath/'ET_RMS.png',dpi=300)

# pltCT.DA_RMS(results['df_performance'],'swc1',ax=ax)
# pltCT.DA_RMS(results['df_performance'],'swc2',ax=ax)
# pltCT.DA_RMS(results['df_performance'],'tensio',ax=ax) 

