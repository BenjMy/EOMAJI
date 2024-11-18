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
from pathlib import Path

#%%
def get_cmd():
    parse = argparse.ArgumentParser()
    process_param = parse.add_argument_group('process_param')
    process_param.add_argument('-run_process', 
                               type=int, 
                               help='run_process',
                               default=1, 
                               required=False
                               ) 
    process_param.add_argument('-scenario_nb', 
                               type=int, 
                               help='scenario_nb',
                               default=0, # only 1 patch of irrigation
                               required=False
                               ) 
    process_param.add_argument('-weather_scenario', 
                               type=str, 
                               help='weather_scenario',
                                default='reference', 
                               # default='plus20p_tp', 
                               required=False
                               )     
    process_param.add_argument('-SMT', 
                               type=int, 
                               help='SMT %TAW (total available water',
                               default=70, # only 1 patch of irrigation
                               required=False
                               )    
    process_param.add_argument('-ApplyEOcons', 
                               type=str, 
                               help='Applying EO cons',
                               default=None, # only 1 patch of irrigation
                               required=False
                               )    
    args = parse.parse_args()
    return(args)    

rootPath = Path(os.getcwd())
dataPath = rootPath / '../data/Spain/Spain_ETp_Copernicus_CDS/'

args = get_cmd() 
figpath = rootPath /f'../figures/scenario_AquaCrop_sc{args.scenario_nb}_weather_{args.weather_scenario}'
figpath.mkdir(parents=True, exist_ok=True)
#%%
# run_process = True # don't rerun the hydro model
# scenario_nb = 3
sc, scenario_EO = load_scenario(args.scenario_nb)

#%% ERA5 reanalysis data in Spain
# -----------------------------------------------------------------------------
analysis_xr = utils.prep_ERA5_reanalysis_data_SPAIN(dataPath)

#%% # Create scenarios
# Scenario 1: +20% precipitation
wdf, scenario_analysis = utils.create_scenario_ERA5(analysis_xr,
                                                    args,
                                                    dataPath
                                                    )

fig, axs = plt.subplots(3,1,sharex=True)
utils.plot_weather_ET_timeserie(analysis_xr,
                                scenario_analysis,
                                axs
                                )
plt.savefig(figpath/'scenario_inputs.png',
            dpi=300,
            )
#%% AquaCrop model parameters 
# -----------------------------------------------------------------------------
# path = get_filepath('champion_climate.txt')
# wdf_test = prepare_weather(path)
# wdf_test
# idpl = np.where(wdf_test.Date == '2018-07-06')[0]
# wdf_test.iloc[13335]
#%%
# wdf_Tnew = wdf.drop('MinTemp',axis=1)
(sim_start, 
 sim_end, 
 soil, 
 crop, 
 initWC, 
 irr_mngt) = utils.prep_AQUACROP_inputs(wdf,args)

#%%
# ss
model = AquaCropModel(sim_start,
                    sim_end,
                    wdf,
                    soil,
                    crop,
                    initial_water_content=initWC,
                    irrigation_management=irr_mngt) # create model
model.run_model(
    till_termination=True
    ) # run model till the end
# outputs.append(model._outputs.final_stats) # save results

#%% Crop developement
# -----------------------------------------------------------------------------
# model.soil
model.crop.Name
crop_growth = model.get_crop_growth()
# crop_growth.columns
crop_growth.z_root

#%% Water fluxes 
# -----------------------------------------------------------------------------
# EsPot (float): Potential surface evaporation current day
# dap: day after planting
# TrPot (float): Daily potential transpiration
# z_gw (float): groundwater depth
water_flux = model.get_water_flux()
water_flux.IrrDay.values
model.irrigation_management
# model.field_management
# model.groundwater

#%% Parse results to xarray
# sc = load_scenario(0)
sc['ETp'] = water_flux.EsPot.values*(1e-3/86400)
sc['nb_days'] = len(water_flux.IrrDay)
sc['irr_time_index'] = [water_flux.IrrDay.index.values]
# sc['irr_datetime'] = water_flux.IrrDay.Date
# sc['irr_flow'] = [model.weather_df.Precipitation.values]
sc['irr_flow'] = [water_flux.IrrDay.values*(1e-3/86400)]
sc['rain_time_index'] = np.arange(0,len(model.weather_df.Precipitation),1)
sc['datetime'] = model.weather_df.Date
sc['rain_flow'] = model.weather_df.Precipitation.values*(1e-3/86400)
sc['z_root'] = crop_growth.z_root.values
sc_df = pd.DataFrame.from_dict(sc,orient='index').T
# sc_df.to_csv('EOMAJI_synthetic_log.csv',index=False)
# sc_df = pd.read_csv('EOMAJI_synthetic_log.csv',index_col=False)

#%% Quality 
# quality_check = {}
# quality_check['nb_of_irr_events'] = np.count_nonzero(model.weather_df.Precipitation.values)
# quality_check['nb_of_rain_events'] = np.count_nonzero(wdf.Precipitation)
#%% Paths
prj_name = f'EOMAJI_AquaCrop_sc{args.scenario_nb}_weather_{args.weather_scenario}_SMT_{args.SMT}_EOcons_{args.ApplyEOcons}' 
sc['figpath'] = figpath
#%% Simulate with irrigation atmospheric boundary conditions
# ----------------------------------------------------------
# s
# sc_EO = utils.check_and_tune_E0_dict(sc)
if args.ApplyEOcons is not None:
    sc_withirr = sc | scenario_EO
else:
    sc_withirr = sc

simu_with_IRR, grid_xr_with_IRR = scenarii2pyCATHY.setup_cathy_simulation(
                                                                         root_path = rootPath,
                                                                         prj_name=prj_name, 
                                                                         scenario=sc_withirr,
                                                                         with_irrigation=True,
                                                                    )

# sc['PERMX']

if args.run_process:
    simu_with_IRR.run_processor(
                                IPRT1=2,
                                verbose=True,
                                DTMIN=1e-1,
                                DTMAX=1e4,
                                DELTAT=1e2,
                            )
# ee
plt.close('all')
#%% Simulate with NO irrigation 
# -----------------------------
simu_baseline, grid_xr_baseline = scenarii2pyCATHY.setup_cathy_simulation(
                                                     root_path = rootPath,
                                                     prj_name=prj_name, 
                                                     scenario=sc,
                                                     with_irrigation=False,
                                                )


if args.run_process:
    simu_baseline.run_processor(
                                IPRT1=2,
                                verbose=True,
                                DTMIN=1e-1,
                                DTMAX=1e4,
                                DELTAT=1e2,
                            )

plt.close('all')


#%%

time_sel = np.arange(0,len(grid_xr_with_IRR.time),10)
grid_xr_with_IRR['irr_daily'].isel(time=time_sel).plot.imshow(x="x", y="y", 
                                                              col="time", 
                                                              col_wrap=4
                                                              )
# plt.title('ETp EO')
plt.savefig(os.path.join(figpath,'irr_daily_aquacrop.png'),
            dpi=300,
            )
# grid_xr_with_IRR['irr_daily'].sum()
# grid_xr_with_IRR['rain_daily'].sum()

time_sel = np.arange(0,len(grid_xr_with_IRR.time),10)
grid_xr_with_IRR['rain_daily'].isel(time=time_sel).plot.imshow(x="x", y="y", 
                                                              col="time", 
                                                              col_wrap=4
                                                              )
# plt.title('ETp EO')
plt.savefig(os.path.join(figpath,'rain_daily_aquacrop.png'),
            dpi=300,
            )

grid_xr_with_IRR.attrs = {}
grid_xr_with_IRR.to_netcdf(rootPath / f'../prepro/grid_xr_EO_AquaCrop_sc{args.scenario_nb}_weather_{args.weather_scenario}.netcdf')
grid_xr_baseline.attrs = {}
grid_xr_baseline.to_netcdf(rootPath / f'../prepro/grid_xr_baseline_AquaCrop_sc{args.scenario_nb}_weather_{args.weather_scenario}.netcdf')


