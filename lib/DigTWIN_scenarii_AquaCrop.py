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

dataPath = Path('../data/Spain/Spain_ETp_Copernicus_CDS/')
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
                               type=int, 
                               help='Applying EO cons',
                               default=1, # only 1 patch of irrigation
                               required=False
                               )    
    args = parse.parse_args()
    return(args)    

args = get_cmd() 
figpath = Path(f'../figures/scenario_AquaCrop_sc{args.scenario_nb}_weather_{args.weather_scenario}')
figpath.mkdir(parents=True, exist_ok=True)
#%%
# run_process = True # don't rerun the hydro model
# scenario_nb = 3
sc = load_scenario(args.scenario_nb)

#%% ERA5 reanalysis data in Spain
# -----------------------------------------------------------------------------

# ERA5ds = xr.open_dataset(dataPath / 'test/data_stream-oper.nc')

ERA5ds = xr.open_dataset(dataPath /'data_SPAIN_ERA5_singlelevel_hourly.nc')
ERA5ds = ERA5ds.rio.write_crs("EPSG:4326")


# Central point in lat/lon
central_lat = 39.978757
central_lon = -5.81843

# Calculate the degree distance for 7.5 km
delta_lat = 100 / 111  # Approximate change in degrees latitude
delta_lon = 100 / (111 * np.cos(np.radians(central_lat)))  # Change in degrees longitude

# Define the bounding box in lat/lon
min_lat = central_lat - delta_lat
max_lat = central_lat + delta_lat
min_lon = central_lon - delta_lon
max_lon = central_lon + delta_lon

# Crop the dataset using the bounding box
cropped_ERA5ds = ERA5ds.sel(latitude=slice(max_lat, min_lat), longitude=slice(min_lon, max_lon))


#%%

fig, ax = plt.subplots()
ERA5ds.pev.isel(valid_time=0).plot.imshow(ax=ax,
                                        add_colorbar=False
                                        )
ax.set_aspect('equal')
# cx.add_basemap(ax, 
#                 crs=cropped_ERA5ds.rio.crs,
#                 alpha=0.4,
#                 # credit=False
#                 )
# Resample to daily frequency and calculate the mean for each day
# daily_ERA5ds = cropped_ERA5ds.resample(valid_time='1D').mean().mean(['latitude','longitude'])

# cropped_ERA5ds['mint2m'] = cropped_ERA5ds.resample(valid_time='1D').min('t2m').mean(['latitude','longitude'])
maxt2m = cropped_ERA5ds.resample(valid_time='1D').max().mean(['latitude','longitude'])['t2m']
mint2m  = cropped_ERA5ds.resample(valid_time='1D').min().mean(['latitude','longitude'])['t2m']
sumtp = cropped_ERA5ds.resample(valid_time='1D').sum().mean(['latitude','longitude'])['tp']
sumpev  = cropped_ERA5ds.resample(valid_time='1D').sum().mean(['latitude','longitude'])['pev']

# Create a new Dataset with the calculated variables
analysis_xr = xr.Dataset({
    'maxt2m': maxt2m - 273.15,
    'mint2m': mint2m - 273.15,
    'sumtp': sumtp*1000,
    'sumpev': abs(sumpev)*1000
})

# Optionally, you can assign attributes to the dataset
analysis_xr.attrs['description'] = 'Daily aggregated statistics from the cropped ERA5 dataset'
# analysis_xr.attrs['units'] = {
#     'maxt2m': 'K',
#     'mint2m': 'K',
#     'sumtp': 'm',
#     'meanpev': 'm'
# }
#%%
# ss

#%%

# Create scenarios
# Scenario 1: +20% precipitation

scenario_analysis = analysis_xr.copy()
if args.weather_scenario == 'plus20p_tp':
    scenario_analysis = analysis_xr.copy()
    scenario_analysis['sumtp'] = scenario_analysis['sumtp'] * 1.20
elif args.weather_scenario =='minus20p_tp':
    # Scenario 2: -20% precipitation
    scenario_analysis = analysis_xr.copy()
    scenario_analysis['sumtp'] = scenario_analysis['sumtp'] * 0.80
elif args.weather_scenario =='plus25p_t2m':
    # Scenario 3: +25% air temperature
    scenario_analysis = analysis_xr.copy()
    scenario_analysis['maxt2m'] = scenario_analysis['maxt2m'] * 1.25
    scenario_analysis['mint2m'] = scenario_analysis['mint2m'] * 1.25

dataPath = Path('../data/Spain/Spain_ETp_Copernicus_CDS/')
# Save the scenario datasets to new NetCDF files
analysis_xr.to_netcdf(dataPath/'era5_scenario_ref.nc')
scenario_analysis.to_netcdf(f'{dataPath}/era5_scenario{args.scenario_nb}_weather_{args.weather_scenario}.nc')
# scenario_analysis

# scenario2.to_netcdf(dataPath/'era5_scenario2_precipitation_minus20.nc')
# scenario3.to_netcdf(dataPath/'era5_scenario3_temperature_plus25.nc')
# scenario1.to_netcdf(dataPath/'era5_scenario1_precipitation_plus20.nc')
# scenario2.to_netcdf(dataPath/'era5_scenario2_precipitation_minus20.nc')
# scenario3.to_netcdf(dataPath/'era5_scenario3_temperature_plus25.nc')

wdf = analysis_xr.to_dataframe()
wdf = wdf.reset_index()
wdf = wdf.rename(columns={
    'valid_time': 'Date', 
    'maxt2m': 'MaxTemp', 
    'mint2m': 'MinTemp', 
    'sumpev': 'ReferenceET', 
    'sumtp': 'Precipitation'
})
wdf = wdf[['MinTemp','MaxTemp','Precipitation','ReferenceET','Date']]

#%%

fig, axs = plt.subplots(3,1,sharex=True)
analysis_xr.plot.scatter(x='valid_time',
                            y='sumpev',
                            ax=axs[0],
                            color='k',
                            s=2
                            )
scenario_analysis.plot.scatter(x='valid_time',
                            y='sumpev',
                            ax=axs[0],
                            color='red',
                            s=2
                            )
analysis_xr.plot.scatter(x='valid_time',
                            y='maxt2m',
                            ax=axs[1],
                            color='k',
                            s=2
                            )
analysis_xr.plot.scatter(x='valid_time',
                            y='mint2m',
                            ax=axs[1],
                            color='k',
                            s=2
                            )
scenario_analysis.plot.scatter(x='valid_time',
                            y='mint2m',
                            ax=axs[1],
                            color='r',
                            s=2
                            )
scenario_analysis.plot.scatter(x='valid_time',
                            y='maxt2m',
                            ax=axs[1],
                            color='r',
                            s=2
                            )
scenario_analysis.plot.scatter(x='valid_time',
                        y='sumtp',
                        ax=axs[2],
                        color='r',
                        s=2
                        )
analysis_xr.plot.scatter(x='valid_time',
                        y='sumtp',
                        ax=axs[2],
                        color='k',
                        s=2
                        )
axs[0].set_title('')
axs[1].set_title('')
axs[2].set_title('')
axs[0].set_xlabel('')
axs[1].set_xlabel('')

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
sim_start = wdf.Date.iloc[0].strftime('%Y/%m/%d')
sim_end = wdf.Date.iloc[365].strftime('%Y/%m/%d')
soil= Soil('SandyLoam')
crop = Crop('Maize',
            planting_date='07/06'
            )
initWC = InitialWaterContent(value=['FC'])
labels=[]
outputs=[]
smt = args.SMT
crop.Name = str(smt) # add helpfull label
labels.append(str(smt))

# SMT (list):  Soil moisture targets (%taw) to maintain in each growth stage (only used if irrigation method is equal to 1)
irr_mngt = IrrigationManagement(irrigation_method=1,
                                SMT=[smt]*4 # same for each developement growth stages 
                                ) # specify irrigation management

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

# aa
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
# water_flux.columns
# water_storage = model.get_water_storage()

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

sc_EO = utils.check_and_tune_E0_dict(sc)

if args.ApplyEOcons==1:
    sc_withirr = sc | sc_EO
else:
    sc_withirr = sc

simu_with_IRR, grid_xr_with_IRR = scenarii2pyCATHY.setup_cathy_simulation(
                                                                         prj_name=prj_name, 
                                                                         scenario=sc_withirr,
                                                                         with_irrigation=True,
                                                                    )
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
grid_xr_with_IRR.to_netcdf(f'../prepro/grid_xr_EO_AquaCrop_sc{args.scenario_nb}_weather_{args.weather_scenario}.netcdf')
grid_xr_baseline.attrs = {}
grid_xr_baseline.to_netcdf(f'../prepro/grid_xr_baseline_AquaCrop_sc{args.scenario_nb}_weather_{args.weather_scenario}.netcdf')


