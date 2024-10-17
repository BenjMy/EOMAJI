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
                               default=0, 
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
cx.add_basemap(ax, 
                crs=cropped_ERA5ds.rio.crs,
                alpha=0.4,
                # credit=False
                )
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
prj_name = f'EOMAJI_AquaCrop_sc{args.scenario_nb}_weather_{args.weather_scenario}_SMT_{args.SMT}' 
sc['figpath'] = figpath
#%% Simulate with irrigation atmospheric boundary conditions
# ----------------------------------------------------------

def check_and_tune_E0_dict(sc):
    '''
    Add EO criteria (resolution, frequency, type, ...)
    '''
    sc_EO = {}
    if sc.get('microwaweMesh'):
        sc_EO.update({'maxdepth': 0.05})
    if sc.get('EO_freq'):
        sc_EO.update({'EO_freq': sc.get('EO_freq')})
    if sc.get('EO_resolution'):
        sc_EO.update({'EO_resolution': sc.get('EO_resolution')})
        
    return sc_EO

sc_EO = check_and_tune_E0_dict(sc)

simu_with_IRR, grid_xr_with_IRR = scenarii2pyCATHY.setup_cathy_simulation(
                                            prj_name=prj_name,
                                            scenario=sc,
                                            # ETp = ETp,
                                            with_irrigation=True,
                                            sc_EO=sc_EO,
                                            # irr_time_index = 5,
                                            # irr_flow = 5e-7 #m/s
                                            )
# grid_xr_with_IRR.irr_daily.sum()

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



# IRRIGATION DELIMITATION
# --------------------------------------------------
#%% Read and plot outputs (scenario with irrigation)
# --------------------------------------------------
print('Read outputs - can take a while')
out_with_IRR = utils.read_outputs(simu_with_IRR)

out_baseline = utils.read_outputs(simu_baseline)

#%%

#%% 
# ETp = np.ones(np.shape(simu_with_IRR.DEM))*sc['ETp']
ds_analysis_EO = utils.get_analysis_ds(out_with_IRR['ETa'])
ds_analysis_baseline = utils.get_analysis_ds(out_baseline['ETa'])
# grid_xr_with_IRR = grid_xr_with_IRR.rename({'time_days': 'time'})
grid_xr_with_IRR = grid_xr_with_IRR.assign_coords(x=grid_xr_with_IRR['x'].astype('float64'),
                                                  y=grid_xr_with_IRR['y'].astype('float64'),
                                                  # time=pd.to_timedelta(grid_xr_with_IRR['time'], unit='D')
                                                  )
grid_xr_with_IRR_interp = grid_xr_with_IRR['ETp_daily'].interp_like(ds_analysis_EO, 
                                                                    method="linear"
                                                                    )
ds_analysis_EO["ETp"] = grid_xr_with_IRR_interp 
ds_analysis_EO["datetime"] = sc['datetime']

# ds_analysis_EO = utils.add_ETp2ds(ETp,ds_analysis_EO)
# ds_analysis_baseline = utils.add_ETp2ds(ETp,ds_analysis_baseline)

time_sel = np.arange(0,len(ds_analysis_EO.time),50)
ds_analysis_EO['ETp'].isel(time=time_sel).plot.imshow(x="x", y="y", 
                                                      col="time", 
                                                      col_wrap=4
                                                      )
plt.title('ETp EO')
plt.savefig(os.path.join(figpath,'ETp.png'),
            dpi=300,
            )

#%%
mask_IN = utils.get_mask_IN_patch_i(grid_xr_with_IRR['irrigation_map'],
                              patchid=2
                              )
mask_OUT = utils.get_mask_OUT(grid_xr_with_IRR['irrigation_map'],
                              )

fig, axs = plt.subplots(2,1,sharex=True,figsize=(12,9))
utils.plot_july_rain_irr(sc['datetime'], 
                         grid_xr_with_IRR, 
                         mask_IN,
                         axs=axs
                         )
fig.savefig(os.path.join(figpath,
                         'july_rain_irr.png'
                         ),
            dpi=300,
            )


#%% Plot evolution of ETa, SW and PSI INSIDE and OUTSIDE of the irrigation area 
# 1D + time
atmbc_df = simu_with_IRR.read_inputs('atmbc')
nb_irr_areas = len(np.unique(grid_xr_with_IRR.irrigation_map)) -1
(irr_patch_centers, 
 patch_centers_CATHY) = utils.get_irr_center_coords(irrigation_map=grid_xr_with_IRR['irrigation_map'])   
maxDEM = simu_with_IRR.grid3d['mesh3d_nodes'][:,2].max()
out_irr = np.where(grid_xr_with_IRR.irrigation_map==1)

start_date = sc['datetime'].values[0]




time_deltas = np.array([np.timedelta64(i, 'D') for i in range(len(grid_xr_with_IRR.time))])  # Example data
dates = [start_date + delta for delta in time_deltas]
dates_pd = pd.to_datetime(dates) 

# sw2plot_with_IRR =  out_with_IRR['psi'].iloc[:,1].values
# len(sw2plot_with_IRR)
# sw2plot_with_IRR =  out_baseline['sw'].iloc[:,1]
# len(sw2plot_with_IRR)


fig, axs = plt.subplots(4,nb_irr_areas,
                        sharex=True,
                        sharey=False,
                        figsize=(7,5)
                        )

utils.plot_patches_irrigated_states(
                                    irr_patch_centers,
                                    patch_centers_CATHY,
                                    simu_with_IRR,
                                    maxDEM,
                                    grid_xr_with_IRR,
                                    sc,
                                    axs,
                                    out_with_IRR,
                                    out_baseline,
                                    dates_pd,
                                  )


fig.savefig(os.path.join(figpath,
                         'plot_1d_evol_irrArea.png'
                         ),
            dpi=300,
            )

#%%
fig, axs = plt.subplots(4,nb_irr_areas,
                        sharex=True,
                        sharey=False,
                        figsize=(15,5)
                        )

utils.plot_patches_NOirrgation_states(
                                        simu_with_IRR,
                                        out_irr,
                                        maxDEM,
                                        grid_xr_with_IRR,
                                        out_with_IRR,
                                        out_baseline,
                                        axs,
                                        sc,
                                        dates
                                    )

# utils.custum_axis_patches_states(axs,
#                                  irr_patch_centers,
#                                  )
                
# axs[0].set_title('WITHIN Irrigation Area')
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

fig.savefig(os.path.join(figpath,
                         'plot_1d_evol_OUTirrArea.png'
                         ),
            dpi=300,
            )

#%% Apply EO rules

# ds_analysis_EO_ruled = utils.apply_EO_rules(ds_analysis_EO,sc_EO)
ds_analysis_EO_ruled = utils.apply_EO_rules(ds_analysis_EO,
                                            sc_EO
                                            )

#%% irrigation_delineation

event_type = utils.irrigation_delineation(ds_analysis_EO_ruled) 
   
#%% Plot timeline 
# ncols = 4
# time_steps = event_type.time.size
# nrows = int(np.ceil(time_steps / ncols))  # Number of rows needed
# fig, axes = plt.subplots(nrows=nrows, ncols=ncols, 
#                          figsize=(15, nrows * 3)
#                          )
# utils.plot_irrigation_schedule(event_type,time_steps,fig,axes)
# plt.savefig(os.path.join(figpath, 'classify_events.png'))


event_type_node_IN = event_type.where(mask_IN, drop=True).mean(['x','y'])
event_type_node_OUT = event_type.where(mask_OUT, drop=True).mean(['x','y'])

#%%
import july
dates_dt = [pd.to_datetime(date).to_pydatetime() for date in ds_analysis_EO_ruled.datetime.values]
fig, axs = plt.subplots(2,1,sharex=True,figsize=(12,9))

im = july.heatmap(dates_dt, 
              event_type_node_IN.values, 
              title='Irrigation detected',
              # cmap=white_cmap,
              ax=axs[0],
              linewidth=1, 
              value_label=True,
              )

im = july.heatmap(dates_dt, 
              event_type_node_IN.values, 
              title='Irrigation detected',
              # cmap=white_cmap,
              ax=axs[1],
              linewidth=1, 
              value_label=True,
              )

utils.plot_july_rain_irr(sc['datetime'], 
                         grid_xr_with_IRR, 
                         mask_IN,
                         axs=axs,
                         )                    
fig.savefig(os.path.join(figpath,
                          'july_rain_irr.png'
                          ),
            dpi=300,
            )

#%% Vheck detection quality



mask_irr_solution = (grid_xr_with_IRR['irr_daily'].where(mask_IN, drop=True).mean(['x','y']) > 0).values
mask_irr_detection = (event_type_node_IN == 1).values
common_mask = mask_irr_solution & mask_irr_detection
nbIrr_detection =  np.sum(common_mask)
nbIrr_solution = np.sum(mask_irr_solution)
perc_detection = (nbIrr_detection/nbIrr_solution)*100

dates = sc['datetime']
mean_irr_daily = grid_xr_with_IRR['irr_daily'].where(mask_IN, drop=True).mean(['x', 'y']).values
mean_rain_daily = grid_xr_with_IRR['rain_daily'].mean(['x','y'])


# Create a list of colors based on the common_mask
colors = ['red' if is_common else 'blue' for is_common in common_mask]

# Plot the bar chart
fig, ax = plt.subplots()
ax.bar(dates, 
       mean_irr_daily, 
       color=colors)

ax.bar(dates, 
        mean_rain_daily, 
        color='g')

# Format the x-axis with datetime labels
ax.xaxis.set_major_locator(mdates.AutoDateLocator())  # Automatically set major ticks
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Set date format

# Rotate the x-axis labels for better readability
plt.xticks(rotation=45)

plt.xlabel('Date')
plt.ylabel('Irrigation Daily Mean')
plt.title(f'%of detected irr events={perc_detection}%')


# Add legend for red and blue bars
red_patch = mpatches.Patch(color='red', label='Detected')
blue_patch = mpatches.Patch(color='blue', label='Not Detected')
ax.legend(handles=[red_patch, blue_patch])

# Optional: Auto-adjust layout to prevent overlap
plt.tight_layout()

fig.savefig(os.path.join(figpath,
                          'Irrigation_detection.png'
                          ),
            dpi=300,
            )


# common_true_indices = common_mask.values[common_mask]

# ds_analysis_EO['event_type_node_IN'] = event_type_node_IN


# ds_analysis_quality = ds_analysis_EO
# ds_analysis_quality



# quality_check



#%%

# Detected irrigation events are further split into low, medium and high probability based on another set
# of thresholds. Since irrigation is normally applied on a larger area, the raster map with per-pixel
# irrigation events is cleaned up by removing isolated pixels in which irrigation was detected.

def set_probability_levels():
    print('to implement')
    pass
    
#%%
# args.scenario_nb=0
# args.weather_scenario=0
grid_xr_with_IRR.attrs = {}
grid_xr_with_IRR.to_netcdf(f'../prepro/grid_xr_EO_AquaCrop_sc{args.scenario_nb}_weather_{args.weather_scenario}.netcdf')
grid_xr_baseline.attrs = {}
grid_xr_baseline.to_netcdf(f'../prepro/grid_xr_baseline_AquaCrop_sc{args.scenario_nb}_weather_{args.weather_scenario}.netcdf')

ds_analysis_EO['time'] = ds_analysis_EO['time'].astype('timedelta64[D]')
ds_analysis_EO.to_netcdf(f'../prepro/EO_scenario_AquaCrop_sc{args.scenario_nb}_weather_{args.weather_scenario}.netcdf')

ds_analysis_baseline['time'] = ds_analysis_EO['time'].astype('timedelta64[D]')
ds_analysis_baseline.to_netcdf(f'../prepro/baseline_scenario_AquaCrop_sc{args.scenario_nb}_weather_{args.weather_scenario}.netcdf')


