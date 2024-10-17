#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Feb 15 10:08:57 2024
@author: ben

Project EOMAJI (Earth Observation system to Manage Africa’s food systems by Joint-knowledge of crop production and Irrigation digitization) 
ET-Based algorithms for net irrigation estimation. 

Scenario 1
----------

- Domain dimension and discretisation:
    - Regional scale = 10x10km
    - Local scale = 300*300m
"""

import numpy as np
import matplotlib.pyplot as plt 
from pyCATHY import CATHY
import rioxarray as rxr
import xarray as xr
# FIG_PATH = '../figures/'
import matplotlib.colors as mcolors
import pandas as pd
#%%
import xarray as xr
import numpy as np

           
def plot_schedule(grid_xr,pp='irr_daily',
                  vmin=0,
                  vmax=1e-9,
                  unit='mm/day'
                  ):
    # Number of time steps
    n_time_steps = len(grid_xr['time'])
    
    # Define the size of the subplot grid (e.g., 2 rows by 5 columns)
    n_cols = 5
    n_rows = (n_time_steps + n_cols - 1) // n_cols  # Compute rows needed    
    # Create the figure and subplots
    fig, axes = plt.subplots(n_rows, 
                             n_cols, 
                             figsize=(15, 10), 
                             constrained_layout=True)
    
    # Flatten the axes array for easy iteration
    axes = axes.flatten()
    
    # Plot each time step
    for i in range(n_time_steps):
        ax = axes[i]
        val = grid_xr[pp]
        if unit=='mm/day':
            val = grid_xr[pp]*(86400*1e3)
        im = val.isel(time=i).plot.imshow(ax=ax, cmap='viridis',
                                                       vmin=vmin,
                                                       vmax=vmax,
                                                       add_colorbar=False  # Disable the automatic colorbar
                                                      )
        ax.set_title(f'Time Day {i}')
        ax.label_outer()  # Hide labels on the inner plots to avoid clutter
        ax.set_aspect('equal')  # Set aspect ratio to be equal (square)
    
    # Hide any remaining empty subplots
    for j in range(n_time_steps, len(axes)):
        axes[j].axis('off')
    
    # Add a shared colorbar
    cbar = fig.colorbar(im, ax=axes, orientation='vertical', 
                        fraction=0.02, 
                        pad=0.02
                        )
    cbar.set_label(pp + ' ' + unit)  # Set the label for the colorbar
    
    # plt.show()

        
    return fig, axes

#%%

def get_irr_areas_colors(
                         irr_map,
                         colormap='Pastel1',
                         bounds=[],
                         ):
    cmap = mcolors.Colormap(colormap)
    bounds = np.linspace(0,len(np.unique(irr_map)))
    # cmap = mcolors.ListedColormap(['magenta', 'green', 'yellow','orange'])
    # bounds = [0.5, 1.5, 2.5, 3.5,4]  # Boundaries for the discrete values
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    # grid_xr['irrigation_map'].plot.imshow(cmap=cmap, norm=norm)
    return cmap, norm

def prepare_scenario(scenario,with_irrigation=True):
    '''
    Take the scenario disctionnary and create a rioxarray dataset containing: 
        - DEM raster
        - Irrigation map (raster with number for each irrigation plot)
        - ETp map
        - ...

    Parameters
    ----------
    scenario : dict
        description of the simulation inputs.

    Returns
    -------
    grid_xr : xarray dataset
    layers : dict
    '''
    # define grid size and resolution
    # -------------------------------------------------------------------------
    region_domain = set_regional_domain()
    dem = set_dem(region_domain)
    local_domain = set_local_domain()
    layers = set_layers()
    
    # save to rioxarray
    grid_xr = xr.DataArray(dem,
                            coords={'x':np.arange(0,
                                                  len(dem)*region_domain['x_spacing'],
                                                  region_domain['x_spacing']
                                                  ),
                                        'y':np.arange(0,
                                                      len(dem)*region_domain['y_spacing'],
                                                      region_domain['y_spacing']
                                                      )
                                        },
                            name='DEM'
                            # variable=dem,
                            )
    grid_xr = grid_xr.to_dataset()
    
    
    grid_xr.attrs['regional_domain'] = region_domain
    grid_xr.attrs['local_domain'] = local_domain
    grid_xr.attrs['layers'] = layers
    

    #%% define irrigation_map
    # -------------------------------------------------------------------------
    irr_zones = np.ones(np.shape(dem))
    grid_xr['irrigation_map'] = (('x', 'y'), irr_zones)
    for irr_i in range(len(scenario['irr_center_point_x'])):
        idx_irr_m = scenario['irr_center_point_x'][irr_i] - scenario['irr_square_size'][irr_i]/2
        idx_irr_p = scenario['irr_center_point_x'][irr_i] + scenario['irr_square_size'][irr_i]/2
        idy_irr_m = scenario['irr_center_point_y'][irr_i] - scenario['irr_square_size'][irr_i]/2
        idy_irr_p = scenario['irr_center_point_y'][irr_i] + scenario['irr_square_size'][irr_i]/2
        grid_xr['irrigation_map'].loc[
            dict(
                x=slice(idx_irr_m, idx_irr_p),
                y=slice(idy_irr_m, idy_irr_p)
            )
        ] = irr_i + 2 
        
    #%% Compute ETp 
    # -------------------------------------------------------------------------
    if len(scenario['ETp'])==1:
        ETp_daily = set_ETp_daily(scenario)
    else:
        ETp_daily = scenario['ETp']

    ETp_daily_spatial = [np.ones([len(grid_xr.x),len(grid_xr.y)])*valETpTi for valETpTi in ETp_daily]

    nb_days = scenario['nb_days']
    
    time_days = pd.to_timedelta(np.arange(0,nb_days,1), unit='D')
    grid_xr['time']=time_days
    grid_xr['time_hours']=np.arange(0,nb_days*24,1)
    grid_xr['ETp_daily'] = (('time','x','y'), ETp_daily_spatial)

    if len(ETp_daily_spatial)<=15:
        fig, axes = plot_schedule(grid_xr,'ETp_daily',vmax=10,unit='mm/day')
        fig.savefig(scenario['figpath'] /  'ETp_daily.png', dpi=300)
    
    #%% Compute irr_daily 
    # -------------------------------------------------------------------------
    irr_daily = [np.zeros(np.shape(dem))]*len(grid_xr['time'])
    grid_xr['irr_daily'] = (('time','x','y'), irr_daily)
    if with_irrigation:
        for i , irrzonei in enumerate(np.unique(grid_xr['irrigation_map'])[1:]):
            for j, irr_ti in enumerate(scenario['irr_time_index'][i]):
                irr_ti_pd = pd.to_timedelta(irr_ti, unit='D')
                mask = grid_xr['irrigation_map'] == irrzonei #(i + 2)
                updated_value = scenario['irr_flow'][i][j] # * scenario['irr_length'][i] #/ (60 * 60)
                grid_xr['irr_daily'].loc[
                    dict(time=irr_ti_pd)
                ] = np.where(mask, updated_value, 
                             grid_xr['irr_daily'].loc[
                                                      dict(time=irr_ti_pd)
                                                     ]
                             )                
        if len(ETp_daily_spatial)<=15:
            fig, axes = plot_schedule(grid_xr,'irr_daily',vmax=10,unit='mm/day')
            fig.savefig(scenario['figpath'] /  'irr_schedule.png', dpi=300)
        
    #%% define rain map
    # -------------------------------------------------------------------------
    rain_daily = [np.zeros(np.shape(dem))]*len(grid_xr['time'])
    grid_xr['rain_daily'] = (('time','x','y'), rain_daily)
    if 'rain_time_index' in scenario:
        for i, rain_ti in enumerate(scenario['rain_time_index']):
              rain_ti_pd = pd.to_timedelta(rain_ti, unit='D')
              updated_value = scenario['rain_flow'][i] #* scenario['rain_length'][i] / (60 * 60)
              grid_xr['rain_daily'].loc[dict(time=rain_ti_pd)] =  updated_value
    if len(ETp_daily_spatial)<=15:
        fig, axes = plot_schedule(grid_xr,'rain_daily',vmax=10,unit='mm/day')
        fig.savefig(scenario['figpath'] /  'rain_schedule.png', dpi=300)

    #%% Compute net atmbc
    # -------------------------------------------------------------------------
    grid_xr['net_atmbc'] = (grid_xr['rain_daily'] + grid_xr['irr_daily']) - abs(grid_xr['ETp_daily'])

    if with_irrigation:
        # grid_xr['irr_daily'].max()
        # grid_xr['ETp_daily'].max()
        # grid_xr['net_atmbc'].max()
        # grid_xr['net_atmbc'].min()
        if len(ETp_daily_spatial)<=15:
            fig, axes = plot_schedule(grid_xr,'net_atmbc',vmax=10,unit='mm/day')
            fig.savefig(scenario['figpath'] /  'net_atmbc_schedule.png', dpi=300)


    # grid_xr['net_atmbc'].max()
    # grid_xr['net_atmbc'].min()

    # set_regional_domain
    
    
    return grid_xr, layers 


def set_regional_domain():
    x_dim = 1 * 1e3  # length in m
    y_dim = 1 * 1e3  # length in m
    x_spacing = 10  # x discretisation in m
    y_spacing = 10
    region_domain = {
        'x_dim': x_dim,
        'y_dim': y_dim,
        'x_spacing': x_spacing,
        'y_spacing': y_spacing,
    }
    return region_domain


def set_local_domain():
    x_dim = 100  # length in m
    y_dim = 100  # length in m
    local_domain = {
        'x_dim': x_dim,
        'y_dim': y_dim,
    }
    return local_domain


def set_dem(region_domain):
    dem = np.ones(
        [    
            int(region_domain['x_dim'] / region_domain['x_spacing']),
            int(region_domain['y_dim'] / region_domain['y_spacing']),
        ]
    )
    dem[-1, -1] = 1 - 1e-3  # fake outlet
    return dem
    

def set_layers():
    nb_layers = 5
    layers_depth = [0.1, 0.2, 0.5, 2]

    layers = {
        'nb_layers': nb_layers,
        'layers_depth': layers_depth,
    }
    return layers 

def get_irr_coordinates(local_domain,
                        region_domain,
                        dem,
                        ):
    
    zone_local = [int(local_domain['x_dim'] / region_domain['x_spacing']),
                  int(local_domain['y_dim'] / region_domain['y_spacing']),
                  ]
    irr_zones = np.ones(np.shape(dem))
    center_zones = [np.shape(irr_zones)[0] / 2, np.shape(irr_zones)[1] / 2]
    start_local_idx = int(center_zones[0] - zone_local[0] / 2)
    end_local_idx = int(center_zones[0] + zone_local[0] / 2)
    start_local_idy = int(center_zones[1] - zone_local[1] / 2)
    end_local_idy = int(center_zones[1] + zone_local[1] / 2)
    return center_zones, start_local_idx, end_local_idx, start_local_idy, end_local_idy

def set_irrigation_zone(dem, 
                        region_domain, 
                        local_domain, 
                        sc
                        ):
    
    if sc['nb_irr_zones'] == 1:
        # nb_zones = 2
        irr_zones = np.ones(np.shape(dem))
    
        (center_zones, 
         start_local_idx, 
         end_local_idx, 
         start_local_idy, 
         end_local_idy) =  get_irr_coordinates(local_domain,
                                               region_domain,
                                               dem,
                                               )
    
        irr_zones[start_local_idx:end_local_idx, 
                  start_local_idy:end_local_idy] = 2
        
    if sc['nb_irr_zones'] == 2:
        pass


    
    fig, ax = plt.subplots()
    ax.imshow(irr_zones)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    fig.savefig(sc['figpath'] /  'irr_zones.png', dpi=300)
    
    return irr_zones


def set_rain():
    # no rain
    pass

def set_ETp_daily(scenario):
    nb_days = scenario['nb_days']
    # nb_hours_ET = scenario['nb_hours_ET'] 
    ETp_m_s = scenario['ETp']
    ETp = []
    for ndi in range(nb_days):
        ETp.append(ETp_m_s)
        # ETp.append(ETp_m_s/nb_hours_ET)
        # ETp.append(np.zeros(int(1/2)))
    ETp = np.hstack(ETp)
    
    return ETp


def set_ETp_hourly(scenario):
    nb_days = scenario['nb_days']
    nb_hours_ET = scenario['nb_hours_ET'] 
    ETp_m_s = scenario['ETp']
    ETp = []
    for ndi in range(nb_days):
        ETp.append(np.ones(int(86400/2))*ETp_m_s)
        ETp.append(np.zeros(int(86400/2)))
    ETp = np.hstack(ETp)
    
    return ETp

def pad_raster_2mesh(raster,
                     pad_width= ((0, 1), (0, 1)),
                     mode='edge',
                     **kwargs
                     ):
    # Raster zones to mesh nodes
    # ----------------------------------------------------------------------
    # pad_width = ((0, 1), (0, 1))  # Padding only the left and top
    padded_zones = np.pad(raster, 
                           pad_width, 
                           mode=mode, 
                           **kwargs
                           )
    
    # plt.imshow(padded_zones)
    padded_zones_1d = padded_zones.flatten()
    return padded_zones, padded_zones_1d

def pad_zone_2mesh(zones):
    # Raster zones to mesh nodes
    # ----------------------------------------------------------------------
    pad_width = ((0, 1), (0, 1))  # Padding only the left and top
    padded_zones = np.pad(zones, 
                           pad_width, 
                           mode='constant', 
                           constant_values=1
                           )
    
    # plt.imshow(padded_zones)
    padded_zones_1d = padded_zones.flatten()
    return padded_zones, padded_zones_1d

#%%

def setup_cathy_simulation(
                            prj_name, 
                            scenario,
                            with_irrigation=False,
                            **kwargs
                           ):
    
    #%% Transform scenario inputs to a grid xarray dataset 
    # -------------------------------------------------------------------------
    (grid_xr, layers) = prepare_scenario(scenario,
                                         with_irrigation
                                         )

    #%% Define mesh depth
    # -------------------------------------------------------------------------
    maxdepth = layers['layers_depth'][-1]
    if 'maxdepth' in kwargs:
        maxdepth = kwargs['sc_EO'].pop('maxdepth')
        
    zb = np.geomspace(1e-1, maxdepth, num=layers['nb_layers'])
    zr = [abs(zb[0] / maxdepth)]
    zr.extend(list(abs(np.diff(zb) / maxdepth)))

    if with_irrigation:
        prj_name += '_withIRR'
    
    #%% Create a simulation object
    # -------------------------------------------------------------------------
    simu = CATHY(dirName='../WB_twinModels/', 
                 prj_name=prj_name
                 )

    simu.update_prepo_inputs(
        DEM=grid_xr['DEM'].values,
        nstr=layers['nb_layers'],
        zratio=zr,
        base=max(zb),
        delta_x=grid_xr.attrs['regional_domain']['x_spacing'],
        delta_y=grid_xr.attrs['regional_domain']['y_spacing'],
    )
    simu.update_parm(TRAFLAG=0)
    simu.create_mesh_vtk(verbose=True)
    simu.run_preprocessor(verbose=True)
    
    #%% Set a constant Evapotranspiration ETp = -1e-7 m/s, all over the domain
    # ----------------------------------------------------------------------
    print('!ETp_scenario is not yet implemented!')
    # ETp_scenario = scenario['ETp']   
    padded_netatmbc_all = []
    for i in range(len(grid_xr['net_atmbc'].values)):
        padded_netatmbc, _ = pad_raster_2mesh(grid_xr['net_atmbc'].isel(time=i).values)
        padded_netatmbc_all.append(padded_netatmbc)
    netValue= [np.hstack(net2d) for net2d in padded_netatmbc_all]
 
    # np.max(netValue)
    
    simu.update_atmbc(
                      HSPATM=0, # spatially variable atmbc
                      IETO=1,
                      time=list(grid_xr['time'].values.astype('timedelta64[s]').astype(int)),
                      netValue=netValue,
                      # show=True,
                    )
    simu.show_input('atmbc')

    #%% Update initial conditions
    # -------------------------------------------------------------------------
    simu.update_ic(INDP=0, 
                   IPOND=0,
                   pressure_head_ini=scenario['pressure_head_ini']
                   )
    
    simu.update_nansfdirbc(time=list(grid_xr['time'].values.astype('timedelta64[s]').astype(int)),
                           no_flow=True
                           )
    
    if 'nansfdirbc' in scenario.keys():
        
        print('Implementation in progress')
        # if scenario['nansfdirbc'] != 'no_flow':
        #     nansfdirbc_val = kwargs['nansfdirbc'].pop('nansfdirbc')
        #     simu.update_nansfdirbc(
        #                             time=t_atmbc,
        #                             nansfdirbc_val
        #                            )

    simu.update_nansfneubc(time=list(grid_xr['time'].values.astype('timedelta64[s]').astype(int)),
                           no_flow=True
                           )
    simu.update_sfbc(time=list(grid_xr['time'].values.astype('timedelta64[s]').astype(int)),
                     no_flow=True
                     )
    
    
    #%% Update zone and soil conditions
    # -------------------------------------------------------------------------
    nb_irr_plots = len(np.unique(grid_xr['irrigation_map'].values))
                       
    padded_zones, padded_zones_1d = pad_raster_2mesh(grid_xr['irrigation_map'].values,
                                                     mode='edge',                                                    
                                                     )
    
    simu.update_zone(grid_xr['irrigation_map'].values)
    
    simu.update_veg_map(grid_xr['irrigation_map'].values)
    
    fig, ax = plt.subplots()
    simu.show_input('root_map',ax=ax)
    fig.savefig(scenario['figpath'] /  'root_map.png', dpi=300)

    
    df_SPP_map = simu.init_soil_SPP_map_df(len(np.unique(grid_xr['irrigation_map'].values)),
                                               layers['nb_layers'],
                                               )
    

    
    SPP_map = simu.set_SOIL_defaults(SPP_map_default=True)
    
    # PERMX = [14,15,16]
    if 'PERMX' in scenario:
        PERMX = scenario['PERMX']
        for zone_i in range(len(PERMX)):
            for pp in ['PERMX','PERMY','PERMZ']:
                SPP_map.loc[zone_i+1,pp] = PERMX[zone_i]
    # if 'porosity' in scenario:
    #     porosity = scenario['porosity']
    #     for zone_i in range(len(porosity)):
    #         SPP_map.loc[zone_i+1,'porosity'] = porosity[zone_i]
                
                
    df_FP_map = simu.init_soil_FP_map_df(nb_irr_plots)
    FP_map = simu.set_SOIL_defaults(FP_map_default=True)
      
    if 'ZROOT' in scenario:
        ZROOT = scenario['ZROOT']
    else:
        ZROOT = [0.3,0.3]
    for zone_i in range(len(ZROOT)):
        FP_map.loc[zone_i+1,'ZROOT'] = ZROOT[zone_i]
        
    simu.update_soil(
                        PMIN=scenario['PMIN'],
                        SPP_map = SPP_map,
                        FP_map = FP_map,
                      )
    
    #%%
    fig, ax = plt.subplots()
    simu.show_input(prop="soil", yprop="ZROOT", ax=ax)
    fig.savefig(scenario['figpath'] /  'ZROOT.png', dpi=300)

    fig, ax = plt.subplots()
    simu.show_input(prop="soil", yprop="PERMX", layer_nb=1,
                    ax=ax)
    fig.savefig(scenario['figpath'] /  'PERMX.png', dpi=300)




    #%% Update simulation parameters
    # -------------------------------------------------------------------------
    # simu.update_parm(TIMPRTi=list(grid_xr['time'].values.astype('timedelta64[s]').astype(int)),
    #                  VTKF=4 # both saturation and pressure head
    #                  )
    
    simu.update_parm(TIMPRTi=list(grid_xr['time'].values.astype('timedelta64[s]').astype(int)),
                     VTKF=0 # both saturation and pressure head
                     )

    return simu, grid_xr