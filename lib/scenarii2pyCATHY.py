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

# FIG_PATH = '../figures/'


def prepare_scenario(scenario):
    region_domain = set_regional_domain()
    dem = set_dem(region_domain)
    local_domain = set_local_domain()
    layers = set_layers()
    zones = set_zone(dem, region_domain, local_domain, scenario)
    ETp = set_ETp(scenario)
    return region_domain, local_domain, dem, layers, zones, ETp


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


def set_zone(dem, region_domain, local_domain, sc):
    nb_zones = 2
    # np.shape(DEM)
    zones = np.zeros(np.shape(dem))
    zone_local = [int(local_domain['x_dim'] / region_domain['x_spacing']),
                  int(local_domain['y_dim'] / region_domain['y_spacing']),
                  ]
    center_zones = [np.shape(zones)[0] / 2, np.shape(zones)[1] / 2]
    start_local_x = int(center_zones[0] - zone_local[0] / 2)
    end_local_x = int(center_zones[0] + zone_local[0] / 2)

    start_local_y = int(center_zones[1] - zone_local[1] / 2)
    end_local_y = int(center_zones[1] + zone_local[1] / 2)

    zones[start_local_x:end_local_x, start_local_y:end_local_y] = 1

    fig, ax = plt.subplots()
    ax.imshow(zones)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    fig.savefig(sc['figpath'] + 'zones.png', dpi=300)
    
    return zones


def set_rain():
    # no rain
    pass

def set_ETp(scenario):
    nb_days = scenario['nb_days']
    nb_hours_ET = scenario['nb_hours_ET'] 
    ETp_m_s = scenario['ETp']
    ETp = []
    for ndi in range(nb_days):
        if ndi%2==0:
            ETp.append(np.ones(int(86400/2))*ETp_m_s)
        else:
            ETp.append(np.zeros(int(86400/2)))
    ETp = np.hstack(ETp)
    
    return ETp




#%%

def setup_cathy_simulation(
                            prj_name, 
                            scenario,
                            # ETp = -1e-7,
                            with_irrigation=False,
                            # irr_time_index = 5,
                            # irr_flow = 5e-7 #m/s
                           ):
    
    
    (region_domain, local_domain, dem, layers, zones, ETp_scenario) = prepare_scenario(scenario)
    
    
    print('!ETp_scenario is not yet implemented!')
    ETp_scenario = scenario['ETp']
    
    maxdepth = layers['layers_depth'][-1]
    zb = np.geomspace(1e-1, maxdepth, num=layers['nb_layers'])
    zr = [abs(zb[0] / maxdepth)]
    zr.extend(list(abs(np.diff(zb) / maxdepth)))



    if with_irrigation:
        prj_name += '_withIRR'
        
    simu = CATHY(dirName='../pyCATHY/', 
                 prj_name=prj_name
                 )
            
    simu.update_prepo_inputs(
        DEM=dem,
        nstr=layers['nb_layers'],
        zratio=zr,
        base=max(zb),
        delta_x=region_domain['x_spacing'],
        delta_y=region_domain['y_spacing'],
    )
    simu.update_parm(TRAFLAG=0)
    simu.create_mesh_vtk(verbose=True)
    simu.update_zone(zones)
    simu.run_preprocessor(verbose=True)

    # t_atmbc = [0,1e2,5e2,1e3,5e3,10e3,20e3]
    t_atmbc_EO = list(np.arange(0,scenario['nb_days']*86400,scenario['EO_freq_days']*86400))
    
    # define intermediate atmbc irrigation times 
    t_irr = t_atmbc_EO[scenario['irr_time_index']] + scenario['irr_length']
    
    t_atmbc_combined = np.concatenate((t_atmbc_EO, [t_irr]))

    # Sort the combined array
    t_atmbc = sorted(t_atmbc_combined)

    # t_atmbc = t_rain + t_irr + t_atmbc_EO

    # Set a constant Evapotranspiration ETp = -1e-7 m/s, all over the domain
    # ----------------------------------------------------------------------
    grid3d = simu.read_outputs('grid3d')
    grid3d['mesh3d_nodes'][int(grid3d['nnod'])]
    v_atmbc = np.ones(int(grid3d['nnod']))*ETp_scenario
    
    # Raster zones to nodes
    # ----------------------------------------------------------------------
    pad_width = ((0, 1), (0, 1))  # Padding only the left and top
    padded_zones = np.pad(zones, 
                           pad_width, 
                           mode='constant', 
                           constant_values=0
                           )
    
    # plt.imshow(padded_zones)
    padded_zones_1d = padded_zones.flatten()
    irr_zone_id = np.where(padded_zones_1d==1)
    
    if with_irrigation:        
        v_atmbc_withirr = v_atmbc.copy()
        v_atmbc_withirr[irr_zone_id] += scenario['irr_flow']
        netValue_tmp = [v_atmbc] * len(t_atmbc)
        netValue = np.copy(netValue_tmp)
        netValue[scenario['irr_time_index']] = v_atmbc_withirr        
    else:
        netValue = [v_atmbc]*len(t_atmbc)
       
    with_rain = False
    print('!with_rain is not yet implemented!')
    if with_rain:
        pass
        
    simu.update_atmbc(
                      HSPATM=0, # spatially variable atmbc
                      IETO=0,
                      time=t_atmbc,
                      netValue=netValue,
                      # show=True,
                    )


    simu.update_ic(INDP=0, 
                   IPOND=0,
                   pressure_head_ini=-30
                   )
    
    simu.update_soil(PMIN=-1e30,
                     )
    
    simu.update_parm(TIMPRTi=t_atmbc,
                     VTKF=2 # both saturation and pressure head
                     )
    
    simu.update_nansfdirbc(time=t_atmbc,no_flow=True)
    simu.update_nansfneubc(time=t_atmbc,no_flow=True)
    simu.update_sfbc(time=t_atmbc,no_flow=True)
    

    return simu, netValue