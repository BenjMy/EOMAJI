#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 10:46:53 2024
"""

import numpy as np
from pyCATHY import CATHY
# from pyCATHY.plotters import cathy_plots as cplt
from pyCATHY.importers import cathy_outputs as out_CT

import os
import matplotlib.pyplot as plt

plt.close('all')

import pandas as pd
from pathlib import Path
import utils

#%%
# C   ARENOD(NNOD)       - area assigned to each surface node
# C                        (needed for conversion of atmospheric
# C                        rainfall/evaporation rates to
# C                        volumetric fluxes)
# C   ATMPOT(NNOD)       - precipitation (+ve) / evaporation (-ve) fluxes
# C                        at current time level for each surface node.
# C                        These are potential infiltration/exfiltration
# C                        values. 
# C   ATMACT(NNOD)       - actual fluxes (infiltration/exfiltration
# C                        values) for atmospheric boundary
# C                        condition nodes at current time level. 
# C                        For IFATM(I)=0,  ATMACT(I) = ATMPOT(I);
# C                        For IFATM(I)=1,  ATMACT(I) = back-calculated
# C                        flux value;
# C                        For IFATM(I)=-1, ATMACT(I) is disregarded.
# C   ATMOLD(NNOD)       - ATMACT values at previous time level
# C   ATMTIM(3)          - most current input time values for atmospheric
# C                        BC's, with ATMTIM(1) < ATMTIM(2) < ATMTIM(3)
# C                        and ATMTIM(2) < TIME <= ATMTIM(3)
# C   ATMINP(3,NNOD)     - input atmospheric rainfall/evaporation rates
# C                        corresponding to ATMTIM times. ATMPOT(I) is
# C                        obtained from ATMINP(2,I) and ATMINP(3,I) by
# C                        linear interpolation and conversion of rate
# C                        to volumetric flux. ATMINP(1,I) values are 
# C                        needed in the event that, after back-stepping,
# C                        we have ATMTIM(1) < TIME <= ATMTIM(2)

# 
#%%
simu = CATHY(dirName='../pyCATHY/', 
             # prj_name='test_ET_1cell_IETO0'
             prj_name='test_ET_1cell_IETO1_withETp'
             )
dem = np.ones([3,3])
dem[-1,-1] = dem[-1,-1]-1e-3

simu.update_prepo_inputs(
    DEM=dem,
    # delta_x=0.5,
    # delta_y=0.5,
    delta_x=1,
    delta_y=1,
)

# simu.create_mesh_vtk(verbose=True)
simu.run_preprocessor(verbose=True)
# simu.update_zone(irr_zones)
# simu.update_ic(pressure_head_ini=-300)
# simu.update_ic(INDP=3, IPOND=0, 
#                WTPOSITION=1
#                )

simu.update_ic(INDP=0, IPOND=0,
               pressure_head_ini=-2)

simu.update_soil(PMIN=-1e35)

net_rain = 5e-6 
net_etp = -5e-6 


times = [0,3600,3600*4,3600*8,3600*12,3600*16,3600*20]
simu.update_atmbc(
                  HSPATM=1,
                  IETO=0,
                  time=times,
                  # netValue=[net_rain,net_etp,0.0],
                  # netValue=[net_rain,0.0,net_etp,net_etp,0.0,0.0],
                    netValue=[0,0,net_etp,0,0.0,0.0],
                  # netValue=[net_rain,0.0,0,0,0.0,0.0],
                   # netValue=[net_rain,0,0,net_rain,0,0],
                  # show=True,
                )


simu.update_nansfdirbc(no_flow=True)
simu.update_nansfneubc(no_flow=True)
simu.update_sfbc(no_flow=True)


simu.update_parm(
                        TIMPRTi=times,
                        # TIMPRTi=resample_times_vtk,
                        IPRT=4,
                        VTKF=4,
                        )



figpath = Path(simu.workdir) / simu.project_name
simu.run_processor(verbose=True,
                   IPRT1=2,
                   TRAFLAG=0
                   )

#%%
mesh3d = simu.read_outputs('grid3d')
nnod = mesh3d['nnod']
nnod3 = mesh3d['nnod3']




#%%

simu.soil_FP


#%%
atmbc_df = simu.read_inputs('atmbc')


dtcoupling_file = os.path.join(simu.workdir,
                        simu.project_name,
                        'output/dtcoupling'
                        )
dtcoupling_file = open(dtcoupling_file, "r")
lines = dtcoupling_file.readlines()
dtcoupling_file.close()
TOTALAREA = float(lines[3].split('=')[1])


#%% Plot df_hgatmsf outputs
# -------------------------
hgatmsf_file = os.path.join(simu.workdir,
                        simu.project_name,
                        'output/hgatmsf'
                        )


df_hgatmsf = out_CT.read_hgatmsf(hgatmsf_file)


# 'OVL. FLUX', # The superficial flow, that is the difference between POT.FLUX and ACT.FLUX;
# 'RET. FLUX', # flusso che dal sotterraneo va al super ciale (return flux);  


parms = [
        "RET. FLUX",
        'OVL. FLUX',
        'SEEP FLUX',   
        'REC. FLUX',   
        # 'REC.VOL.',
        ]
     
markers = ['^','v','*','.']

fig, ax = plt.subplots()
# axs = axs.ravel()
for i, pp in enumerate(parms): 
    print(pp)
    df_hgatmsf.set_index('TIME').plot(y=pp,
                                      ax=ax,
                                      label=pp,
                                      marker=markers[i]
                                      )

    ax.set_title(pp)
    
        
    # TOTALAREA*net_rain
    ax.axhline(y=TOTALAREA*net_rain, 
                color='k', 
                linestyle='--', 
                label='TOTALAREA*net_rain')

        
    ax.axvline(x=atmbc_df.time[1], 
                color='k', 
                linestyle='--', 
                label='end rain'
                )
    ax.grid(which='both')
    ax.minorticks_on()

    ax.set_ylabel('Flux (m3/day)')
    
fig.savefig(figpath / 'fluxes.png')



parms = [
        'ACT. FLUX',
        'POT. FLUX',
        # 'SEEP FLUX',   
        # 'REC. FLUX',   
        # 'REC.VOL.',
        ]
     
markers = ['^','v','*','.']

fig, ax = plt.subplots()
# axs = axs.ravel()
for i, pp in enumerate(parms): 
    print(pp)
    df_hgatmsf.set_index('TIME').plot(y=pp,
                                      ax=ax,
                                      label=pp,
                                      marker=markers[i]
                                      )

    ax.set_title(pp)
    
        
    # TOTALAREA*net_rain
    ax.axhline(y=TOTALAREA*net_rain, 
                color='k', 
                linestyle='--', 
                label='TOTALAREA*net_rain')

        
    ax.axvline(x=atmbc_df.time[1], 
                color='k', 
                linestyle='--', 
                label='end rain'
                )
    ax.grid(which='both')
    ax.minorticks_on()

    ax.set_ylabel('Flux (m3/day)')
    
fig.savefig(figpath / 'act_pot_fluxes.png')

#%% Plot recharge output
# -------------------------
# sss
recharge_file = os.path.join(simu.workdir,
                        simu.project_name,
                        'output/recharge'
                        )
recharge = out_CT.read_recharge(recharge_file)
# ETa_xarray = ETa.set_index(['time', 'x', 'y']).to_xarray()

recharge_xr = recharge.set_index(['time', 'x', 'y']).to_xarray()

# fig, ax = plt.subplots()
recharge_xr['recharge'].plot.imshow(x="x", y="y", 
                                    col="time", 
                                    col_wrap=4,
                                    )
plt.savefig(figpath / 'spatial_recharge.png',)


#%% Plot ET outputs
# -------------------------
# sss
fort777 = os.path.join(simu.workdir,
                        simu.project_name,
                        'fort.777'
                        )
ETa = out_CT.read_fort777(fort777)
ETa_xarray = ETa.set_index(['time', 'x', 'y']).to_xarray()

# fig, ax = plt.subplots()
ETa_xarray['ACT. ETRA'].plot.imshow(x="x", y="y", 
                                    col="time", 
                                    col_wrap=4,
                                    )
plt.savefig(figpath / 'spatial_ETa.png',)

# import xarray as xr

# Define resampling functions for each variable
resampling_functions = {
    'ACT. ETRA': 'sum',    # Use last value for 'ACT. ETRA'
}

import xarray as xr

# Apply resampling to each variable with the specified function
resampled_vars = {}
for var, func in resampling_functions.items():
    resampled_vars[var] = getattr(ETa_xarray[var].resample(time='1D'), func)()

# Combine the resampled variables into a new dataset
resampled_ds = xr.Dataset(resampled_vars)


# Resample using '1D' for daily frequency, specify aggregation (mean, sum, etc.)
# daily_ds = ETa_xarray.resample(time='1D', skipna=True).sum()

# print(daily_ds)

#%%

#Atmpot-vf (9) : Potential atmospheric forcing (rain +ve / evap -ve) as a volumetric flux [L^3/T]
#Atmpot-v (10) : Potential atmospheric forcing volume [L^3] (See parm input file for units)
#Atmpot-r (11) : Potential atmospheric forcing rate [L/T]
#Atmpot-d (12) : Potential atmospheric forcing depth [L]
#Atmact-vf(13) : Actual infiltration (+ve) or exfiltration (-ve) at atmospheric BC nodes as a volumetric flux [L^3/T]
#Atmact-v (14) : Actual infiltration (+ve) or exfiltration (-ve) volume [L^3]
#Atmact-r (15) : Actual infiltration (+ve) or exfiltration (-ve) rate [L/T]
#Atmact-d (16) : Actual infiltration (+ve) or exfiltration (-ve) depth [L]

#%%
dtcoupling_file = os.path.join(simu.workdir,
                        simu.project_name,
                        'output/dtcoupling'
                        )

dtcoupling = out_CT.read_dtcoupling(dtcoupling_file)

np.shape(dtcoupling['Atmpot-v'])
simu.grid3d

# dtcoupling_irr = dtcoupling[dtcoupling.Time <= 12e3]
dtcoupling['Atmpot-v_cumsum'] = dtcoupling['Atmpot-v'].cumsum()
dtcoupling['Atmpot-r_cumsum'] = dtcoupling['Atmpot-r'].cumsum()
dtcoupling['Atmact-v_cumsum'] = dtcoupling['Atmact-v'].cumsum()
dtcoupling['Atmact-r_cumsum'] = dtcoupling['Atmact-r'].cumsum()
dtcoupling['Atmact-vf_cumsum'] = dtcoupling['Atmact-vf'].cumsum()
# Atmact-vf

dtcoupling['Atmpot-v_cumsum'].max()
# fig, axs = plt.subplots(2,1,sharex=True)

# dtcoupling.set_index('Time').plot(y='Atmact-v',ax=axs[0])
# dtcoupling.set_index('Time').plot(y='Atmact-vf',ax=axs[1])

# fig, axs = plt.subplots(2,1,sharex=True)
# dtcoupling.set_index('Time').plot(y='Atmact-v_cumsum',ax=axs[0])
# dtcoupling.set_index('Time').plot(y='Atmact-r_cumsum',ax=axs[1])



# fig, axs = plt.subplots(2,1,sharex=True)
# dtcoupling.set_index('Time').plot(y='Atmpot-v',ax=axs[0])
# dtcoupling.set_index('Time').plot(y='Atmpot-r',ax=axs[1])

Atmactvmax = dtcoupling['Atmact-v'].max()
Atmactmax = dtcoupling['Atmact-r'].max()

max_atmpot_v_cumsum = dtcoupling['Atmpot-v_cumsum'].max()
max_atmpot_r_cumsum = dtcoupling['Atmpot-r_cumsum'].max()
max_atmact_v_cumsum = dtcoupling['Atmact-v_cumsum'].max()
max_atmact_r_cumsum = dtcoupling['Atmact-r_cumsum'].max()

max_atmact_vf_cumsum = dtcoupling['Atmact-vf_cumsum'].max()


max_atmpot_r_cumsum/max_atmpot_v_cumsum
max_atmpot_v_cumsum/max_atmpot_r_cumsum



# print(f'Max POT volume {Atmpot-v} [L]')
print(f'Max ACT {Atmactmax} [L/T]')
print(f'Max ACT volume {Atmactvmax} [L]')

print(f'Cum. Sum POT volume {max_atmpot_v_cumsum} [L]')
print(f'Cum. Sum ACT volume {max_atmact_v_cumsum} [L]')

print(f'Cum. Sum POT  {max_atmpot_r_cumsum} [L/T]')
print(f'Cum. Sum ACT  {max_atmact_r_cumsum} [L/T]')


# print(f'exfiltration {max_atmpot_v} [L]')
# print(f'exfiltration {max_atmpot_r} [L] [L/T]')


#%%

parms = [
        # "Atmpot-r_cumsum",
        "Atmact-r_cumsum",
        # "Atmpot-v_cumsum",
        "Atmact-v_cumsum",
        "Atmact-vf_cumsum",
        ]
fig, axs = plt.subplots(2,2,
                        sharex=True
                        )
axs = axs.ravel()

for i, pp in enumerate(parms): 
    print(pp)
    dtcoupling.set_index('Time').plot(y=pp,ax=axs[i])
    axs[i].set_title(pp)
    
    if pp == 'Atmpot-r':
        axs[i].axhline(y=atmbc_df.value[0], 
                    color='k', 
                    linestyle='--', 
                    label='input rain')
        
    axs[i].axvline(x=atmbc_df.time[1], 
                color='k', 
                linestyle='--', 
                label='end rain'
                )
    if '-v' in pp:
        axs[i].set_ylabel('[$m^{3}$]')
    else:
        axs[i].set_ylabel('[m/s]')

    
fig.savefig(figpath / 'Atm_cumsum.png')

#%%

parms = [
        "Atmpot-r",
        "Atmact-r",
        "Atmpot-v",
        "Atmact-v",
        ]
fig, axs = plt.subplots(2,2,
                        sharex=True
                        )
axs = axs.ravel()

for i, pp in enumerate(parms): 
    print(pp)
    simu.show(prop="dtcoupling", 
              yprop=pp,
              ax=axs[i]
              )
    axs[i].set_title(pp)
    
    if pp == 'Atmpot-r':
        axs[i].axhline(y=atmbc_df.value[0], 
                    color='k', 
                    linestyle='--', 
                    label='input rain')
        
    axs[i].axvline(x=atmbc_df.time[1], 
                color='k', 
                linestyle='--', 
                label='end rain'
                )
    
fig.savefig(figpath / 'Atm.png')

#%%
# simu.show(prop="hgsfdet")
# simu.show(prop="hgraph")
# simu.show(prop="cumflowvol")

#%%
# simu.show(prop="mbeconv")
mbeconv_file = os.path.join(simu.workdir,
                        simu.project_name,
                        'output/mbeconv'
                        )


mbeconv = out_CT.read_mbeconv(mbeconv_file)

parms = [
        "STORE1",
        "STORE2",
        'VOUT',
        'CUM.VOUT',
        ]

# STORE1 e STORE2: indicano la stessa cosa, ovvero una stima del
# volume d'acqua nel volume simulato. Sono diversi in quanto vengono
# calcolati in modo diverso: più sono simili, meglio è! Può essere utile
# plottare TIME vs STORE2, così vedo l'andamento nel tempo dello
# storage e capisco se serve aggiungere time step;

fig, axs = plt.subplots(2,2,
                        sharex=True)
axs = axs.ravel()

for i, pp in enumerate(parms): 
    
    mbeconv.set_index('TIME').plot(y=pp,
                                   ax=axs[i]
                                   )
    axs[i].set_title(pp)       
    axs[i].axvline(x=atmbc_df.time[1], 
                color='k', 
                linestyle='--', 
                label='end rain'
                )
    plt.show()
fig.savefig(figpath / 'mbeconv.png')

#%%

# simu.show(prop="mbeconv")
mbeconv_file = os.path.join(simu.workdir,
                        simu.project_name,
                        'output/mbeconv'
                        )


mbeconv = out_CT.read_mbeconv(mbeconv_file)

parms = [
        "DSTORE", #mostra di quanto è variato lo storage rispetto al passo precedente;
        "CUM.DSTORE", # CUM.DSTORE: tiene conto di tutte le variazioni cumulative dellostorage;
        'VIN', # Dovrebbe corrispondere alla pioggia · passo temporale · area
        'CUM. VIN',
        ]

fig, axs = plt.subplots(2,2,
                        sharex=True)
axs = axs.ravel()

for i, pp in enumerate(parms): 
    
    mbeconv.set_index('TIME').plot(y=pp,
                                   ax=axs[i]
                                   )
    axs[i].set_title(pp)       
    axs[i].axvline(x=atmbc_df.time[1], 
                color='k', 
                linestyle='--', 
                label='end rain'
                )
    plt.show()
fig.savefig(figpath / 'mbeconv.png')

#%%
import pandas as pd


def CATHY_2_daily_WaterBalance():
    # Convert the TIME column to datetime if it's not already
    # mbeconv['TIME'] = pd.to_datetime(mbeconv['TIME'], unit='s')  # Adjust the unit if necessary
    # Set TIME as the index
    # mbeconv.set_index('TIME', inplace=True)
    # Define your start date
    # start_date = pd.to_datetime('YYYY-MM-DD HH:MM:SS')  # Replace with your actual start date
    # Convert elapsed time to a timedelta and create a new datetime column
    # mbeconv['DATETIME'] = start_date + pd.to_timedelta(mbeconv['TIME'], unit='s')  # Adjust unit if necessary
    # Calculate cumulative sum of DELTAT
    mbeconv['CUM_DELTA'] = mbeconv['DELTAT'].cumsum()
    # Create a new column to represent the elapsed time in seconds
    mbeconv['ELAPSED_TIME'] = pd.to_timedelta(mbeconv['CUM_DELTA'], unit='s')  # Adjust unit if necessary
    # Set the new DATETIME column as the index
    mbeconv.set_index('ELAPSED_TIME', inplace=True)
    
    # Define aggregation functions for each column
    aggregation_functions = {
        # 'NSTEP': 'mean',         
        'DELTAT': 'sum',        
        'STORE1': 'sum',        
        'DSTORE': 'sum',         
        'CUM.DSTORE': 'last',         
        'VIN': 'sum',        
        # 'CUM. VIN': 'last',       
        'VOUT': 'sum',         
        'CUM.VOUT': 'last',
        'VIN+VOUT': 'sum',
        # 'REL. MBE': 'sum',        
        # 'CUM. MBE': 'last',       
        # 'CUM.': 'last',           
        # 'CUM_DELTA': 'last',         
    }
    
    # Resample to daily frequency with specified aggregation functions
    daily_mbeconv = mbeconv.resample('D').agg(aggregation_functions)
    
    return 
    




