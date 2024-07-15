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
)

simu.update_parm(TRAFLAG=0)
# simu.create_mesh_vtk(verbose=True)
simu.run_preprocessor(verbose=True)
# simu.update_zone(irr_zones)
# simu.update_ic(pressure_head_ini=-300)
# simu.update_ic(INDP=3, IPOND=0, 
#                WTPOSITION=1
#                )

simu.update_ic(pressure_head_ini=-25)

simu.update_soil(PMIN=-1e35)

# The values are those of a 200-min rainfall event at a uniform
# intensity of 3.3·10-4 m/min, followed by 100 min of drainage.
 
net_rain = 1e-7 # L/s or m/s??
net_etp = -1e-7

simu.update_atmbc(
                  HSPATM=1,
                  IETO=1,
                  time=[0,3600,3600*30],
                  # netValue=[net_rain,net_etp,0.0],
                  netValue=[net_rain,net_etp,0.0],
                  # show=True,
                )
figpath = Path(simu.workdir) / simu.project_name
simu.run_processor(verbose=True,
                   IPRT1=2
                   )

#%%
mesh3d = simu.read_outputs('grid3d')
nnod = mesh3d['nnod']
nnod3 = mesh3d['nnod3']

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


parms = [
        "RET. FLUX",
        'ACT. FLUX',
        'OVL. FLUX',
        'POT. FLUX',
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

    ax.set_ylabel('Flux (m/s?)')
    
fig.savefig(figpath / 'fluxes.png')

#%%

import numpy as np
import xarray as xr

# Sample data: shape (3, 16) where 3 is time and 16 is x (spatial dimension)
data = np.random.rand(3, 16)

# Sample coordinates: shape (2, 16), assuming 2 for x and y coordinates for 16 points
xy = np.random.rand(2, 16)

# Create coordinate arrays
x_coords = xy[0, :]
y_coords = xy[1, :]

# Create time coordinates
time_coords = np.arange(data.shape[0])

# Create the DataArray with coordinates
data_array = xr.DataArray(
    data,
    coords={
        'time': time_coords,
        'x': x_coords,
        'y': ('x', y_coords)  # Assigning y_coords to the x dimension
    },
    dims=['time', 'x']
)

print(data_array)


# Plot using xarray's plot.imshow
fig, axes = plt.subplots(1, data_array.sizes['time'], figsize=(15, 5))

for t in range(data_array.sizes['time']):
    ax = axes[t] if data_array.sizes['time'] > 1 else axes
    data_array.isel(time=t).plot.imshow(ax=ax, aspect='auto')
    ax.set_title(f'Time {t}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

plt.tight_layout()
plt.show()


#%%

out_with_IRR = utils.read_outputs(simu)

out_with_IRR['psi']

psi_surf = out_with_IRR['psi'].iloc[:,0:int(nnod)]
np.shape(psi_surf.to_numpy())

import xarray as xr
data = xr.DataArray(psi_surf,
                    # coords={"Time":out_with_IRR['psi'].index
                            # }
                    )
# data.assign_coords({'x':coords[:,0]})
# data.dims


# .to_numpy()
# np.shape(psi_surf)
# mesh3d = simu.read_outputs('grid3d')
# nnod = mesh3d['nnod']
# coords = mesh3d['mesh3d_nodes'][0:int(nnod)]
# len(coords)

# import xarray as xr
# data = xr.DataArray(psi_surf, 
#                     # dims=("time","x"), 
#                     coords={"x": coords[:,0],
#                             # "y": coords[:,1],
#                             # "time": out_with_IRR['psi'].index
#                             }
#                     )


#%% Plot ET outputs
# -------------------------
# sss
fort777 = os.path.join(simu.workdir,
                        simu.project_name,
                        'fort.777'
                        )
ETa = out_CT.read_fort777(fort777)
ETa_xarray = ETa.set_index(['time', 'X', 'Y']).to_xarray()

# fig, ax = plt.subplots()
ETa_xarray['ACT. ETRA'].plot.imshow(x="X", y="Y", 
                                    col="time", 
                                    col_wrap=4,
                                    )
plt.savefig(figpath / 'spatial_ETa.png',)

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


# dtcoupling_irr = dtcoupling[dtcoupling.Time <= 12e3]
dtcoupling['Atmpot-v_cumsum'] = dtcoupling['Atmpot-v'].cumsum()
dtcoupling['Atmpot-r_cumsum'] = dtcoupling['Atmpot-r'].cumsum()
dtcoupling['Atmact-v_cumsum'] = dtcoupling['Atmact-v'].cumsum()
dtcoupling['Atmact-r_cumsum'] = dtcoupling['Atmact-r'].cumsum()
dtcoupling['Atmact-vf_cumsum'] = dtcoupling['Atmact-vf'].cumsum()
# Atmact-vf

# fig, axs = plt.subplots(2,1,sharex=True)

# dtcoupling.set_index('Time').plot(y='Atmact-v',ax=axs[0])
# dtcoupling.set_index('Time').plot(y='Atmact-vf',ax=axs[1])

# fig, axs = plt.subplots(2,1,sharex=True)
# dtcoupling.set_index('Time').plot(y='Atmact-v_cumsum',ax=axs[0])
# dtcoupling.set_index('Time').plot(y='Atmact-r_cumsum',ax=axs[1])



# fig, axs = plt.subplots(2,1,sharex=True)
# dtcoupling.set_index('Time').plot(y='Atmpot-v',ax=axs[0])
# dtcoupling.set_index('Time').plot(y='Atmpot-r',ax=axs[1])


max_atmpot_v = dtcoupling['Atmpot-v'].max()
max_atmpot_v_cumsum = dtcoupling['Atmpot-v_cumsum'].max()
max_atmpot_r = dtcoupling['Atmpot-r'].max()
max_atmpot_r_cumsum = dtcoupling['Atmpot-r_cumsum'].max()

print(f'Cum. Sum exfiltration {max_atmpot_v_cumsum} [L]')
print(f'exfiltration {max_atmpot_v} [L]')
print(f'Cum. Sum  exfiltration {max_atmpot_r_cumsum} [L/T]')
print(f'exfiltration {max_atmpot_r} [L] [L/T]')


#%%

parms = [
        "Atmpot-r_cumsum",
        "Atmact-r_cumsum",
        "Atmpot-v_cumsum",
        "Atmact-v_cumsum",
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



