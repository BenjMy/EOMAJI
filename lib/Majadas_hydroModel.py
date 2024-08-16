#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 13:46:29 2024
"""

import xarray as xr
import numpy as np

import pyCATHY 
from pyCATHY import CATHY
from pyCATHY.importers import cathy_inputs as in_CT
from pyCATHY.plotters import cathy_plots as cplt

from pathlib import Path
import matplotlib.pyplot as plt
import pyvista as pv
import pandas as pd

import Majadas_utils

#%% Import input data 

ETp_ds = xr.open_dataset('../prepro/ETp_Majadas.netcdf')
ETp_ds = ETp_ds.rename({"__xarray_dataarray_variable__": "ETp"})
RAIN_ds = xr.open_dataset('../prepro/RAIN_Majadas.netcdf')
RAIN_ds = RAIN_ds.rename({"__xarray_dataarray_variable__": "RAIN"})


ETp = ETp_ds.to_dataarray().isel(variable=0,band=0).sortby('time')
RAIN = RAIN_ds.to_dataarray().isel(variable=0,band=0).sortby('time')

# Issue with rain!

print('Errrrrorrr in rain evaluation in the input!')
# data_array = data_array.where((data_array <= 300) & (data_array > 0), other=np.nan)
RAIN = RAIN.where((RAIN <= 300) & (RAIN > 0), other=0)


# Determine the overlapping time range
start_time = max(RAIN.time.min(), ETp.time.min())
end_time = min(RAIN.time.max(), ETp.time.max())

# Create a mask for the common time range
mask_time = (RAIN.time >= start_time) & (RAIN.time <= end_time)
mask_time2 = (ETp.time >= start_time) & (ETp.time <= end_time)

# Filter the DataArrays using the mask
filtered_RAIN = RAIN.sel(time=mask_time)
filtered_ETp = ETp.sel(time=mask_time2)

cutoffDate = ['01/01/2023','01/01/2024']
start_time, end_time = pd.to_datetime(cutoffDate[0]), pd.to_datetime(cutoffDate[1])
mask_time = (filtered_RAIN.time >= start_time) & (filtered_RAIN.time <= end_time)

# Filter the DataArrays using the mask
filtered_RAIN = filtered_RAIN.sel(time=mask_time)
filtered_ETp = filtered_ETp.sel(time=mask_time)
np.shape(filtered_ETp)

#%% Create CATHY mesh based on DEM


plt.close('all')
hydro_Majadas = CATHY(
                        dirName='../WB_FieldModels/',
                        prj_name="Majadas"
                      )

DEM_notopo = np.ones([
                    np.shape(RAIN.isel(time=0))[0]-1,
                    np.shape(RAIN.isel(time=0))[1]-1
                    ]
                    )
DEM_notopo[-1,-1]= 1-1e-3

#%% Create CATHY mesh based on DEM
# # dem_mat, str_hd_dem = in_CT.read_dem(DTM_rxr.values)
# fig, ax = plt.subplots(1)
# img = ax.imshow(
#                 DTM_rxr.values[0],
#                 vmin=0
#                 )
# plt.colorbar(img)
# hydro_Majadas.show_input(prop="dem")

#%% Update prepro inputs and mesh
# hydro_Majadas.update_prepo_inputs(
#                                 DEM=DEM_notopo,
#                                 delta_x = 1,
#                                 delta_y = 1,
#                                 # N=np.shape(dem_mat)[1],
#                                 # M=np.shape(dem_mat)[0],
#                                 xllcorner=0,
#                                 yllcorner=0
#                                 # xllcorner=RAIN.x.min().values,
#                                 # yllcorner=RAIN.y.min().values
#                                 )



hydro_Majadas.update_prepo_inputs(
                                DEM=DEM_notopo,
                                delta_x = 300,
                                delta_y = 300,
                                # N=np.shape(dem_mat)[1],
                                # M=np.shape(dem_mat)[0],
                                # xllcorner=0,
                                # yllcorner=0
                                xllcorner=RAIN.x.min().values,
                                yllcorner=RAIN.y.min().values
                                )


# dem_mat, str_hd_dem = in_CT.read_dem(DTM_rxr.values)

# fig, ax = plt.subplots(1)
# img = ax.imshow(
#                 DTM_rxr.values[0],
#                 vmin=0
#                 )
# plt.colorbar(img)


hydro_Majadas.show_input(prop="dem")

# hydro_Majadas.update_prepo_inputs(
#                                 DEM=DTM_rxr.values[0],
#                                 # N=np.shape(dem_mat)[1],
#                                 # M=np.shape(dem_mat)[0],
#                             )

fig = plt.figure()
ax = plt.axes(projection="3d")
hydro_Majadas.show_input(prop="dem", ax=ax)

hydro_Majadas.run_preprocessor(verbose=True)

hydro_Majadas.create_mesh_vtk(verbose=True)

#%% Update atmbc



RAIN_3d = np.array([filtered_RAIN.isel(time=t).values for t in range(filtered_RAIN.sizes['time'])])
ETp_3d = np.array([filtered_ETp.isel(time=t).values for t in range(filtered_ETp.sizes['time'])])

v_atmbc = RAIN_3d*(1e-3/86400) - ETp_3d*(1e-3/86400)
np.shape(v_atmbc)
np.shape(RAIN_3d)

v_atmbc_reshaped = v_atmbc.reshape(v_atmbc.shape[0], -1)
np.shape(v_atmbc_reshaped)

filtered_RAIN['Elapsed_Time_s'] = (filtered_RAIN.time - filtered_RAIN.time[0]).dt.total_seconds()

    
hydro_Majadas.update_atmbc(HSPATM=0,
                            IETO=1,
                            time=filtered_RAIN['Elapsed_Time_s'].values,
                            # netValue=np.zeros(np.shape(v_atmbc_reshaped))
                            netValue=v_atmbc_reshaped
                            )

hydro_Majadas.show_input('atmbc')

#%%
hydro_Majadas.update_nansfdirbc(no_flow=True)
hydro_Majadas.update_nansfneubc(no_flow=True)
hydro_Majadas.update_sfbc(no_flow=True)

#%%
hydro_Majadas.update_ic(
                        INDP=0, 
                        IPOND=0, 
                        pressure_head_ini=-10
                    )


#%% Update root depth

root_map = np.ones(np.shape(DEM_notopo))
hydro_Majadas.update_veg_map(root_map)


SPP, FP = hydro_Majadas.read_inputs('soil')
FP['ZROOT']=1 # almost no roots/ bare soil swale

hydro_Majadas.update_soil(
                        PMIN=-1e35,
                        FP_map=FP,
                        SPP=SPP,
                    )

#%% Run simulation
hydro_Majadas.update_parm(
                        TIMPRTi=filtered_RAIN['Elapsed_Time_s'].values,
                        IPRT=4,
                        VTKF=2,
                        )
hydro_Majadas.run_processor(
                      IPRT1=2,
                      TRAFLAG=0,
                      DTMIN=1e-3,
                      DTMAX=1e3,
                      DELTAT=100,
                      verbose=True
                      )
#%%

cplt.show_vtk_TL(
                unit="saturation",
                notebook=False,
                path= str(Path(hydro_Majadas.workdir) / hydro_Majadas.project_name / "vtk"),
                show=False,
                x_units='days',
                # clim = [0.55,0.70],
                savefig=True,
            )

#%%
# import Majadas_utils
# from Majadas_utils import get_Majadas_POIs
import geopandas as gpd
majadas_aoi = gpd.read_file('../data/AOI/majadas_aoi.geojson')
# majadas_aoi.to_crs(RAIN.rio.crs, inplace=True)
# majadas_POIs, POIs_coords = Majadas_utils.get_Majadas_POIs()

RAIN.x.min()

# SMC_XY = POIs_coords #[[10,10]]
SMC_XY = [[RAIN.x.min().values+1000,RAIN.y.min().values+1000],
          [RAIN.x.min().values+1000,RAIN.y.min().values+1000]
          ]
SMC_depths = [-1,1]

hydro_Majadas.grid3d['mesh3d_nodes'][:,2]

#%% Plot SMC sensors on top of vtk mesh

# Find the altitudes of the nodes at the mesh positions
all_nodes_SMC = []
all_closestPos = []

for idx, (x, y) in enumerate(SMC_XY):
    _, closest = hydro_Majadas.find_nearest_node([x, y, 0])
    for d in [SMC_depths[idx]]:
        SMC_XYZi = [x, y, closest[0][2] - d]
        nodeId, closest_depth = hydro_Majadas.find_nearest_node(SMC_XYZi)
        all_nodes_SMC.append(nodeId)
        all_closestPos.append(closest_depth)

# Convert lists to numpy arrays for easier manipulation
all_nodes_SMC = np.vstack(all_nodes_SMC)
all_closestPos = np.vstack(all_closestPos)

# Plot the mesh and the sensor points
pl = pv.Plotter(notebook=True)
mesh = pv.read(Path(hydro_Majadas.workdir) / hydro_Majadas.project_name / f'vtk/{hydro_Majadas.project_name}.vtk')

pl.add_mesh(mesh, opacity=0.1)

for i, cp in enumerate(all_closestPos):
    pl.add_points(cp, color='red',
                  # markersize=10
                  )
pl.show_grid()
pl.show()

#%%
POROSITY_GUESS = 0.75
df_sw, _ = hydro_Majadas.read_outputs('sw')

from pyCATHY import cathy_utils
dates = cathy_utils.change_x2date(df_sw.index.values, 
                                  filtered_RAIN.time.isel(time=0).values
                                  )


# Create a figure and the first axis
fig, ax1 = plt.subplots()

# Plot the first dataset on the primary y-axis
ax1.plot(dates, df_sw[all_nodes_SMC[0]], 'b-')  # Plot in blue
ax1.plot(dates, df_sw[all_nodes_SMC[1]], 'b--')  # Plot in blue
ax1.set_xlabel('time (s)')
ax1.set_ylabel('saturation (-)', color='b')
ax1.tick_params(axis='y', labelcolor='b')

ETp_poi = filtered_ETp.sel(x=all_nodes_SMC[0],y=all_nodes_SMC[1], method="nearest")
rain_poi = filtered_RAIN.sel(x=all_nodes_SMC[0],y=all_nodes_SMC[1], method="nearest")
# rain_poi.plot(ax=ax1)

plt.tight_layout()
# plt.savefig(str(fig_path) + prj_name + '_Saturation_simu_VS_Real.png',dpi=300)