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
import geopandas as gpd
import rioxarray as rxr
from pathlib import Path
import matplotlib.pyplot as plt
import pyvista as pv
import pandas as pd
import utils
import Majadas_utils
import os
import argparse
from matplotlib.animation import FuncAnimation
from matplotlib import animation

cwd = os.getcwd()
# prj_name = 'Majadas_2024_WTD1' #Majadas_daily Majadas_2024


#%%

def get_cmd():
    parse = argparse.ArgumentParser()
    process_param = parse.add_argument_group('process_param')
    process_param.add_argument('-prj_name', type=str, help='prj_name',
                        default='Majadas_test', required=False) 
    process_param.add_argument('-het_soil', type=int, help='Heterogeneous soil',
                        default=1, required=False) 
    process_param.add_argument('-WTD', type=float, help='WT height',
                        # default=100, required=False) 
                        # default=4, required=False) 
                        default=2, required=False) 
    process_param.add_argument('-ic', type=int, help='Heterogeneous ic',
                        default=1, required=False) 
    process_param.add_argument('-short', type=int, help='select only one part of the dataset',
                        default=0, required=False) 
    process_param.add_argument('-SCF', type=float, help='Soil Cover Fraction',
                        default=0.5, required=False)
    process_param.add_argument('-OMEGA', type=float, 
                               help='Compensatory mechanisms for root water uptake',
                        default=1, required=False)
    args = parse.parse_args()
    return(args)    

args = get_cmd()   

figPath = Path(cwd) / '../figures/' / args.prj_name
figPath.mkdir(parents=True, exist_ok=True)
prepoEOPath = Path('/run/media/z0272571a/SENET/iberia_daily/E030N006T6')


#%% Import input data 

ds_analysis_EO = utils.read_prepo_EO_datasets(fieldsite='Majadas')

file_pattern = '*TPday*.tif'
ET_0_filelist = list(prepoEOPath.glob(file_pattern))
crs_ET = rxr.open_rasterio(ET_0_filelist[0]).rio.crs
ET_test = rxr.open_rasterio(ET_0_filelist[0])


if args.short==True:
    cutoffDate = ['01/01/2023','01/03/2024']
    start_time, end_time = pd.to_datetime(cutoffDate[0]), pd.to_datetime(cutoffDate[1])
    mask_time = (ds_analysis_EO.time >= start_time) & (ds_analysis_EO.time <= end_time)
    # Filter the DataArrays using the mask
    ds_analysis_EO = ds_analysis_EO.sel(time=mask_time)


#%%

# Get a handle on the figure and the axes
fig, ax = plt.subplots(figsize=(12,6))

# Plot the initial frame. 
cax = ds_analysis_EO['ETp'].isel(time=0).plot.imshow(
    add_colorbar=True,
    cmap='coolwarm',
    vmin=0, vmax=6,
    cbar_kwargs={
        'extend':'neither'
    }
)
# Next we need to create a function that updates the values for the colormesh, as well as the title.
def animate(frame):
    cax.set_array(ds_analysis_EO['ETp'].isel(time=frame).values)
    ax.set_title("Time = " + str(ds_analysis_EO['ETp'].coords['time'].values[frame])[:13])

# Finally, we use the animation module to create the animation.
ani = FuncAnimation(
    fig,             # figure
    animate,         # name of the function above
    frames=40,       # Could also be iterable or list
    interval=200     # ms between frames
)

ani.save(filename=figPath/"ETp.gif", 
         writer="pillow")


#%% Create CATHY mesh based on DEM

plt.close('all')
hydro_Majadas = CATHY(
                        dirName='../WB_FieldModels/',
                        prj_name=args.prj_name
                      )

DEM_notopo = np.ones([
                    np.shape(ds_analysis_EO['RAIN'].isel(time=0))[0]-1,
                    np.shape(ds_analysis_EO['RAIN'].isel(time=0))[1]-1,
                    ]
                    )
maskDEM = (ds_analysis_EO['ETp'].isel(time=0).isnull()).values

# DEM_notopo[maskDEM]=-9999
# firstidDEM = np.where(DEM_notopo[0,:]!=-9999)[0]
# DEM_notopo[0,firstidDEM[0]]= 1-1e-3
print('Issue between DEM dimension and xarray EO when DEM is null on the borders!')
DEM_notopo[0,0]= 1-1e-3

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

# sss
deltax = ds_analysis_EO['RAIN'].rio.resolution()[0]
deltay = ds_analysis_EO['RAIN'].rio.resolution()[1]
hydro_Majadas.update_prepo_inputs(
                                DEM=DEM_notopo,
                                delta_x = abs(deltax),
                                delta_y = abs(deltay),
                                # N=np.shape(dem_mat)[1],
                                # M=np.shape(dem_mat)[0],
                                # xllcorner=0,
                                # yllcorner=0,
                                base=6,
                                xllcorner=ds_analysis_EO['RAIN'].x.min().values,
                                yllcorner=ds_analysis_EO['RAIN'].y.min().values
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
hydro_Majadas.grid3d['mesh3d_nodes'][:,2]

#%% Update atmbc

RAIN_3d = np.array([ds_analysis_EO['RAIN'].isel(time=t).values for t in range(ds_analysis_EO['RAIN'].sizes['time'])])
ETp_3d = np.array([ds_analysis_EO['ETp'].isel(time=t).values for t in range(ds_analysis_EO['ETp'].sizes['time'])])

v_atmbc = RAIN_3d*(1e-3/86400) - ETp_3d*(1e-3/86400)
np.shape(v_atmbc)
np.shape(ETp_3d)
np.shape(RAIN_3d)

v_atmbc_reshaped = v_atmbc.reshape(v_atmbc.shape[0], -1)
np.shape(v_atmbc_reshaped)

ds_analysis_EO['Elapsed_Time_s'] = (ds_analysis_EO.time - ds_analysis_EO.time[0]).dt.total_seconds()

    
hydro_Majadas.update_atmbc(HSPATM=0,
                            IETO=1,
                            time=list(ds_analysis_EO['Elapsed_Time_s'].values),
                            # netValue=np.zeros(np.shape(v_atmbc_reshaped))
                            netValue=v_atmbc_reshaped
                            )
np.shape(v_atmbc_reshaped)
np.shape(list(ds_analysis_EO['Elapsed_Time_s'].values))
np.shape(hydro_Majadas.DEM)

hydro_Majadas.show_input('atmbc')

#%%
hydro_Majadas.update_nansfdirbc(no_flow=True)
hydro_Majadas.update_nansfneubc(no_flow=True)
hydro_Majadas.update_sfbc(no_flow=True)

#%%
# hydro_Majadas.update_ic(
#                         INDP=0, 
#                         IPOND=0, 
#                         pressure_head_ini=-10
#                     )

hydro_Majadas.update_ic(
                        INDP=4, 
                        WTPOSITION=args.WTD, 
                    )


#%%
# hydro_Majadas.update_soil()

    

#%% Update root depth
# ss
CLC_Majadas_clipped_grid = xr.open_dataset('../prepro/CLCover_Majadas.netcdf')
CLC_Majadas_clipped_grid.rio.resolution()
CLC_Majadas_clipped_grid = CLC_Majadas_clipped_grid.rio.write_crs(crs_ET)
CLC_Majadas_clipped_grid.rio.crs
# new_resolution = (300, 300)  # 300x300 meter resolution
# CLC_Majadas_resampled = CLC_Majadas_clipped_grid.rio.reproject(
#                                                                 CLC_Majadas_clipped_grid.rio.crs,  # Keep the current CRS
#                                                                 resolution=new_resolution           # Set the new resolution
#                                                             )

CLC_codes = utils.get_CLC_code_def()

print('build lookup table for root depth')
CLC_Majadas_clipped_grid
CLC_Majadas_clipped_grid.Code_18_str.values
code18_values_unique = np.unique(CLC_Majadas_clipped_grid.Code_18.values)
code18_str_values_unique = np.unique(CLC_Majadas_clipped_grid.Code_18_str.values)

# code18_rootmap_indice = [ (cci,i) for i, cci in enumerate(code18_values_unique)]
# code18_rootmap_indice.append((-9999,0))
# replacement_dict = dict(code18_rootmap_indice)

code18_str_rootmap_indice = [ (cci,i+2) for i, cci in enumerate(code18_str_values_unique)]
replacement_dict = dict(code18_str_rootmap_indice)
replacement_dict['nodata'] = 1


mapped_data = np.copy(CLC_Majadas_clipped_grid.Code_18.values)
for key, value in replacement_dict.items():
    mapped_data[CLC_Majadas_clipped_grid.Code_18_str.values == key] = value

print('!dimension of CLCover is 24*21 but should be 23*20!')
mapped_data = mapped_data[1:,0:-1] 
min_veg_data_shift = np.min(mapped_data)
mapped_data = mapped_data - min_veg_data_shift + 1

#%%
# np.shape(mapped_data[0:-1,0:-1])
hydro_Majadas.update_veg_map(mapped_data)

fig, ax = plt.subplots()
hydro_Majadas.show_input('root_map',ax=ax,
                         # cmap=''
                         )
fig.savefig(figPath/'root_map_Majadas.png', dpi=300
            )
CLC_root_depth = utils.CLC_2_rootdepth()

try:
    SPP, FP = hydro_Majadas.read_inputs('soil',MAXVEG=1) #,MAXVEG=15)
    FP_new = pd.concat([FP]*(len(np.unique(mapped_data))), 
                       ignore_index=False
                       )
    FP_new.index = np.arange(1,len(np.unique(mapped_data))+1,1)
    FP_new.index.name = 'Veg. Indice'
except:
    SPP, FP = hydro_Majadas.read_inputs('soil',MAXVEG=len(np.unique(mapped_data))) #,MAXVEG=15)
    FP_new = FP

for rd in replacement_dict.keys():
    if rd != 'nodata':
        code_str = CLC_codes[rd]
        Rootdepth = CLC_root_depth[code_str]
    else:
        Rootdepth = 0
    idveg = replacement_dict[rd] - min_veg_data_shift +1
    if np.sum((idveg==FP_new.index))==1:
        FP_new.loc[idveg,'ZROOT'] = Rootdepth

hydro_Majadas.update_soil(
                            PMIN=-1e35,
                            FP_map=FP_new,
                            SPP=SPP,
                            show=True
                          )
# hydro_Majadas.cathyH

#%% Run simulation
len(ds_analysis_EO['Elapsed_Time_s'].values)

# resample_times_vtk = ds_analysis_EO['Elapsed_Time_s'].values[np.arange(0,len(ds_analysis_EO['Elapsed_Time_s'].values),30)]
# len(resample_times_vtk)

hydro_Majadas.update_parm(
                        TIMPRTi=ds_analysis_EO['Elapsed_Time_s'].values,
                        # TIMPRTi=resample_times_vtk,
                        IPRT=4,
                        VTKF=0,
                        )

#%%
hydro_Majadas.run_processor(
                      IPRT1=2,
                      TRAFLAG=0,
                      DTMIN=1e-3,
                      DTMAX=1e3,
                      DELTAT=1e2,
                      verbose=True
                      )
#%%
