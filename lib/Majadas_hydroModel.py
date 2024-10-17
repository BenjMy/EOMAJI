#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 13:46:29 2024

1a. Import dataset of Earth Observation analysis from TSEB (ds_analysis_EO) that contains:
    - DAILY Spatial Potential Evapotranspiration at 30m resolution 
    - DAILY Spatial Rain 
1b. Import DEM Majadas Catchment and build the mesh
    - Resample so DEM is same size that EO resolution 
    (
     this can be long as the catchement is big and with several outlets
     **activate**: Boundary channel constraction (No:0,Yes:1) =  1
     )
1c. Import Corinne Land Cover raster for Majadas
   - Use lookup table to convert Land Cover to:
       - vegetation map type
       - root depth
2. Parametrize Majadas hydrological model
    - Update atmospheric boundary conditions
    - Update boundary conditions
    - Update initial conditions
    - Uodate soil conditions
3. Run hydrological model

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
import rioxarray as rio

cwd = os.getcwd()
# prj_name = 'Majadas_2024_WTD1' #Majadas_daily Majadas_2024


#%%

def get_cmd():
    parse = argparse.ArgumentParser()
    process_param = parse.add_argument_group('process_param')
    process_param.add_argument('-prj_name', type=str, help='prj_name',
                        # default='Majadas_test', 
                        default='Majadas', 
                        required=False
                        ) 
    process_param.add_argument('-AOI', type=str, help='Area of Interest',
                        default='Buffer_5000', 
                        # default='H2_Bassin', 
                        required=False
                        ) 
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

file_pattern = '*TPday*.tif'
ET_0_filelist = list(prepoEOPath.glob(file_pattern))
crs_ET = rxr.open_rasterio(ET_0_filelist[0]).rio.crs
ET_test = rxr.open_rasterio(ET_0_filelist[0])

ds_analysis_EO = utils.read_prepo_EO_datasets(fieldsite='Majadas',
                                              AOI=args.AOI,
                                              crs=crs_ET
                                              )

if args.short==True:
    cutoffDate = ['01/01/2023','01/03/2024']
    start_time, end_time = pd.to_datetime(cutoffDate[0]), pd.to_datetime(cutoffDate[1])
    mask_time = (ds_analysis_EO.time >= start_time) & (ds_analysis_EO.time <= end_time)
    # Filter the DataArrays using the mask
    ds_analysis_EO = ds_analysis_EO.sel(time=mask_time)

majadas_aoi = Majadas_utils.get_Majadas_aoi(buffer=5000)

# fig, ax = plt.subplots()
# ds_analysis_EO.ETa.isel(time=1).plot.imshow(ax=ax)
# majadas_aoi.plot(ax=ax)
# ds_analysis_EO.RAIN.isel(time=0).plot.imshow()

#%% Look at ETp
# ----------------------------------------------------------------------------
# # Get a handle on the figure and the axes
# fig, ax = plt.subplots(figsize=(12,6))

# # Plot the initial frame. 
# cax = ds_analysis_EO['ETp'].isel(time=0).plot.imshow(
#     add_colorbar=True,
#     cmap='coolwarm',
#     vmin=0, vmax=6,
#     cbar_kwargs={
#         'extend':'neither'
#     }
# )
# # Next we need to create a function that updates the values for the colormesh, as well as the title.
# def animate(frame):
#     cax.set_array(ds_analysis_EO['ETp'].isel(time=frame).values)
#     ax.set_title("Time = " + str(ds_analysis_EO['ETp'].coords['time'].values[frame])[:13])

# # Finally, we use the animation module to create the animation.
# ani = FuncAnimation(
#     fig,             # figure
#     animate,         # name of the function above
#     frames=40,       # Could also be iterable or list
#     interval=200     # ms between frames
# )

# ani.save(filename=figPath/"ETp.gif", 
#          writer="pillow")


#%% Create CATHY mesh based on DEM
# ----------------------------------------------------------------------------

plt.close('all')
hydro_Majadas = CATHY(
                        dirName='../WB_FieldModels/',
                        prj_name=args.prj_name + args.AOI
                      )
hydro_MajadasPath = Path(hydro_Majadas.workdir) / hydro_Majadas.project_name

# DEM_notopo = np.ones([
#                     np.shape(ds_analysis_EO['RAIN'].isel(time=0))[0]-1,
#                     np.shape(ds_analysis_EO['RAIN'].isel(time=0))[1]-1,
#                     ]
#                     )
# maskDEM = (ds_analysis_EO['ETp'].isel(time=0).isnull()).values
# DEM_notopo[maskDEM]=-9999
# firstidDEM = np.where(DEM_notopo[0,:]!=-9999)[0]
# DEM_notopo[0,firstidDEM[0]]= 1-1e-3
# print('Issue between DEM dimension and xarray EO when DEM is null on the borders!')
# DEM_notopo[0,0]= 1-1e-3

#%% Create CATHY mesh based on DEM
# ----------------------------------------------------------------------------

DTM_rxr = rio.open_rasterio('../data/Spain/clipped_DTM_Majadas_AOI.tif')
# DTM_rxr = rio.open_rasterio('../data/Spain/clipped_DTM_H2_Majadas_AOI.tif')
DTM_rxr = DTM_rxr.where(DTM_rxr != DTM_rxr.attrs['_FillValue'], -9999)
DTM_rxr.attrs['_FillValue'] = -9999
fig, ax = plt.subplots(1)
img = ax.imshow(
                DTM_rxr.values[0],
                vmin=0
                )
plt.colorbar(img)
fig.savefig(figPath/'DEM_Catchement_Majadas.png', dpi=300
            )

nb_of_valid_cells = (len(DTM_rxr.x)*len(DTM_rxr.y))-509
nb_of_valid_nodes = nb_of_valid_cells + len(DTM_rxr.x) + len(DTM_rxr.y)
fill_value = DTM_rxr.attrs['_FillValue']
no_data_mask_DEM = DTM_rxr.isel(band=0) == fill_value  # This creates a boolean mask

# a
#%% Update prepro inputs and mesh
# no topo case 
# ----------------------------------------------------------------------------
# deltax = ds_analysis_EO['RAIN'].rio.resolution()[0]
# deltay = ds_analysis_EO['RAIN'].rio.resolution()[1]

# hydro_Majadas.update_prepo_inputs(
#                                 DEM=DEM_notopo,
#                                 delta_x = abs(deltax),
#                                 delta_y = abs(deltay),
#                                 # N=np.shape(dem_mat)[1],
#                                 # M=np.shape(dem_mat)[0],
#                                 # xllcorner=0,
#                                 # yllcorner=0,
#                                 base=6,
#                                 xllcorner=ds_analysis_EO['RAIN'].x.min().values,
#                                 yllcorner=ds_analysis_EO['RAIN'].y.min().values
#                                 )
# abs(DTM_rxr.rio.resolution()[0])
# DTM_rxrs

# resample 
# Define the new resolution (e.g., new pixel size)
# new_resolution = (300, 300)  # e.g., 30x30 meters

#%% Update prepro inputs and mesh
# with topo case 
# ----------------------------------------------------------------------------
from rasterio.enums import Resampling
reprojected_DEM = DTM_rxr.isel(band=0).rio.reproject_match(ds_analysis_EO['ETp'])

reprojected_DEM = DTM_rxr.isel(band=0).rio.reproject(
                                        DTM_rxr.rio.crs,
                                        shape=(len(ds_analysis_EO['ETp'].y),
                                               len(ds_analysis_EO['ETp'].x)),
                                        resampling=Resampling.bilinear,
                                    )

no_data_mask_DEM = reprojected_DEM == fill_value

# catchement topo case 
# ----------------------------------------------------------------------------
hydro_Majadas.update_prepo_inputs(
                                DEM=reprojected_DEM.to_numpy(),
                                delta_x = int(abs(reprojected_DEM.rio.resolution()[0])),
                                delta_y = int(abs(reprojected_DEM.rio.resolution()[1])),
                                # dr  = 1,
                                # delta_x = int(abs(DTM_rxr.rio.resolution()[1])),
                                # delta_y = int(abs(DTM_rxr.rio.resolution()[0])),
                                base=5,
                                xllcorner=reprojected_DEM.x.min().values,
                                yllcorner=reprojected_DEM.y.min().values,
                                # dr = int(abs(reprojected_DEM.rio.resolution()[1]))/2,
                                dr = 1,
                                # ivert=4
                                )

fig = plt.figure()
ax = plt.axes(projection="3d")
hydro_Majadas.show_input(prop="dem", ax=ax)
hydro_Majadas.create_mesh_vtk(verbose=True)
grid3d = hydro_Majadas.read_outputs('grid3d')

#%% Show mesh
# -----------------------------------------------------------------------------

pl = pv.Plotter(off_screen=True)
mesh = pv.read(f'{hydro_MajadasPath}/vtk/{hydro_Majadas.project_name}.vtk')
pl.add_mesh(mesh,show_edges=True)
pl.set_scale(zscale=3)
pl.show_grid()
pl.view_xz()
pl.show()
pl.screenshot(f'{figPath}/{hydro_Majadas.project_name}.png')

#%% PAD RAIN and ETp
ETp_meshnodes = Majadas_utils.xarraytoDEM_pad(ds_analysis_EO['ETp'])
Rain_meshnodes = Majadas_utils.xarraytoDEM_pad(ds_analysis_EO['RAIN'])

# ETp_meshnodes = ds_analysis_EO['ETp']
# Rain_meshnodes = ds_analysis_EO['RAIN']


# fig, axs = plt.subplots(2)
# ETp_meshnodes.isel(time=0).plot.imshow(ax=axs[0])
# ds_analysis_EO['ETp'].isel(time=0).plot.imshow(ax=axs[1])

#%% Update atmbc according to EO
# -----------------------------------------------------------------------------
ds_analysis_EO['Elapsed_Time_s'] = (ds_analysis_EO.time - ds_analysis_EO.time[0]).dt.total_seconds()

RAIN_3d = np.array([Rain_meshnodes.isel(time=t).values for t in range(ds_analysis_EO['RAIN'].sizes['time'])])
# RAIN_3d = np.array([ds_analysis_EO['RAIN'].isel(time=t).values for t in range(ds_analysis_EO['RAIN'].sizes['time'])])
ETp_3d = np.array([ETp_meshnodes.isel(time=t).values for t in range(ds_analysis_EO['ETp'].sizes['time'])])

# np.shape(RAIN_3d)

# utils.pad

v_atmbc = RAIN_3d*(1e-3/86400) - ETp_3d*(1e-3/86400)
v_atmbc_reshaped = v_atmbc.reshape(v_atmbc.shape[0], -1)
# v_atmbc = np.ravel(v_atmbc)

np.shape(np.zeros([len(list(ds_analysis_EO['Elapsed_Time_s'].values)),
                            int(grid3d['nnod'])]
                  )
)

int(grid3d['nnod'])
np.shape(v_atmbc_reshaped)
grid3d

non_nan_count = np.count_nonzero(~np.isnan(RAIN_3d[0].flatten()))
# non_nan_count = np.count_nonzero(~np.isnan(v_atmbc_reshaped[0]))

# hydro_Majadas.grid3d


test_vatmbc_reshaped = np.zeros([len(list(ds_analysis_EO['Elapsed_Time_s'].values)),
                                 int(grid3d['nnod'])]
                                )
np.shape(test_vatmbc_reshaped)

hydro_Majadas.update_atmbc(HSPATM=0,
                            IETO=1,
                            time=list(ds_analysis_EO['Elapsed_Time_s'].values),
                            # netValue=np.zeros([len(list(ds_analysis_EO['Elapsed_Time_s'].values)),
                            #                            int(grid3d['nnod'])]
                            #                  )
                            netValue=v_atmbc_reshaped
                            )

#%% Show atmbc time serie
# -----------------------------------------------------------------------------
hydro_Majadas.show_input('atmbc')

#%%
hydro_Majadas.update_nansfdirbc(no_flow=True)
hydro_Majadas.update_nansfneubc(no_flow=True)
hydro_Majadas.update_sfbc(no_flow=True)

#%% Update iniital conditions
# -----------------------------------------------------------------------------
# hydro_Majadas.update_ic(
#                         INDP=0, 
#                         IPOND=0, 
#                         pressure_head_ini=-10
#                     )

hydro_Majadas.update_ic(
                        INDP=4, 
                        WTPOSITION=args.WTD, 
                    )

#%% Update root depth according to CLC mapping
# -----------------------------------------------------------------------------

CLC_Majadas_clipped_grid = xr.open_dataset(f'../prepro/Majadas/{args.AOI}/CLCover_Majadas.netcdf',
                                            # engine='scipy'
                                           )

# CLC_Majadas_clipped_grid.Code_18_str

# ss

CLC_Majadas_clipped_grid.rio.resolution()
CLC_Majadas_clipped_grid = CLC_Majadas_clipped_grid.rio.write_crs(crs_ET)
crsCLC = CLC_Majadas_clipped_grid.rio.crs
# CLC_Majadas_clipped_grid_new = CLC_Majadas_clipped_grid.drop_vars('Code_18_str')

reprojected_CLC_Majadas = CLC_Majadas_clipped_grid.rio.reproject_match(ds_analysis_EO['ETp'])

CLC_codes = utils.get_CLC_code_def()

print('build lookup table for root depth')
code18_values_unique = np.unique(reprojected_CLC_Majadas.Code_18.values)
CLC_values_unique = np.unique(CLC_Majadas_clipped_grid.Code_CLC.values)

code18_str_rootmap_indice = [ (cci,i+1) for i, cci in enumerate(CLC_values_unique[:-1])]
replacement_dict = dict(code18_str_rootmap_indice)
replacement_dict[np.nan] = 0

mapped_data = np.zeros(np.shape(reprojected_CLC_Majadas.Code_CLC))
i = 1
for key, value in replacement_dict.items():
    mapped_data[reprojected_CLC_Majadas.Code_CLC.values == key] = i
    if np.sum(reprojected_CLC_Majadas.Code_CLC.values == key) > 1:
        i += 1 

# np.shape(CLC_Majadas_clipped_grid.Code_CLC.values)

# print('!dimension of CLCover is 24*21 but should be 23*20!')
# mapped_data = mapped_data - min_veg_data_shift + 1
# mapped_data = mapped_data[1:,0:-1] 
reprojected_CLC_Majadas.Code_CLC.where(reprojected_CLC_Majadas.Code_CLC==244.).plot.imshow()
# ss
#%%
# mapped_data = mapped_data[1:,0:-1] 
# min_veg_data_shift = np.min(mapped_data)
# np.shape(mapped_data[0:-1,0:-1])

# print(len(np.unique(mapped_data)))
hydro_Majadas.update_veg_map(mapped_data)
# print(hydro_Majadas.MAXVEG)

fig, ax = plt.subplots()
hydro_Majadas.show_input('root_map',ax=ax,
                         # cmap=''
                         )
fig.savefig(figPath/'root_map_Majadas.png', dpi=300
            )
CLC_root_depth = utils.CLC_2_rootdepth()

try:
    SPP, FP = hydro_Majadas.read_inputs('soil',MAXVEG=1) #,MAXVEG=15)
    FP_new = pd.concat([FP]*len(np.unique(mapped_data)+1), 
                       ignore_index=False
                       )
    FP_new.index = np.arange(1,len(np.unique(mapped_data))+1,1)
    FP_new.index.name = 'Veg. Indice'
except:
    SPP, FP = hydro_Majadas.read_inputs('soil',
                                        MAXVEG=len(np.unique(mapped_data))
                                        ) #,MAXVEG=15)
    FP_new = FP
    
    
for rd, value in replacement_dict.items():
    if str(rd) !='nan':
        code_str = CLC_codes[str(int(rd))]
        try:
            print(rd,code_str,CLC_root_depth[code_str])
            Rootdepth = CLC_root_depth[code_str]
        except:
            Rootdepth = 1e-3
    else:
        Rootdepth = 1e-3
    idveg = replacement_dict[rd]
    if np.sum((idveg==FP_new.index))==1:
        FP_new.loc[idveg,'ZROOT'] = Rootdepth
        print(idveg,Rootdepth)

# FP_new[1:]
# print(hydro_Majadas.MAXVEG)
#%%
hydro_Majadas.update_soil(
                            PMIN=-1e35,
                            FP_map=FP_new,
                            SPP=SPP,
                            show=True
                          )
# hydro_Majadas.cathyH

# from pyCATHY.plotters import cathy_plots as plt_CT
# update_map_veg = hydro_Majadas.map_prop_veg(FP_new)
# fig, ax = plt_CT.dem_plot_2d_top(update_map_veg,
#                                  label="all"
#                                  )
# fig, ax = plt.subplots(1)
# hydro_Majadas.show_input(prop="root_map", ax=ax,
#                          # linewidth=0
#                          )
# aa
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
