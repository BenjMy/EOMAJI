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

cwd = os.getcwd()
figPath = Path(cwd) / '../figures/Majadas'
prepoEOPath = Path('/run/media/z0272571a/SENET/iberia_daily/E030N006T6')


#%%

def get_cmd():
    parse = argparse.ArgumentParser()
    process_param = parse.add_argument_group('process_param')
    process_param.add_argument('-het_soil', type=int, help='Heterogeneous soil',
                        default=1, required=False) 
    process_param.add_argument('-WTD', type=float, help='WT height',
                        # default=100, required=False) 
                        # default=4, required=False) 
                        default=99, required=False) 
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

#%% Import input data 

ds_analysis_EO = utils.read_prepo_EO_datasets(fieldsite='Majadas')

file_pattern = '*TPday*.tif'
ET_0_filelist = list(prepoEOPath.glob(file_pattern))
crs_ET = rxr.open_rasterio(ET_0_filelist[0]).rio.crs
ET_test = rxr.open_rasterio(ET_0_filelist[0])


if args.short==True:
    cutoffDate = ['01/01/2023','01/01/2024']
    start_time, end_time = pd.to_datetime(cutoffDate[0]), pd.to_datetime(cutoffDate[1])
    mask_time = (ds_analysis_EO.time >= start_time) & (ds_analysis_EO.time <= end_time)
    # Filter the DataArrays using the mask
    ds_analysis_EO = ds_analysis_EO.sel(time=mask_time)


#%%
from matplotlib.animation import FuncAnimation
from matplotlib import animation



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
                        prj_name="Majadas"
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
hydro_Majadas.update_ic(
                        INDP=0, 
                        IPOND=0, 
                        pressure_head_ini=-10
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

# FP_new['CLC']=None
# FP_new['Code_18']=None
print('This is wrongggg!')
for rd in replacement_dict.keys():
    # if replacement_dict[rd]<=len(np.unique(mapped_data)):
    # print(rd)
    if rd != 'nodata':
        code_str = CLC_codes[rd]
        # print(code_str)
        Rootdepth = CLC_root_depth[code_str]
    else:
        Rootdepth = 0
    idveg = replacement_dict[rd] - min_veg_data_shift +1
#     print(idveg,code_str)
    if np.sum((idveg==FP_new.index))==1:
        FP_new.loc[idveg,'ZROOT'] = Rootdepth
        # FP_new.loc[idveg,'CLC'] = code_str
        # FP_new.loc[idveg,'Code_18'] = rd

hydro_Majadas.update_soil(
                            PMIN=-1e35,
                            FP_map=FP_new,
                            SPP=SPP,
                            show=True
                          )
# hydro_Majadas.cathyH

#%% Run simulation
len(ds_analysis_EO['Elapsed_Time_s'].values)

resample_times_vtk = ds_analysis_EO['Elapsed_Time_s'].values[np.arange(0,len(ds_analysis_EO['Elapsed_Time_s'].values),30)]
len(resample_times_vtk)

hydro_Majadas.update_parm(
                        TIMPRTi=resample_times_vtk,
                        IPRT=4,
                        VTKF=2,
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
majadas_aoi = gpd.read_file('../data/AOI/majadas_aoi.geojson')
# majadas_aoi.to_crs(RAIN.rio.crs, inplace=True)
majadas_POIs, POIs_coords = Majadas_utils.get_Majadas_POIs()


#%% Read TDR

coord_SWC_CT, gdf_SWC_CT = Majadas_utils.get_SWC_pos(
                                                    target_crs=crs_ET
                                                    )


TDR_SWC, depths = Majadas_utils.get_SWC_data()
TDR_SWC.columns
# s
# Majadas_utils.get_SWC_data()

#%%
SMC_XY = [list(pi) for pi in POIs_coords]
SMC_XY.append(list(gdf_SWC_CT.iloc[0].geometry.coords[0]))
# SMC_XY = np.hstack(SMC_XY)

SMC_depths = [-di/100+1 for di in depths] # add altitude of DEM =1 (flat no top case)


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
from pyCATHY.importers import cathy_outputs as out_CT

df_fort777 = out_CT.read_fort777(os.path.join(hydro_Majadas.workdir,
                                              hydro_Majadas.project_name,
                                              'fort.777'),
                                 )


#%%
fig, ax = plt.subplots(1)
hydro_Majadas.show('spatialET',
                   ax=ax, 
                   ti=10,
                    clim=[0,5e-9],
                   )
ax.scatter(np.array(SMC_XY)[:,0],
           np.array(SMC_XY)[:,1],
           color='r'
           )

fig.savefig(figPath/'spatialET_Majadas.png',
            dpi=300
            )

#%%

# Get a handle on the figure and the axes
fig, ax = plt.subplots(figsize=(12,6))
# Plot the initial frame. 
cax = hydro_Majadas.show('spatialET',
                         ax=ax, 
                         ti=0,
                         clim=[0,5e-9],

                   )
ti = df_fort777['time'].unique()[1]
df_fort777_select_t_xr = df_fort777.set_index(['time','X','Y']).to_xarray()
df_fort777_select_t_xr = df_fort777_select_t_xr.rio.set_spatial_dims('X','Y')

# Next we need to create a function that updates the values for the colormesh, as well as the title.
def animate(frame):
    vi = df_fort777_select_t_xr.isel(time=frame)['ACT. ETRA'].values
    cax.set_array(vi)
    ax.set_title("Time = " + str(df_fort777_select_t_xr.coords['time'].values[frame])[:13])

# Finally, we use the animation module to create the animation.
ani = FuncAnimation(
    fig,             # figure
    animate,         # name of the function above
    frames=100,       # Could also be iterable or list
    interval=50     # ms between frames
)
# To save the animation using Pillow as a gif
writer = animation.PillowWriter(fps=15,
                                metadata=dict(artist='Me'),
                                bitrate=1800)
ani.save(figPath/'spatialET_Majadas.gif', writer=writer)

#%% Compare ETa from TSEB with ETa from pyCATHY

ETa_poi = df_fort777_select_t_xr.sel(X=all_closestPos[0],
                                     Y=all_closestPos[1], 
                                     method="nearest"
                                    )

ETa_poi_datetimes = filtered_RAIN.time.isel(time=0).values + ETa_poi.time.values

fig, ax = plt.subplots()
for pi in range(1):
    ax.plot(ETa_poi_datetimes, 
            ETa_poi['ACT. ETRA'].isel(X=pi,Y=pi).values*1000*86400,
            linestyle='--',
            label='CATHY'
            )

ETa_TSEB = xr.open_dataset('../prepro/ETa_Majadas.netcdf')
ETa_TSEB = ETa_TSEB.rename({"__xarray_dataarray_variable__": "ETa_TSEB"})
ETa_TSEB = ETa_TSEB.to_dataarray().isel(variable=0,band=0).sortby('time')

ETa_TSEB_poi = ETa_TSEB.sel(x=all_closestPos[0],
                            y=all_closestPos[1], 
                            method="nearest"
                            )
for pi in range(1):
    ax.plot(ETa_TSEB.time,
            ETa_TSEB.isel(x=pi,y=pi).values,
            label='TSEB'
            )
ax.set_xlabel('Date')
ax.set_ylabel('ETa (mm/day)')
plt.legend()
fig.savefig(figPath/'ETa_hydro_ETa_Energy_Majadas.png',
            dpi=300
            )

#%%
POROSITY_GUESS = 0.75
df_sw, _ = hydro_Majadas.read_outputs('sw')

from pyCATHY import cathy_utils
dates = cathy_utils.change_x2date(df_sw.index.values, 
                                  filtered_RAIN.time.isel(time=0).values
                                  )

# Create a figure and the first axis
# fig, ax = plt.subplots()

# Define the mosaic layout
mosaic_layout = """
                a
                b
                b
                b
                """
fig, ax = plt.subplot_mosaic(mosaic_layout,
                             sharex=True,
                             figsize=(8,4)
                             )

# Plot the first dataset on the primary y-axis
ax['b'].plot(dates, df_sw[all_nodes_SMC[0]], 
             'b-',
             label=str(all_closestPos[0][2])
             )  
ax['b'].plot(dates, df_sw[all_nodes_SMC[1]], 
             'b--',
             label=str(all_closestPos[1][2])
             )  
ax['b'].legend()
ax['b'].set_xlabel('time (s)')
ax['b'].set_ylabel('saturation (-)', color='b')
ax['b'].tick_params(axis='y', labelcolor='b')

ETp_poi = filtered_ETp.sel(x=all_closestPos[0],
                           y=all_closestPos[1],
                           method="nearest")

rain_poi = filtered_RAIN.sel(x=all_closestPos[0],
                             y=all_closestPos[1], 
                             method="nearest")

# create here second axis and plot on it 

hydro_Majadas.atmbc['atmbc_df']


for pi in range(1):
    ax['a'].plot(rain_poi.time,
            rain_poi.isel(x=pi,y=pi).values,
            )
ax['a'].invert_yaxis()
ax['a'].set_ylabel('rain (mm)', color='b')

plt.tight_layout()
plt.savefig(figPath/'saturation_simu_Majadas.png',
            dpi=300
            )