#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 13:46:29 2024

1a. Import dataset of Earth Observation analysis from TSEB (ds_analysis_EO) that contains:
    - DAILY Spatial Potential Evapotranspiration at 30m resolution 
    - DAILY Spatial Rain 
1b. Import DEM Burkina Faso Catchment and build the mesh
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
# import utils
import BurkinaFaso_utils
from centum import utils


import sys
# Get the parent directory and add it to sys.path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

import EOMAJI_utils
import os
import argparse
from matplotlib.animation import FuncAnimation
from matplotlib import animation
import rioxarray as rio

cwd = os.getcwd()
# prj_name = 'Majadas_2024_WTD1' #Majadas_daily Majadas_2024

AOI = 'burkina_faso_aoi'
EPGS = 32630 # Adapted for Burkina Faso
# EPGS = 4326

#%%

def get_cmd():
    parse = argparse.ArgumentParser()
    process_param = parse.add_argument_group('process_param')
    process_param.add_argument('-prj_name', type=str, help='prj_name',
                        default='BurkinaFaso', 
                        required=False
                        ) 
    process_param.add_argument('-AOI', type=str, help='Area of Interest',
                        default='', 
                        required=False
                        )
    process_param.add_argument('-dayMax', 
                               type=int, 
                               help='NbofSimulatedDays',
                               default=5, 
                               required=False
                               ) 
    process_param.add_argument('-WTD', type=float, 
                               help='WT height',
                        # default=100, required=False) 
                        # default=4,
                        default=2,
                        ) 
    process_param.add_argument('-SCF', type=float, help='Soil Cover Fraction',
                        default=1.0, required=False)

    args = parse.parse_args()
    return(args)    

args = get_cmd()   

figPath = Path(cwd) / '../figures/' / args.prj_name
figPath.mkdir(parents=True, exist_ok=True)
rootPrepoPath = Path(cwd) / '../../data/prepro/' / args.prj_name
rootDataPath= Path(cwd) / '../../data/'


#%% Import input data 

# crs_ET = EOMAJI_utils.get_crs_ET_fromfile()
crs_ET = "EPSG:27701"
# BurkinaFaso_aoi = gpd.read_file(f'../../data/AOI/{AOI}.geojson')
# BurkinaFaso_aoi_reproj = BurkinaFaso_aoi.to_crs(crs_ET)

ds_analysis_EO, RAIN = EOMAJI_utils.read_prepo_EO_datasets(fieldsite='BurkinaFaso',
                                                      # AOI=args.AOI,
                                                      rootPath=rootPrepoPath,
                                                      crs=crs_ET
                                                      )
RAIN['RAIN'].sum()

# args.dayMax = 35
def reproj_4326_and_clean_urban(ds_analysis_EO,RAIN,crs_ET):
    
    RAIN = RAIN.rio.write_crs(crs_ET)
    ds_analysis_EO = ds_analysis_EO.isel(time=slice(0, args.dayMax))
    ds_analysis_EO = ds_analysis_EO.rio.reproject(f"EPSG:{EPGS}",
                                                  nodata=np.nan)
        

    RAIN = RAIN.rio.reproject(f"EPSG:{EPGS}",
                              nodata=np.nan
                              )
    
    valid_mask_np_nan = ~np.isnan(ds_analysis_EO['ETa'].isel(time=0))
    valid_mask_np_nan_expanded = valid_mask_np_nan.expand_dims(time=ds_analysis_EO.time)
    ds_analysis_EO_clean = ds_analysis_EO.where(valid_mask_np_nan_expanded, drop=True)
    
    # Step 1: Apply filling on a copy of the dataset
    ds_filled = (
        ds_analysis_EO_clean
        .ffill(dim='x')
        .bfill(dim='x')
        .ffill(dim='y')
        .bfill(dim='y')
    )
    
    valid_mask_below5 = (ds_filled['ETa'].isel(time=0) < 5).values  # shape (y,x)
    valid_mask_below5_expanded = np.broadcast_to(valid_mask_below5, 
                                                 ds_filled['ETa'].shape
                                                 )
    
    ETa_masked = ds_filled['ETa'].where(valid_mask_below5_expanded)
    ETp_masked = ds_filled['ETp'].where(valid_mask_below5_expanded)
    ds_analysis_EO_clean2 = ds_analysis_EO_clean.copy()
    ds_analysis_EO_clean2['ETa'] = ETa_masked
    ds_analysis_EO_clean2['ETp'] = ETp_masked
    
    RAIN_interp = RAIN.interp_like(ds_analysis_EO_clean2.isel(time=0), kwargs={'fill_value': 'extrapolate'})   
    
    RAIN_filled = (
        RAIN_interp
        .ffill(dim='x')
        .bfill(dim='x')
        .ffill(dim='y')
        .bfill(dim='y')
    )
      
    ds_analysis_EO_clean2['RAIN'] = RAIN_filled['RAIN']
    ds_analysis_EO_clean2['valid_mask'] = (('time', 'y', 'x'), valid_mask_below5_expanded.astype(int))
    
    # fig, ax = plt.subplots()
    # ds_analysis_EO_clean2['ETa'].isel(time=0).plot.imshow(ax=ax)
    # ds_analysis_EO_clean2['ETp'].isel(time=0).plot.imshow(ax=ax)
    # RAIN_interp['RAIN'].isel(time=0).plot.imshow(ax=ax)
    # ds_analysis_EO_clean2['RAIN'].isel(time=0).plot.imshow(ax=ax)

    return ds_analysis_EO_clean2

ds_analysis_EO_clean2 = reproj_4326_and_clean_urban(ds_analysis_EO,RAIN,crs_ET)
# ds_analysis_EO_clean2['RAIN'].sum()

# rain_sum = RAIN['RAIN'].sum(dim=['x', 'y'], skipna=True)
# first_nonzero_time = rain_sum.where(rain_sum != 0, drop=True).time[0].index


fig, ax = plt.subplots()
ds_analysis_EO_clean2['ETp'].isel(time=0).plot.imshow(ax=ax)
# ds_analysis_EO_clean2['RAIN'].isel(time=0).plot.imshow(ax=ax)

# ds_analysis_EO_clean2['RAIN'].sum()

#%%


from rasterio.enums import Resampling
from affine import Affine
# Target shape
target_height = 120
target_width = 120

# target_height = 475
# target_width = 475

# Get spatial extent
x_min, x_max = float(ds_analysis_EO_clean2.x.min()), float(ds_analysis_EO_clean2.x.max())
y_min, y_max = float(ds_analysis_EO_clean2.y.min()), float(ds_analysis_EO_clean2.y.max())


# Calculate new resolution
res_x = (x_max - x_min) / target_width
res_y = (y_max - y_min) / target_height


transform = Affine.translation(x_min, y_max) * Affine.scale(res_x, -res_y)

# Apply reproject
ds_EO_resampled = ds_analysis_EO_clean2.rio.reproject(
                                                        dst_crs=ds_analysis_EO_clean2.rio.crs,
                                                        shape=(target_height, target_width),
                                                        transform=transform,
                                                        resampling=Resampling.bilinear
                                                    )

fig, axs = plt.subplots(1,2,sharey=True)
ds_analysis_EO_clean2['ETp'].isel(time=0).plot.imshow(ax=axs[0])
ds_EO_resampled['ETp'].isel(time=0).plot.imshow(ax=axs[1])
# ds_EO_resampled['RAIN'].isel(time=0).plot.imshow(ax=axs[2])

fig, ax = plt.subplots()
ds_EO_resampled['RAIN'].isel(time=0).plot.imshow()

# ds_EO_resampled['ETp_fill0'] = ds_EO_resampled[np.where(ds_EO_resampled['ETp']==np.nan,0)
ds_EO_resampled['ETp_fill0'] = ds_EO_resampled['ETp'].fillna(0)

fig, ax = plt.subplots()
ds_EO_resampled['ETp_fill0'].isel(time=0).plot.imshow()


# Reproject to UTM (automatically chooses the UTM zone based on dataset extent)
ds_proj = ds_EO_resampled.rio.reproject(crs_ET
                                        )  # Example: UTM zone 30N

# Get resolution in meters
deltax, deltay = ds_proj['ETp'].rio.resolution()
deltax = np.round(abs(deltax))  # x-resolution in meters
deltay = np.round(abs(deltay))  # y-resolution in meters






# s
# gg
# if args.short==True:
#     cutoffDate = ['01/01/2023','01/03/2024']
#     start_time, end_time = pd.to_datetime(cutoffDate[0]), pd.to_datetime(cutoffDate[1])
#     mask_time = (ds_analysis_EO.time >= start_time) & (ds_analysis_EO.time <= end_time)
#     # Filter the DataArrays using the mask
#     ds_analysis_EO = ds_analysis_EO.sel(time=mask_time)

#%% Create CATHY mesh based on DEM
# ----------------------------------------------------------------------------

plt.close('all')
# prjname = f'{args.prj_name}_DOI_{args.AOI}_WTD{args.WT}'
prjname = '_'.join(f"{key}_{value}" for key, value in vars(args).items())

hydro_BurkinaFaso = CATHY(
                        dirName='../../WB_FieldModels/BurkinaFaso/',
                        prj_name=prjname
                      )
BurkinaFasoPath = Path(hydro_BurkinaFaso.workdir) / hydro_BurkinaFaso.project_name

#%% Create CATHY mesh based on DEM
# Update prepro inputs and mesh

# no topo case 
# ----------------------------------------------------------------------------



rain_avg = ds_EO_resampled['RAIN'].mean(dim=["x", "y"], skipna=True)

plt.figure(figsize=(12, 5))
plt.plot(rain_avg.time.values, rain_avg.values, lw=0.8)
plt.title("Rainfall Time Series")
plt.xlabel("Date")
plt.ylabel("Rainfall (units)")
plt.grid(True)
plt.tight_layout()
   

# Compute spatial means
eta_avg = ds_EO_resampled['ETa'].mean(dim=["x", "y"], skipna=True)

# Create figure and twin axis
fig, ax1 = plt.subplots(figsize=(12, 5))

# Plot Rain on left y-axis
ax1.plot(rain_avg.time.values, rain_avg.values, color='tab:blue', lw=0.8, label="Rain")
ax1.set_xlabel("Date")
ax1.set_ylabel("Rainfall (mm/day)", color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.grid(True)

# Create second y-axis for ETa
ax2 = ax1.twinx()
ax2.plot(eta_avg.time.values, eta_avg.values*(1e-3/86400), color='tab:green', lw=0.8, label="ETa")
ax2.set_ylabel("ETa (mm/day)", color='tab:green')
ax2.tick_params(axis='y', labelcolor='tab:green')

# Title and layout
plt.title("Rainfall and ETa Time Series")
fig.tight_layout()

#%%

# ymin = ds_EO_resampled.y.min().item()
# ds_ymin = ds_EO_resampled['ETa'].sel(y=ymin, method='nearest').isel(time=0)
# first_valid_idx = ds_ymin.notnull().argmax().values

# ds_EO_resampled['ETa'].isel(time=0,y=0,x=first_valid_idx)
# mask9999dem = ds_EO_resampled['ETa'].isel(time=0).isnull()

# mask9999dem.plot.imshow()
# ds_EO_resampled['ETa'].isel(time=0).plot.imshow()

# DEM_notopo = np.ones(np.shape(mask9999dem)) #*1e-3
# DEM_notopo[mask9999dem] = -9999
# # DEM_notopo[0,first_valid_idx] = 0

# # ff

# deltax = np.round(ds_EO_resampled['ETa'].rio.resolution()[0])
# deltay = np.round(ds_EO_resampled['ETa'].rio.resolution()[1])


# # create_gently_sloped_dem(DEM_notopo)
# DEMnew, mask, outlet_coord = EOMAJI_utils.create_gently_sloped_dem(DEM_notopo,
#                                                                    base_elevation=1
#                                                                    )

# DEMnew = np.where(np.isnan(DEMnew), -9999, DEMnew)

#%%


#%%

DEMnew = np.ones(np.shape(ds_EO_resampled['ETp'].isel(time=0))) #*1e-3


# Reproject to UTM (automatically chooses the UTM zone based on dataset extent)
ds_proj = ds_EO_resampled.rio.reproject(crs_ET
                                        )  # Example: UTM zone 30N

# Get resolution in meters
deltax, deltay = ds_proj['ETp'].rio.resolution()
deltax = np.round(abs(deltax))  # x-resolution in meters
deltay = np.round(abs(deltay))  # y-resolution in meters


deltax = np.round(ds_EO_resampled['ETp'].rio.resolution()[0])
deltay = np.round(ds_EO_resampled['ETp'].rio.resolution()[1])
# np.shape(DEMnew)

#%%
# DEMnew, mask, outlet_coord = EOMAJI_utils.create_gently_sloped_dem(DEMnew,
                                                                   # base_elevation=0,
                                                                   # boundary_raise = 1e-3,
                                                                   # smooth_sigma = 1
                                                                   # )
# np.shape(DEMnew)
DEMnew[1,1] = DEMnew[1,1] - 1e-3
# s
#%%
maxdepth = 10
zb = np.linspace(0, maxdepth, 10)
nstr = len(zb) - 1
zr = list((np.ones(len(zb))) / (nstr))

hydro_BurkinaFaso.update_prepo_inputs(
                                DEM=DEMnew,
                                # DEM=np.ones(np.shape(DEM_notopo)),
                                # DEM=np.ones([600,600]),
                                dr = deltax/2,
                                # dr = 10,
                                # ivert=1,
                                delta_x = abs(deltax),
                                delta_y = abs(deltay),
                                # delta_x = 10,
                                # delta_y = 10,
                                # N=50,
                                # M=10,
                                # xllcorner=0,
                                # yllcorner=0,
                                imethod=2,
                                ndcf=1,
                                # base=6,
                                xllcorner=ds_EO_resampled['ETa'].x.min().values,
                                yllcorner=ds_EO_resampled['ETa'].y.min().values,
                                # zratio=zr,
                                base=max(zb),
                                )


#% Update prepro inputs and mesh

fig = plt.figure()
ax = plt.axes(projection="3d")
hydro_BurkinaFaso.show_input(prop="dem", ax=ax)

#%%
hydro_BurkinaFaso.create_mesh_vtk(verbose=True)
grid3d = hydro_BurkinaFaso.read_outputs('grid3d')

# s
#%% Update atmbc according to EO
# -----------------------------------------------------------------------------
print('interpolate vars on surface_nodes')
mesh3d_nodes = grid3d['mesh3d_nodes']
# np.shape(mesh3d_nodes)

interpolate_ET_on_mesh_nodes = False
if interpolate_ET_on_mesh_nodes:
    ds_EO_resampled_nodes = EOMAJI_utils.interpolate_vars_on_surface_nodes(ds_EO_resampled, 
                                                                           mesh3d_nodes, 
                                                                           n_jobs=10
                                                                           )  # or n_jobs=-1 for all CPUs



ds_EO_resampled_nodes_Rain = EOMAJI_utils.xarraytoDEM_pad(ds_EO_resampled['RAIN'])
ds_EO_resampled_nodes_ETp = EOMAJI_utils.xarraytoDEM_pad(ds_EO_resampled['ETp_fill0'])
# np.shape(ds_EO_resampled_nodes)
# ds_EO_resampled_nodes = xr.open_dataset(rootPrepoPath/'ds_EO_resampled_nodes.netcdf')
# ds_EO_resampled_nodes['net_atmbc'] = (
#                                         ds_EO_resampled_nodes['RAIN_surface_nodes']*(1e-3/86400) -
#                                         ds_EO_resampled_nodes['ETp_surface_nodes']*(1e-3/86400)
#                                     )
#%%
for var in ds_EO_resampled.data_vars:
    if 'grid_mapping' in ds_EO_resampled[var].attrs:
        ds_EO_resampled[var].attrs.pop('grid_mapping')

# for var in ds_EO_resampled_nodes.data_vars:
#     if 'grid_mapping' in ds_EO_resampled_nodes[var].attrs:
#         ds_EO_resampled_nodes[var].attrs.pop('grid_mapping')
        
ds_EO_resampled.to_netcdf(os.path.join(hydro_BurkinaFaso.workdir,
                                              hydro_BurkinaFaso.project_name,
                                              'ds_EO_resampled.netcdf'
                                              )
                                )

# ds_EO_resampled_nodes.to_netcdf(os.path.join(hydro_BurkinaFaso.workdir,
#                                               hydro_BurkinaFaso.project_name,
#                                               'ds_EO_resampled_nodes.netcdf'
#                                               )
#                                 )
#%%
ds_EO_resampled_nodes = (
                            ds_EO_resampled_nodes_Rain*(1e-3/86400) -
                            ds_EO_resampled_nodes_ETp*(1e-3/86400)
                        )

net_atmbc_nodes = ds_EO_resampled_nodes.stack(space=('y', 'x')).values 
 
#%%


ds_EO_resampled['Elapsed_Time_s'] = (ds_EO_resampled.time - ds_EO_resampled.time[0]).dt.total_seconds()
test_vatmbc_reshaped = np.ones([len(list(ds_EO_resampled['Elapsed_Time_s'].values)),
                                 int(grid3d['nnod'])]
                                )*(-1e-7)
np.shape(test_vatmbc_reshaped)

hydro_BurkinaFaso.update_atmbc(
                            HSPATM=0,
                            IETO=1,
                            time=list(ds_EO_resampled['Elapsed_Time_s'].values),
                            netValue=net_atmbc_nodes
                            # netValue=test_vatmbc_reshaped
                            )
#%%

ds_EO_resampled_nodes_Rain_avg = ds_EO_resampled_nodes_Rain.mean(dim=['x','y'], 
                                                            skipna=True)

plt.figure(figsize=(12, 5))
plt.plot(ds_EO_resampled_nodes_Rain_avg.time.values, ds_EO_resampled_nodes_Rain_avg.values, lw=0.8)
plt.title("Rainfall Time Series")
plt.xlabel("Date")
plt.ylabel("Rainfall (units)")
plt.grid(True)
plt.tight_layout()
plt.show()

#%%

# hydro_BurkinaFaso.show_input('atmbc',
#                              )

#%%
datetimeEO=np.unique(ds_EO_resampled_nodes_Rain_avg.time.values)

# hydro_BurkinaFaso.show_input('atmbc',
#                              datetime=np.hstack([np.unique(ds_EO_resampled_nodes.time.values)]*11689)
#                              )
# len(np.unique(ds_EO_resampled_nodes.time.values))
# 4266485/365

#%%

import matplotlib.pyplot as plt

# Extract original dataset surface node coordinates
x_ds = ds_EO_resampled['x'].values
y_ds = ds_EO_resampled['y'].values

x = ds_EO_resampled_nodes_Rain['x'].values
y = ds_EO_resampled_nodes_Rain['y'].values
net = net_atmbc_nodes[0]

# Extract mesh3d node coordinates (x, y)
x_mesh = mesh3d_nodes[:, 0]
y_mesh = mesh3d_nodes[:, 1]

plt.figure(figsize=(10, 8))
ds_EO_resampled_nodes_ETp.isel(time=0).plot.imshow()
ds_EO_resampled_nodes_ETp.isel(time=1).plot.imshow()


# sc = plt.scatter(x, y, c=net, cmap='coolwarm', s=10)

# Plot mesh3d nodes points
# plt.scatter(x_mesh, y_mesh, s=10, c='red', label='mesh3d nodes', alpha=0.4)

plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.title('Scatter plot of dataset surface nodes and mesh3d nodes')
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.show()



#%% Show atmbc time serie
# -----------------------------------------------------------------------------
# hydro_BurkinaFaso.show_input('atmbc')

#%%
hydro_BurkinaFaso.update_nansfdirbc(no_flow=True)
hydro_BurkinaFaso.update_nansfneubc(no_flow=True)
hydro_BurkinaFaso.update_sfbc(no_flow=True)

#%% Update iniital conditions
# -----------------------------------------------------------------------------
# hydro_BurkinaFaso.update_ic(
#                         INDP=0, 
#                         IPOND=0, 
#                         pressure_head_ini=-10
#                     )

hydro_BurkinaFaso.update_ic(
                        INDP=4, 
                        WTPOSITION=args.WTD, 
                    )

#%% Update root depth according to CLC mapping
# -----------------------------------------------------------------------------
 # '244': 'Agro-forestry areas',
 # '212': 'Permanently irrigated land'
 
 
# CLC_Majadas_clipped_grid = xr.open_dataset(f'../prepro/Majadas/{args.AOI}/CLCover_Majadas.netcdf',
#                                             # engine='scipy'
#                                            )
# (reprojected_CLC_Majadas,
#  mapped_data )=  Majadas_utils.get_Majadas_root_map_from_CLC(ds_analysis_EO,
#                                                             CLC_Majadas_clipped_grid,
#                                                             crs_ET
#                                                             )


# # Create the figure and axis
# fig, ax = plt.subplots()
# reprojected_CLC_Majadas.Code_CLC.where(reprojected_CLC_Majadas.Code_CLC == 244).plot.imshow(
#     ax=ax, cmap='Greens', add_colorbar=False)
# reprojected_CLC_Majadas.Code_CLC.where(reprojected_CLC_Majadas.Code_CLC == 212).plot.imshow(
#     ax=ax, cmap='Blues', add_colorbar=False)
# ax.set_title('Land Cover Types: 244 (Green) and 212 (Blue)')
    
    

# ss
#%%

print('''
      !!!13 Corinne Land Cover distribution is not recongnised by CATHY remove! 
      Same issue than Daniele La Cec.
      '''
      )

#%% Corinne land cover shapefile to raster
# -----------------------------------------------------------------------------
clc_codes = utils.get_CLC_code_def()
CLC_BurkinaFaso_clipped = BurkinaFaso_utils.get_LandCoverMap()

CLC_BurkinaFaso_clipped_proj = CLC_BurkinaFaso_clipped.rio.reproject(ds_EO_resampled.rio.crs
                                                                     ) 



CLC_root_depth = EOMAJI_utils.CLC_LookUpTable(rootPrepoPath/'../../LandCover/BurkinaFaso_CGLOPS_LUT')

rootDepth_mapping_dict = CLC_root_depth['rootDepth'].to_dict()


CLC_BurkinaFaso_resampled = CLC_BurkinaFaso_clipped_proj.interp_like(ds_EO_resampled.isel(time=0))

fig, axs = plt.subplots(1,2,sharey=True)
CLC_BurkinaFaso_clipped_proj['landcover'].plot.imshow(ax=axs[0])
CLC_BurkinaFaso_resampled['landcover'].plot.imshow(ax=axs[1])

CLC_BurkinaFaso_resampled['rootDepth'] = xr.apply_ufunc(
    lambda x: rootDepth_mapping_dict.get(int(x) if not np.isnan(x) else -1, 1e-4),
    CLC_BurkinaFaso_resampled['landcover'],
    vectorize=True,
    dask='parallelized',
    output_dtypes=[float],
)

# Get unique non-NaN land cover values
unique_rootDepth = np.unique(CLC_BurkinaFaso_resampled['rootDepth'].values[~np.isnan(CLC_BurkinaFaso_resampled['rootDepth'].values)])
# Create a mapping: landcover value → integer starting from 1
veg_mapping = {val: i +1 for i, val in enumerate(sorted(unique_rootDepth))}

CLC_BurkinaFaso_resampled['vegMap'] = xr.apply_ufunc(
    lambda x: veg_mapping.get(x if not np.isnan(x) else -1, 0),
    CLC_BurkinaFaso_resampled['rootDepth'],
    vectorize=True,
    dask='parallelized',
    output_dtypes=[float],
)

fig, axs = plt.subplots(1,2,sharey=True)
CLC_BurkinaFaso_resampled['landcover'].plot.imshow(ax=axs[0])
# CLC_BurkinaFaso_clipped['rootDepth'].plot.imshow(ax=axs[1],
#                                                  # vmin=,
#                                                  )
CLC_BurkinaFaso_resampled['vegMap'].plot.imshow(ax=axs[1],
                                                 )
CLC_BurkinaFaso_resampled['rootDepth'].rio.to_raster(rootPrepoPath/'../../LandCover/BurkinaFaso_rootDeph_LC.tif')

#%%

# d
np.shape(CLC_BurkinaFaso_resampled['vegMap'].values)
hydro_BurkinaFaso.update_veg_map(CLC_BurkinaFaso_resampled['vegMap'].values)
hydro_BurkinaFaso.show_input('root_map')

SPP_map = hydro_BurkinaFaso.init_soil_SPP_map_df(nzones=1, nstr=15)
SPP_map = hydro_BurkinaFaso.set_SOIL_defaults(SPP_map_default=True)

# self.veg_map

FP_map = hydro_BurkinaFaso.init_soil_FP_map_df(nveg=len(np.unique(CLC_BurkinaFaso_resampled['vegMap'].values)))
FP_map = hydro_BurkinaFaso.set_SOIL_defaults(FP_map_default=True)

veg_mapping = {i +1: float(val) for i, val in enumerate(sorted(unique_rootDepth))}
FP_map['ZROOT'] = FP_map.index.map(veg_mapping)


hydro_BurkinaFaso.update_soil(
                            PMIN=-200,
                            # PMIN=-1e35,
                            FP_map=FP_map,
                            SPP_map=SPP_map,
                            SCF=args.SCF,
                            show=True
                          )
# hydro_BurkinaFaso.cathyH

# from pyCATHY.plotters import cathy_plots as plt_CT
# update_map_veg = hydro_BurkinaFaso.map_prop_veg(FP_new)
# fig, ax = plt_CT.dem_plot_2d_top(update_map_veg,
#                                   label="all"
#                                   )
fig, ax = plt.subplots(1)
hydro_BurkinaFaso.show_input(prop="root_map", ax=ax,
                          # linewidth=0
                          )


# aa
#%% Run simulation

hydro_BurkinaFaso.update_parm(
                        TIMPRTi=ds_EO_resampled['Elapsed_Time_s'].values,
                        # TIMPRTi=resample_times_vtk,
                        IPRT=4,
                        VTKF=2, # dont write vtk files
                        )
#%%
plt.close('all')
hydro_BurkinaFaso.run_processor(
                      IPRT1=2,
                      TRAFLAG=0,
                      DTMIN=1e-2,
                      DTMAX=1e4,
                      DELTAT=5e3,
                      verbose=True
                      )

#%%

df_sw, _ = hydro_BurkinaFaso.read_outputs('sw')
df_sw.head()

#%%
node, node_pos = hydro_BurkinaFaso.find_nearest_node([5e3,5e3,1])
node2, node_pos2 = hydro_BurkinaFaso.find_nearest_node([5,5,1])
print(node_pos[0])

pl = pv.Plotter(notebook=False)
cplt.show_vtk(unit="pressure",
              timeStep=1,
              path=os.path.join(hydro_BurkinaFaso.workdir,
                                hydro_BurkinaFaso.project_name,
                                'vtk'
                                ),
              # style='wireframe',
              # opacity=0.5,
              ax=pl,
              )
pl.add_points(node_pos[0],
              color='red'
              )
# pl.add_points(node_pos2[0],
#               color='red'
#               )
pl.show()

fig, ax = plt.subplots()
df_sw[node].plot(ax=ax)
# df_sw[node2].plot(ax=ax)
ax.set_xlabel('time (s)')
ax.set_ylabel('saturation (-)')

# datetimeEO

from pyCATHY import cathy_utils

datetimeEO_sw = cathy_utils.change_x2date(df_sw.index,start_date=datetimeEO[0])
df_sw.index = datetimeEO_sw
df_sw['datetimeEO_sw'] = datetimeEO_sw
df_sw.set_index('datetimeEO_sw', inplace=True)

fig, ax = plt.subplots()
df_sw[node].plot(ax=ax)
# df_sw[node2].plot(ax=ax)
ax.set_xlabel('time (s)')
ax.set_ylabel('saturation (-)')

#%%


cplt.show_vtk_TL(
                unit="saturation",
                notebook=False,
                path=hydro_BurkinaFaso.workdir + hydro_BurkinaFaso.project_name + "/vtk/",
                show=False,
                x_units='days',
                # clim = [0.55,0.70],
                clim = [0.2,0.5],
                savefig=True,
            )

#%%


cplt.show_vtk_TL(
                unit="pressure",
                notebook=False,
                path=hydro_BurkinaFaso.workdir + hydro_BurkinaFaso.project_name + "/vtk/",
                show=False,
                x_units='days',
                # clim = [0.55,0.70],
                clim = [-25,0],
                savefig=True,
            )

#%%

ET = hydro_BurkinaFaso.read_outputs('ET')

# df_fort777 = out_CT.read_fort777(
#     os.path.join(self.workdir,
#                  self.project_name,
#                  f'DA_Ensemble/cathy_{nensi+1}',
#                  'fort.777'
#                  )
# )
ds_EO_resampled_nodes_Rain_avg.plot()
# hydro_BurkinaFaso.show_input('atmbc')

ET = ET.drop_duplicates()
ET = ET.set_index(['time', 'X', 'Y']).to_xarray()
ET['ACT. ETRA_mmday'] = ET['ACT. ETRA'] * 1000 * 86400

ET.to_netcdf(hydro_BurkinaFaso.workdir + hydro_BurkinaFaso.project_name + "ET.netcdf")

ET['ACT. ETRA'].isel(time=0).plot.imshow()
ET['ACT. ETRA'].isel(time=1).plot.imshow()

ET['ACT. ETRA_mmday'].isel(time=1).plot.imshow()

#%%

import matplotlib.pyplot as plt
import imageio
import os
import numpy as np
import pandas as pd

os.makedirs("gif_frames", exist_ok=True)

data = ET['ACT. ETRA_mmday']  # ETa data: dims (time, X, Y)
rain_avg = ds_EO_resampled_nodes_Rain_avg  # Your 1D Rain_avg time series
# ET_avg = ds_EO_resampled_nodes_ETp_avg  # Your 1D Rain_avg time series

vmin = float(data.min())
vmax = float(data.max())

start_date = np.datetime64('2018-01-01')

filenames = []

for i in range(data.sizes['time']):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # 1) ETa plot
    im = ax1.imshow(data.isel(time=i).T, origin="lower", cmap='viridis',
                    vmin=vmin, vmax=vmax,
                    extent=[
                        float(data.X.min()), float(data.X.max()),
                        float(data.Y.min()), float(data.Y.max())
                    ])
    absolute_date = start_date + data.time[i].values  # timedelta64 + datetime64
    time_label = pd.to_datetime(absolute_date).strftime('%Y-%m-%d')
    ax1.set_title(f"ETa (mm/day) - Time: {time_label}")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    plt.colorbar(im, ax=ax1, label='mm/day')

    # 2) Rain_avg time series plot
    ax2.plot(rain_avg.time.values, rain_avg.values, label='Rain Avg')
    # Find index in rain_avg.time closest to current absolute_date
    idx = np.searchsorted(rain_avg.time.values, absolute_date)
    if idx >= len(rain_avg.time):
        idx = len(rain_avg.time) - 1
    # Plot moving red marker
    ax2.plot(rain_avg.time.values[idx], rain_avg.values[idx], 'ro')
    ax2.set_title('Rain Avg Time Series')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Rainfall')
    ax2.legend()
    fig.autofmt_xdate()

    fname = f"gif_frames/frame_{i:03d}.png"
    plt.savefig(fname)
    plt.close(fig)
    filenames.append(fname)

# Create GIF
gif_path = hydro_BurkinaFaso.workdir + hydro_BurkinaFaso.project_name + "ETa_animation.gif"
with imageio.get_writer(gif_path, mode='I', duration=0.5) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

# Cleanup
for filename in filenames:
    os.remove(filename)
os.rmdir("gif_frames")


#%%


