#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 14:13:38 2024
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
# import DEPRECATED_utils as utils
# from centum import utils
import Majadas_utils
import os
import argparse
from shapely.geometry import mapping
from scipy import stats
from pyCATHY import cathy_utils
import matplotlib.dates as mdates

cwd = os.getcwd()
# prj_name = 'prj_name_Majadas_AOI_Buffer_100_WTD_5.0_short_1_SCF_1.0_OMEGA_1'
# prj_name = 'prj_name_Majadas_AOI_Buffer_100_WTD_2.0_short_1_SCF_1.0_OMEGA_1'
# prj_name = 'prj_name_Majadas_AOI_Buffer_100_WTD_1.0_short_1_SCF_1.0_OMEGA_1'
# prj_name = 'prj_name_Majadas_AOI_Buffer_100_WTD_5.0_short_0_SCF_1.0_OMEGA_1'
prj_name = 'prj_name_Majadas_AOI_Buffer_100_WTD_2.0_short_0_SCF_1.0_OMEGA_1'


figPath = Path(cwd) / '../../figures/' / prj_name
figPath.mkdir(parents=True, exist_ok=True)

prepoEOPath = Path('/run/media/z0272571a/SENET/iberia_daily/E030N006T6')

plt.close('all')
hydro_Majadas = CATHY(
                        dirName='../../WB_FieldModels/',
                        prj_name=prj_name
                      )
plt.rcParams['font.family'] = 'serif'  # You can also use 'Times New Roman', 'STIXGeneral', etc.

#%% Import input data 


file_pattern = '*TPday*.tif'
ET_0_filelist = list(prepoEOPath.glob(file_pattern))
crs_ET = rxr.open_rasterio(ET_0_filelist[0]).rio.crs
ET_test = rxr.open_rasterio(ET_0_filelist[0])

ds_analysis_EO = Majadas_utils.read_prepo_EO_datasets(fieldsite='Majadas',
                                                      crs=crs_ET
                                                      )

#%% Get areas and point of interest 

gdf_AOI_POI_Majadas = Majadas_utils.get_AOI_POI_Majadas(crs_ET)
majadas_aoi = gpd.read_file('../../data/AOI/majadas_aoi.geojson')
majadas_POIs, POIs_coords = Majadas_utils.get_Majadas_POIs()

is_point = gdf_AOI_POI_Majadas.geometry.geom_type == 'Point'
DEM, hd_DEM = hydro_Majadas.read_inputs('dem')
df_atmbc = hydro_Majadas.read_inputs('atmbc')
df_atmbc.groupby('time').mean()

#%%
gdf_AOI_POI_Majadas['meshNode'] = None
for geom, indexPOI in zip(gdf_AOI_POI_Majadas[is_point].geometry,gdf_AOI_POI_Majadas[is_point].index):
    x, y = geom.coords[0]
    meshNodePOI, closest = hydro_Majadas.find_nearest_node([x,y, np.max(DEM)])
    gdf_AOI_POI_Majadas.loc[indexPOI,'meshNode'] = meshNodePOI[0]
    
# gdf_AOI_POI_Majadas['meshNode']

#%%
POROSITY_GUESS = 0.55
#%%
df_sw, _ = hydro_Majadas.read_outputs('sw')
df_psi = hydro_Majadas.read_outputs('psi')

#%%
start_date = ds_analysis_EO.time.isel(time=0).values
dates = cathy_utils.change_x2date(df_sw.index.values, 
                                  ds_analysis_EO.time.isel(time=0).values,
                                  # start_date = '2023-01-01',
                                  formatIn = '%Y-%m-%d',
                                  formatOut="%Y-%m-%d"
                                  )
# len(df_sw.index.values)
# Create a figure and the first axis
# fig, ax = plt.subplots()
#%% Read TDR field sensors
coord_SWC_CT, gdf_SWC_CT = Majadas_utils.get_SWC_pos(
                                                    target_crs=crs_ET
                                                    )
TDR_SWC, depths = Majadas_utils.get_SWC_data()
TDR_SWC.columns


 # 'SWC_2014_S*' in TDR_SWC.columns 
 
SWC_pos_root = 'SWC_2014_S'
matching_columns = [col for col in TDR_SWC.columns if col.startswith(SWC_pos_root+'_')]
gdf_AOI_POI_Majadas.set_index('SWC sensor').loc[SWC_pos_root]['meshNode']

#%%

gdf_AOI_POI_Majadas['meshNode'] = None
for geom, indexPOI in zip(gdf_AOI_POI_Majadas[is_point].geometry,gdf_AOI_POI_Majadas[is_point].index):
    x, y = geom.coords[0]
    meshNodePOI, closest = hydro_Majadas.find_nearest_node([x,y, np.max(DEM)])
    gdf_AOI_POI_Majadas.loc[indexPOI,'meshNode'] = meshNodePOI[0]

    
x, y =  gdf_AOI_POI_Majadas.iloc[4].geometry.coords[0]
# depths = [5,50,100,200]
# depths = [5,100,400]
depths = [5,100]

hydro_Majadas.grid3d

grid3d = hydro_Majadas.read_outputs('grid3d')
# grid3d['mesh3d_nodes'][meshNodes2plot]

meshNodes2plot = []
for d in depths:
    meshNodePOI, closest = hydro_Majadas.find_nearest_node([x,y, np.max(DEM)-d/100])
    meshNodes2plot.append(meshNodePOI)
    
#%%
rain = ds_analysis_EO['RAIN'].mean(dim=['x','y'])



#%%
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

ax['a'].bar(rain.time,
            rain.values,
        )
ax['a'].invert_yaxis()
ax['a'].set_ylabel('rain (mm)', color='k')
ax['a'].yaxis.label.set_color('k')
ax['a'].tick_params(axis='y', colors='k')


styles = ['-','--','-.']
for i, node in enumerate(meshNodes2plot):
    # print(i)
    ax['b'].plot(dates, 
                    df_sw[node].values*POROSITY_GUESS*100, 
                    color='darkblue',
                    linestyle = styles[i]
                    )  
ax['b'].set_ylabel('Estimated SMC', color='darkblue')
# Set color of the right y-axis ticks and label
ax['b'].yaxis.label.set_color('darkblue')
ax['b'].tick_params(axis='y', colors='darkblue')
ax['b'].legend(labels=depths)

depths_field = [5,100]
ax_b_right = ax['b'].twinx()
for i, di in enumerate(depths_field):
    ax_b_right.plot(TDR_SWC.index, 
                    TDR_SWC[f'{SWC_pos_root}_{di}cm'], 
                    color='k',
                    linestyle = styles[i],
                    label = str(di)
                    ) 
ax_b_right.set_ylabel('Measured SMC')
# plt.legend()
plt.tight_layout()
plt.savefig(figPath/'saturation_simu_Majadas.png',
            dpi=300
            )
#%%
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

rain = ds_analysis_EO['RAIN'].mean(dim=['x','y'])

ax['a'].bar(rain.time,
            rain.values,
        )
ax['a'].invert_yaxis()
ax['a'].set_ylabel('rain (mm)', color='b')

# gdf_AOI_POI_Majadas.set_index('POI/AOI').columns
# Plot the first dataset on the primary y-axis
for node in meshNodes2plot:
    ax['b'].plot(dates, 
                 df_psi[node], 
                 'b-',
                 )  

ax['b'].legend()
ax['b'].set_xlabel('time (s)')
ax['b'].set_ylabel('Pressure head (-)', color='b')
ax['b'].tick_params(axis='y', labelcolor='b')

plt.tight_layout()
plt.savefig(figPath/'saturation_simu_Majadas.png',
            dpi=300
            )

#%% Plot time lapse saturation

# try:
#     cplt.show_vtk_TL(
#                     unit="saturation",
#                     notebook=False,
#                     path= str(Path(hydro_Majadas.workdir) / hydro_Majadas.project_name / "vtk"),
#                     show=False,
#                     x_units='days',
#                     # clim = [0.55,0.70],
#                     savefig=True,
#                 )
# except:
#     pass

# TDR_SWC.columns
# # SMC_XY = [list(pi) for pi in POIs_coords]
# # SMC_XY.append(list(gdf_SWC_CT.iloc[0].geometry.coords[0]))
# SMC_XY=[[gdf_SWC_CT.iloc[i].geometry.coords[0][0],gdf_SWC_CT.iloc[i].geometry.coords[0][1]] for i in range(len(gdf_SWC_CT))]
# SMC_XY = np.vstack(SMC_XY)
# SMC_depths = [-di/100+1 for di in depths] # add altitude of DEM =1 (flat no top case)

# SMC_XY = np.array([list(geom.coords[0]) for geom in gdf_SWC_CT.geometry])


#%% Plot SMC sensors on top of vtk mesh

# Plot the mesh and the sensor points
pl = pv.Plotter(notebook=True)
mesh = pv.read(Path(hydro_Majadas.workdir) / 
               hydro_Majadas.project_name / 
               f'vtk/{hydro_Majadas.project_name}.vtk'
               )

pl.add_mesh(mesh, opacity=0.1)

for i, cp in enumerate(gdf_AOI_POI_Majadas[is_point].geometry):
    x, y = cp.coords[0]
    cp_coords = np.array([x, y, 1])
    pl.add_points(cp_coords, 
                  color='red',
                  label=gdf_AOI_POI_Majadas[is_point]['POI/AOI'].iloc[i]
                  # markersize=10
                  )
pl.add_legend()
pl.show_grid()
pl.show()


#%% read ET actual and create it as an xarray
from pyCATHY.importers import cathy_outputs as out_CT

df_fort777 = out_CT.read_fort777(os.path.join(hydro_Majadas.workdir,
                                              hydro_Majadas.project_name,
                                              'fort.777'),
                                 )
df_fort777.time.unique()

df_fort777_select_t_xr = df_fort777.set_index(['time','y','x']).to_xarray()
# df_fort777_select_t_xr = df_fort777_select_t_xr.rename({'Y': 'y','X':'x'})

# df_fort777_select_t_xr = df_fort777_select_t_xr.rio.set_spatial_dims('y','x', inplace=True)

#%%
from matplotlib.colors import ListedColormap, BoundaryNorm

# Load your GeoDataFrame
gdf = gdf_AOI_POI_Majadas  # Assuming it's already loaded

# Encode the 'POI/AOI' column with unique integers
gdf['category_code'] = gdf['POI/AOI'].astype('category').cat.codes

# Create a colormap with distinct colors
categories = gdf['POI/AOI'].unique()
n_categories = len(categories)
cmap = ListedColormap(plt.cm.get_cmap("tab10").colors[:n_categories])

# Define normalization to map category codes to colors
norm = BoundaryNorm(range(n_categories + 1), cmap.N)

# Plot
fig, ax = plt.subplots(figsize=(10, 8))
gdf.plot(column='category_code', cmap=cmap, linewidth=0.5, edgecolor='k', ax=ax)

# Add a discrete colorbar
cbar = plt.colorbar(
    plt.cm.ScalarMappable(cmap=cmap, norm=norm),
    ax=ax,
    boundaries=range(n_categories + 1),
    ticks=range(n_categories),
)
cbar.ax.set_yticklabels(categories)
cbar.set_label('POI/AOI Categories')

# Title and axis adjustments
ax.set_title('Categorical Map of POI/AOI')
ax.axis('off')

plt.show()

#%% Clip ET xarray to geometry of land cover 
gdf_AOI_POI_Majadas.plot()
        
LCnames = ['irrigated','agroforestry']
fig, axs = plt.subplots(1,len(LCnames),
                        sharex=True,sharey=True,
                        )
df_fort777_select_t_xr = Majadas_utils.clip_ET_withLandCover(LCnames = LCnames,
                                                       gdf_AOI = gdf_AOI_POI_Majadas,
                                                       ETxr = df_fort777_select_t_xr,
                                                       ETname = 'ACT. ETRA',
                                                       crs_ET = crs_ET,
                                                       axs = axs
                                                    )
fig.savefig(f'{figPath}/WB_ETa_CATHY_CLCmask.png',
            dpi=300
            )


fig, axs = plt.subplots(1,len(LCnames),
                        sharex=True,sharey=True
                        )
ds_analysis_EO = Majadas_utils.clip_ET_withLandCover(LCnames = LCnames,
                                              gdf_AOI = gdf_AOI_POI_Majadas,
                                              ETxr = ds_analysis_EO,
                                              ETname = 'ETa',
                                              crs_ET = crs_ET,
                                               axs = axs
                                            )
fig.savefig(f'{figPath}/TSEB_ETa_CATHY_CLCmask.png',
            dpi=300
            )
    
    

#%% Compare ETa agroforestry with ETa irrigated land

ETa_baseline_irrigated = df_fort777_select_t_xr[LCnames[0] + '_CLCmask'].mean(dim=['x','y'])* 1000 * 86400
ETa_baseline_agroforestry = df_fort777_select_t_xr[LCnames[1] + '_CLCmask'].mean(dim=['x','y'])* 1000 * 86400
ETa_baseline_datetimes = ds_analysis_EO.time.isel(time=0).values + ETa_baseline_irrigated.time.values

ETa_TSEB_irrigated = ds_analysis_EO[LCnames[0] + '_CLCmask'].mean(dim=['x','y'])
ETa_TSEB_agroforestry = ds_analysis_EO[LCnames[1] + '_CLCmask'].mean(dim=['x','y'])

fig, axs = plt.subplots(1,2,sharex=True,sharey=True)

axs[0].scatter(ETa_baseline_irrigated,
           ETa_baseline_agroforestry,
           )
axs[1].scatter(ETa_TSEB_irrigated,
               ETa_TSEB_agroforestry,
               )

y_pred_baseline, r2_baseline = Majadas_utils.perf_linreg(ETa_baseline_irrigated,
                                           ETa_baseline_agroforestry
                                           )
y_pred_TSEB, r2_TSEB = Majadas_utils.perf_linreg(ETa_TSEB_irrigated,
                                   ETa_TSEB_agroforestry,
                                   )
len(ETa_TSEB_irrigated.time)
len(ETa_TSEB_agroforestry.time)
# len(y_pred_TSEB)

axs[0].plot(ETa_baseline_irrigated, 
            y_pred_baseline, 'b-', 
            label=f'Fit: $R^2 = {r2_baseline:.2f}$'
            )
axs[1].plot(ETa_TSEB_irrigated, 
            y_pred_TSEB, 'b-', 
            label=f'Fit: $R^2 = {r2_TSEB:.2f}$'
            )

for i in range(2):
    # Add a 1:1 line
    min_val = min(axs[i].get_xlim()[0], axs[i].get_ylim()[0])
    max_val = max(axs[i].get_xlim()[1], axs[i].get_ylim()[1])
    axs[i].plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 line')
    axs[i].set_aspect('equal')
    axs[i].legend()
    axs[i].set_xlabel(f'ETa {LCnames[0]} (mm/day)')
    axs[i].set_ylabel(f'ETa {LCnames[1]} (mm/day)')

# Set equal scaling on both axes
axs[0].set_title('Baseline')
axs[1].set_title('TSEB')

fig.savefig(f'{figPath}/11_MASK_mean_ETa_{LCnames[0]}_VS_{LCnames[1]}_{prj_name}.png',
            dpi=300
            )

#%% test ET map
fig, axs = plt.subplots(1,2,
                        sharey=True
                        )

for axi, lcn in zip(axs,LCnames):
  
    ETa_AOIi = df_fort777_select_t_xr[lcn + '_CLCmask']
    ETa_AOIi_datetimes = ds_analysis_EO.time.isel(time=0).values + ETa_AOIi.time.values

    ETa_AOIi_mean = ETa_AOIi.mean(dim=['x','y'])
    ETa_AOIi_mean = ETa_AOIi_mean.assign_coords(datetime=('time', ETa_AOIi_datetimes))

    
    ETa_AOIi_TSEB = ds_analysis_EO[lcn + '_CLCmask']
    ETa_AOIi_mean_TSEB = ETa_AOIi_TSEB.mean(dim=['x','y'])
    ETa_AOIi_mean_TSEB_interp = ETa_AOIi_mean_TSEB.interp(time=ETa_AOIi_mean['datetime'])
    
    x = ETa_AOIi_mean.values * 1000 * 86400
    y = ETa_AOIi_mean_TSEB_interp.values

    axi.scatter(x,y)

    # Perform linear regression using scipy.stats.linregress
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    y_pred = slope * x + intercept
    r2 = r_value**2  # Compute R^2 value

    axi.plot(x, y_pred, 'b-', label=f'Fit: $R^2 = {r2:.2f}$')

    # Add a 1:1 line
    min_val = min(axi.get_xlim()[0], axi.get_ylim()[0])
    max_val = max(axi.get_xlim()[1], axi.get_ylim()[1])
    axi.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 line')
    
    # Set equal scaling on both axes
    axi.set_aspect('equal')
    axi.set_title(lcn)
    # Optionally, add labels and a legend
    axi.set_xlabel('ETa Water-Balance (mm/day)')
    axi.set_ylabel('ETa Energy Balance (mm/day)')
    axi.legend()
fig.savefig(f'{figPath}/{lcn}_11_MASK_mean_ETa_hydro_ETa_Energy_{prj_name}.png',
            dpi=300
            )


#%%
fig, axs = plt.subplots(2,1,sharex=True)

for axi, lcn in zip(axs,LCnames):
   
    ETa_AOIi = df_fort777_select_t_xr[lcn + '_CLCmask']
    # ETa_AOIi_datetimes = ds_analysis_EO.time.isel(time=0).values + ETa_AOIi.time.values
    # ETa_AOIi_datetimes = ds_analysis_EO.time.values #.isel(time=0).values + ETa_AOIi.time.values
    ETa_AOIi_datetimes = ds_analysis_EO.time.values #.isel(time=0).values + ETa_AOIi.time.values
 
    ETa_AOIi_mean = ETa_AOIi.mean(dim=['x','y'])
    ETa_AOIi_mean['datetime'] = start_date + ETa_AOIi_mean.time
 
    
    ETa_AOIi_TSEB = ds_analysis_EO[lcn + '_CLCmask']
    ETa_AOIi_mean_TSEB = ETa_AOIi_TSEB.mean(dim=['x','y'])
    ETa_AOIi_mean_TSEB_interp = ETa_AOIi_mean_TSEB.interp(time=ETa_AOIi_mean['datetime'])
    
    axi.plot(ETa_AOIi_mean_TSEB.time,
            ETa_AOIi_mean_TSEB.values,
            linestyle='-.',
            color='r',
            label='TSEB'
            )
    
    axi.plot(ETa_AOIi_mean['datetime'],
            ETa_AOIi_mean.values*1000*86400,
            linestyle='-.',
            label='CATHY'
            )
    
    axi.set_ylabel('ETa (mm/day)')
    axi.set_title(lcn)
    
    axi.set_xlim([pd.to_datetime('01/01/2023'),
                  max(ETa_AOIi_mean['datetime'])
                  ]
                 )
    
    # axi.set_xlim([min(ETa_AOIi_mean['datetime']),
    #               max(ETa_AOIi_mean['datetime'])
    #               ]
    #              )
    axi.set_ylabel(r'$ET_{a}$ [mm/day]')
    axi.axvspan('2023-07-01', '2023-09-01',
                color='blue', 
                alpha=0.1,
                linestyle='--')

    axi.xaxis.set_major_locator(mdates.AutoDateLocator())  # Automatically adjust tick spacing
    axi.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    axi.tick_params(axis='x', rotation=45)  
    axi.set_ylim([0,8
                 ])

axi.set_xlabel('Date')
plt.legend([r'$ET_{a_{EB}}$', r'$ET_{a_{WB}}$'])
# axs[0].set_xlabel('')
fig.savefig(f'{figPath}/{lcn}_MASK_mean_ETa_hydro_ETa_Energy_{prj_name}.png',
            dpi=400
            )
    

#%% test ET map
# fig, ax = plt.subplots(1)
# hydro_Majadas.show('spatialET',
#                    ax=ax, 
#                    ti=10,
#                     clim=[0,5e-9],
#                    )
# plt.title('ET at tindex = 10')
# fig.savefig(figPath/'spatialET_Majadas.png',
#             dpi=300
#             )

#%% spatialET_Majadas animated

# fig, ax = plt.subplots(figsize=(12,6))
# ani, writer = ani, writer = utils.spatial_ET_animated(hydro_Majadas,fig,ax)
# ani.save(figPath/'spatialET_Majadas.gif', writer=writer)


#%% Compare ETa from TSEB with ETa from pyCATHY
# gdf_AOI_POI_Majadas = gdf_AOI_POI_Majadas.set_index('POI/AOI')

for key_cover in ['Intensive Irrigation','Tree-Grass','Agricutural fields','Lake']:
    fig, ax = plt.subplots()
    xPOI, yPOI =  gdf_AOI_POI_Majadas.set_index('POI/AOI').loc[key_cover].geometry.coords[0]
    Majadas_utilsutils.plot_time_serie_ETa_CATHY_TSEB(key_cover,
                                         df_fort777_select_t_xr,
                                         ds_analysis_EO,
                                         xPOI, yPOI,
                                         ax,
                                       )
    fig.savefig(f'{figPath}/{key_cover}ETa_hydro_ETa_Energy_Majadas.png',
                dpi=300
                )

#%%

