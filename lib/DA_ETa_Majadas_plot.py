"""
Data assimilation of actual ET observation into hydrological model
Reference model is based on weill et al. with spatially heterogeneous ETp
python file to run weiletal_ET_ref_atmbc_spatially_variable
"""

#%% Import libraries 
import os
import numpy as np

from pyCATHY import CATHY
from pyCATHY.plotters import cathy_plots as pltCT
from pyCATHY.importers import cathy_outputs as out_CT
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyCATHY.plotters import cathy_plots as cplt

import matplotlib.dates as mdates
from pyCATHY.cathy_utils import change_x2date, MPa2m, kPa2m, clip_ET_withLandCover
from pyCATHY.DA.cathy_DA import dictObs_2pd
import pyCATHY.meshtools as mt

import utils
import argparse
from pathlib import Path
import rioxarray as rxr
import Majadas_utils
import geopandas as gpd

def get_cmd():
    parser = argparse.ArgumentParser(description='plot_results')
    parser.add_argument('-idsimu','--idsimu', 
                        type=int, 
                        help='study id',
                        required=False,
                        default=10
                        )  #default='ZROOTdim2') #default='ET_scenarii')
    parser.add_argument('-AOI', type=str, help='Area of Interest',
                        default='Buffer_100', 
                        # default='Buffer_5000', 
                        # default='H2_Bassin', 
                        required=False
                        ) 
    parser.add_argument('-short', type=int, help='select only one part of the dataset',
                        default=0, required=False) 
    args = parser.parse_args()

    return(args)

rootPath = Path(os.getcwd())


#%% Build project name
args = get_cmd()
#%%
plt.close('all')
cwd = os.getcwd()
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
# gg
if args.short==True:
    cutoffDate = ['01/01/2023','01/03/2024']
    start_time, end_time = pd.to_datetime(cutoffDate[0]), pd.to_datetime(cutoffDate[1])
    mask_time = (ds_analysis_EO.time >= start_time) & (ds_analysis_EO.time <= end_time)
    # Filter the DataArrays using the mask
    ds_analysis_EO = ds_analysis_EO.sel(time=mask_time)

idsimu=args.idsimu
start_date = pd.to_datetime(ds_analysis_EO.time.min().values)

#%% create CATHY objects

results_df = pd.read_csv('DA_ET_Majadas_log.csv',index_col=0)
simu2plot = results_df.loc[idsimu]
prj_name_ref = simu2plot['refModel']
prj_name_DA = 'DA_ET_Majadas' + str(idsimu)

path2prj_ref = "../WB_FieldModels/"  # add your local path here
simu_ref = CATHY(dirName=path2prj_ref, 
                 prj_name=prj_name_ref
                 )

path2prj = "../WB_FieldModels/DA_ET/"  # add your local path here
simu = CATHY(dirName=path2prj, 
             prj_name=prj_name_DA
             )
fig_path = rootPath/f'../figures/Majadas_DA/'
fig_path.mkdir(parents=True, exist_ok=True)

#%% Load simulation results
backupDA =  os.path.join(simu.workdir,prj_name_DA,prj_name_DA+ '_df.pkl')
results = simu.load_pickle_backup(backupDA)
results.keys()
dem, dem_hd= simu_ref.read_inputs('dem')
ET_DA = results['ET_DA_xr']
# ET_DA = results['df_performance']
df_performance = results['df_performance']


#%% Get areas and point of interest 

gdf_AOI_POI_Majadas = Majadas_utils.get_AOI_POI_Majadas(crs_ET)
majadas_aoi = gpd.read_file('../data/AOI/majadas_aoi.geojson')
majadas_POIs, POIs_coords = Majadas_utils.get_Majadas_POIs()

is_point = gdf_AOI_POI_Majadas.geometry.geom_type == 'Point'

DEM, hd_DEM = simu_ref.read_inputs('dem')


df_atmbc = simu_ref.read_inputs('atmbc')
df_atmbc.groupby('time').mean()

#%%

# import xarray as xr
# CLC_Majadas_clipped_grid = xr.open_dataset(f'../prepro/Majadas/{args.AOI}/CLCover_Majadas.netcdf',
#                                             # engine='scipy'
#                                            )
# (reprojected_CLC_Majadas,
#  mapped_data )=  Majadas_utils.get_Majadas_root_map_from_CLC(ET_DA,
#                                                             CLC_Majadas_clipped_grid,
#                                                             crs_ET
#                                                             )
# # reprojected_CLC_Majadas.Code_18
# # np.shape(mapped_data)       
                                     
#%% Get point of interest (1 per zone)

gdf_AOI_POI_Majadas['meshNode'] = None
gdf_AOI_POI_Majadas['meshNodeCoords'] = None
for geom, indexPOI in zip(gdf_AOI_POI_Majadas[is_point].geometry,
                          gdf_AOI_POI_Majadas[is_point].index
                          ):
    x, y = geom.coords[0]
    meshNodePOI, closest = simu_ref.find_nearest_node([x,y, np.max(DEM)])
    gdf_AOI_POI_Majadas.loc[indexPOI,'meshNode'] = meshNodePOI[0]
    # gdf_AOI_POI_Majadas.loc[indexPOI,'meshNodeCoords'] = closest[0]


# agroforestry


# Create a msk here instead
# x, y =  gdf_AOI_POI_Majadas.iloc[4].geometry.coords[0]
# x_irr, y_irr =  gdf_AOI_POI_Majadas[gdf_AOI_POI_Majadas['POI/AOI']=='Intensive Irrigation'].geometry.coords
# x_irr, y_irr =  gdf_AOI_POI_Majadas.iloc[2].geometry.coords[0]
# coordsInterest = [[x, y],[x_irr, y_irr]]
# depths = [5,50,100,200]
depths = [5,100,400]

grid3d = simu_ref.read_outputs('grid3d')
# grid3d['mesh3d_nodes'][meshNodes2plot]

# meshNodes2plot = []
# meshNodes2plot_pos = []
# for d in depths:
#     meshNodePOI, closest = simu_ref.find_nearest_node([x,y, np.max(DEM)-d/100])
#     meshNodes2plot.append(meshNodePOI)
#     meshNodes2plot_pos.append(closest[0])
    

meshNodes2plot_surfET = []
meshNodes2plot_pos_surfET = []

# for 
meshNodePOI, closest = simu_ref.find_nearest_node([x,y, np.max(DEM)])
meshNodes2plot_surfET.append(meshNodePOI[0])
meshNodes2plot_pos_surfET.append(closest[0])
    
    
#%% identify masked times 
# reload atmbc
df_atmbc = simu.read_inputs('atmbc')
atmbc_times = df_atmbc.time.unique()
nnod = len(df_atmbc)/len(atmbc_times)
ntimes = len(df_atmbc.time.unique())
nodenb = np.tile(np.arange(0,nnod),ntimes)
df_atmbc['nodeNb'] = nodenb
df_atmbc.set_index(['time','nodeNb'], inplace=True)


#%%

# ----------------------------------------------------------------------------
# Update CATHY inputs
# ----------------------------------------------------------------------------
simu.update_dem_parameters()
simu.update_prepo_inputs()
DEM, _ = simu.read_inputs('dem')
simu.update_dem(DEM)
simu.update_veg_map()
simu_ref.update_veg_map()


#%%  reload observations 
#--------------------------------
NENS = len(results['df_DA']['Ensemble_nb'].unique())
observations = dictObs_2pd(results['dict_obs'])
# startDate = observations['datetime'].min()
try:
    ETa_times_seconds = list(ET_DA.time_sec.isel(ensemble=0).isel(x=0).isel(y=0).values)
    assimilationDate = [(start_date + pd.Timedelta(tobsi, unit='seconds')).date() for tobsi in ETa_times_seconds]
    assimilationDate = pd.to_datetime(assimilationDate)
    assimilationDate =  pd.to_datetime(assimilationDate.date)
except:
    try:
        ETa_times_seconds = list(ET_DA.time_sec.isel(ensemble=0).values)
        assimilationDate = [(start_date + pd.Timedelta(tobsi, unit='seconds')) for tobsi in ETa_times_seconds]
        assimilationDate = np.hstack(assimilationDate)
        assimilationDate = pd.to_datetime(assimilationDate)
        assimilationDate =  pd.to_datetime(assimilationDate.date)
    except:
        raise ValueError
        

#%%
# observations.datetime[0].values

# datetime_obs_2substitute_tmp = [obsdatei.values for obsdatei in observations.datetime.to_numpy()]
# obs_dates = pd.to_datetime(datetime_obs_2substitute_tmp).unique()
# observations.datetime = datetime_obs_2substitute_tmp

obs_dates = pd.to_datetime(observations.datetime.unique().date)
non_match_mask = ~np.isin(assimilationDate, obs_dates)
cloud_dates = np.array(assimilationDate)[non_match_mask]
cloud_dates_indices = np.where(non_match_mask)[0]

#%%
CLC_of_interest = ['agroforestry','irrigated']
ET_DA_withMask = clip_ET_withLandCover(CLC_of_interest,
                                        gdf_AOI_POI_Majadas,
                                        ET_DA,
                                        ETname='ACT. ETRA',
                                        crs_ET=crs_ET,
                                        )
ET_DA_withMask['agroforestry_CLCmask']
ET_DA_withMask['assimilation'] = assimilationDate
ET_DA['assimilation'] = assimilationDate

fig, axs = plt.subplots(2, 1, sharex=True, 
                        figsize=(11, 6)
                        )

for axi, CLCi in zip(axs, CLC_of_interest):
    
    selected_data = ET_DA_withMask[CLCi + '_CLCmask'].isel(assimilation=0, ensemble=0)
    mask_positions = np.argwhere(~np.isnan(selected_data.values))
    mask_flat_positions = np.ravel_multi_index(mask_positions.T, selected_data.shape)
    mask_flat_positions_str = [f'ETact{mfp}' if mfp != 0 else 'ETact' for mfp in mask_flat_positions]
    # print(mask_flat_positions_str[-1])
    observations_mask = observations.loc[mask_flat_positions_str]
    observations_mask = observations_mask.loc[observations_mask.datetime.values<=ET_DA_withMask.assimilation.max().values]
    # observations_mask.set_index('datetime',inplace=True)
    observations_mask_mean = observations_mask[['data','datetime']].groupby(level=1).mean()
    # observations_mask_mean['datetime'] = observations_mask.datetime

    pltCT.DA_plot_ET_dynamic(
        ET_DA_withMask[CLCi+'_CLCmask'],
        observations=observations_mask_mean,
        ax=axi,
        unit="mm/day"
    )
    axi.set_title(CLCi)

    # Create a secondary y-axis
    axi2 = axi.twinx()
    axi2.bar(
        ET_DA.assimilation.values,
        np.hstack(
            df_atmbc.xs(0, level=1).values[:len(ET_DA.assimilation)] * (1e3 * 86400)
        ),
        color='tab:blue',
        alpha=0.2,
        label="net forcing",
    )


fig.tight_layout()
fig.savefig(fig_path/f'ID{idsimu}_time_serie_ETA_averaged_CLC.png', dpi=300)
plt.show()

#%%

#%%
fig, axs = plt.subplots(2, 1, sharex=True, figsize=(6, 6))

for axi, CLCi in zip(axs, CLC_of_interest):

    selected_data = ET_DA_withMask[CLCi + '_CLCmask'].isel(assimilation=0, ensemble=0)
    mask_positions = np.argwhere(~np.isnan(selected_data.values))
    mask_flat_positions = np.ravel_multi_index(mask_positions.T, selected_data.shape)
    mask_flat_positions_str = [f'ETact{mfp}' if mfp != 0 else 'ETact' for mfp in mask_flat_positions]
    observations_mask = observations.loc[mask_flat_positions_str]
    ET_DA_sel = ET_DA_withMask[CLCi+'_CLCmask'].isel(assimilation=non_match_mask==False)
    pltCT.DA_plot_ET_performance(ET_DA_sel,
                                # nodeposi=None,
                                # nodei,
                                observations = observations_mask,
                                axi = axi,
                                unit="mm/day"
                                )
    axi.legend('')
    axi.set_title(CLCi)

# Final adjustments
plt.tight_layout()
plt.show()
fig.savefig(fig_path/f'ID{idsimu}_1:1_ETA.png', dpi=300)

    




#%%

fig, axs = plt.subplots(2, 1, sharex=True, 
                        figsize=(11, 6)
                        )

# Loop over each node position
for axi, nodeposi, nodei in zip(axs, 
                      meshNodes2plot_pos_surfET, 
                      meshNodes2plot_surfET
                      ):
    
    # ET_DA['assimilation'] = observations.datetime.unique()[0:len(ET_DA.assimilation)]
    ET_DA['assimilation'] = assimilationDate
    # ET_DA['ACT. ETRA'] = ET_DA['ACT. ETRA']*(1e3*86400)

    # Plot dynamic ET on the primary axis
    pltCT.DA_plot_ET_dynamic(
        ET_DA,
        nodeposi,
        nodei,
        observations,
        axi,
        # unit="mm/day"
    )

    # Create a secondary y-axis
    axi2 = axi.twinx()
    axi2.bar(
        ET_DA.assimilation.values,
        np.hstack(
            df_atmbc.xs(0, level=1).values[:len(ET_DA.assimilation)] * (1e3*86400)
        ),
        color='tab:blue',
        alpha=0.7,
        label="net forcing",
    )

    axi2.set_ylim(-15, 10)  # Set desired limits before scaling
    axi2.spines['right'].set_position(('outward', 50))  # Shift right by 50 points
    axi2.tick_params(axis='y', which='both', direction='inout', 
                     length=6, 
                     width=2, 
                     labelsize=10)
    axi2.tick_params(axis='y', which='major', pad=20)  # Move tick labels away from the plot
    # axi2.set_yticks(axi2.get_yticks())  # Preserve original ticks
    # axi2.set_yticklabels([format_yticks(tick, factor=factor) for tick in axi2.get_yticks()])
    axi2.set_ylabel("Net Forcing (mm/day)", color='tab:blue')
    axi2.tick_params(axis='y', labelcolor='tab:blue')
   
    for cloudDatei in cloud_dates:
        axi.axvline(x=cloudDatei, 
                    color='k', 
                    linestyle='--', 
                    alpha=0.1,
                    )
    
axi2.legend(['obs','pred','forcing'])
fig.tight_layout()
fig.savefig(fig_path/f'ID{idsimu}_time_serie_ETA.png', dpi=300)
# Add a title and adjust layout
plt.show()

#%%
# fig, axs = plt.subplots(2, 1, sharex=True, figsize=(6, 6))

# for axi, nodeposi, nodei in zip(axs, 
#                                 meshNodes2plot_pos_surfET,
#                                 meshNodes2plot_surfET
#                                 ):

#     ET_DA_sel = ET_DA.isel(assimilation=non_match_mask==False)
   
#     pltCT.DA_plot_ET_performance(ET_DA_sel,
#                                 nodeposi,
#                                 nodei,
#                                 observations,
#                                 axi,
#                                 unit="mm/day"
#                                 )

# # Final adjustments
# plt.tight_layout()
# plt.show()
# fig.savefig(fig_path/f'ID{idsimu}_1:1_ETA.png', dpi=300)

    

#%%get assimilation times 
assimilationTimes = observations.xs('ETact', level=0).index.to_list()
obsAssimilated = observations.index.get_level_values(0).unique().values

#%% Plot performance
# -------------------------

# fig, ax = plt.subplots(2,figsize=[11,4])
# pltCT.DA_RMS(results['df_performance'],'ETact',ax=ax)
# fig.savefig(os.path.join(simu.workdir,simu.project_name,
#                          'df_performance.png'),
#             dpi=300,
#             bbox_inches='tight'
#             )

#%% Plot state dynamic
# -------------------------
# read psi and sw 
psi = simu_ref.read_outputs('psi')
sw, sw_times = simu_ref.read_outputs('sw')
tass = results['df_DA'].time.unique()


l_tda = len(results['df_DA'].time.unique())
nens = len(results['df_DA'].Ensemble_nb.unique())
grid3d = simu_ref.read_outputs('grid3d')
grid3d_nnod3 = grid3d['nnod3']


#%% Plot states_dyn sw 
fig, axs = plt.subplots(1, 
                        sharex=True,
                        sharey=True,
                        )
sw_datetimes = change_x2date(sw_times,start_date)
# axs= axs.ravel()
for i, nn in enumerate(meshNodes2plot_surfET):
    axs.plot(sw_datetimes[1:],
                sw.iloc[1:,nn],
                'r',
                label='ref',
                linestyle='--',
                marker='.'
                )
    pltCT.DA_plot_time_dynamic(results['df_DA'],
                                'sw',
                                nn,
                                savefig=False,
                                ax=axs,
                                start_date=start_date,
                                atmbc_times=atmbc_times
                                ) 
    axs.legend().remove()
    axs.set_ylabel('sw (-)')
    dd = simu.DEM.max() - np.round(simu.grid3d['mesh3d_nodes'][nn][0][2]*10)/10
    if i == len(axs):
        axs.set_xlabel('Assimiliation time')
    plt.legend(['solution','pred. ensemble'])
    plt.tight_layout()
fig.savefig(fig_path/f'ID{idsimu}_states_SW_dyn.png',
            dpi=300)

#%% Plot states_dyn psi  
# fig, axs = plt.subplots(2,2, 
#                         sharex=True,
#                         sharey=True,
#                        )
# sw_datetimes = change_x2date(sw_times,start_date)
# # results['df_DA']['psi_bef_update'] = np.log(-results['df_DA']['psi_bef_update'])

# axs= axs.ravel()
# for i, nn in enumerate(nodes2plots):
#     axs[i].plot(sw_datetimes[1:],psi.iloc[1:,nn],'r',label='ref',linestyle='--',marker='.')
#     pltCT.DA_plot_time_dynamic(results['df_DA'],
#                                 'psi',
#                                 nn,
#                                 savefig=False,
#                                 ax=axs[i],
#                                 start_date=start_date,
#                                 atmbc_times=atmbc_times
#                                 )  
#     # axs[i].set_yscale('log')
#     axs[i].legend().remove()
#     axs[i].set_ylabel(r'$\psi_{soil}$ (m)')
#     dd = simu.DEM.max() - np.round(simu.grid3d['mesh3d_nodes'][nn][0][2]*10)/10
#     if i == len(axs):
#         axs[i].set_xlabel('Assimiliation time')
#     # axs
#     plt.legend(['solution','pred. ensemble'])
#     plt.tight_layout()
# fig.savefig(fig_path/f'ID{idsimu}_states_dyn_psi.png',
#             dpi=300)
    
    
#%% Plot parameters dynamic
# -------------------------
veg_map, veg_map_hd = simu_ref.read_inputs('root_map')
df_SPP, df_FP = simu_ref.read_inputs('soil',MAXVEG=len(np.unique(veg_map)))
    
for kk in results['dict_parm_pert'].keys():
    # try:
        fig, ax = plt.subplots(figsize=[11,4])
        _, df = pltCT.DA_plot_parm_dynamic_scatter(parm = kk, 
                                            dict_parm_pert=results['dict_parm_pert'], 
                                            list_assimilation_times = assimilationTimes,
                                            ax=ax,
                                                  )
        zone_nb = int(''.join(filter(str.isdigit, kk)))
        nb_ass_times= len(results['dict_parm_pert'][kk]['ini_perturbation'])
        if 'ZROOT' in kk:
            ax.plot(np.arange(0,len(df.columns),1),[df_FP['ZROOT'].iloc[zone_nb]]*len(df.columns),
                    color='red',linestyle='--')
        fig.savefig(fig_path/f'ID{idsimu}_{kk}.png',dpi=300)

