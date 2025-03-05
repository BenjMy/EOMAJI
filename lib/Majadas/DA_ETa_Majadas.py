"""
Majadas DA ETa
==================

"""

#%% Import libraries 

import os
import matplotlib.pyplot as plt
from pyCATHY.DA.cathy_DA import DA
from pyCATHY.DA import perturbate
from pyCATHY.DA.cathy_DA import dictObs_2pd
from pyCATHY.DA.observations import read_observations, prepare_observations, make_data_cov
from pyCATHY.plotters import cathy_plots as cplt
import pyCATHY.meshtools as mt
from pyCATHY.importers import cathy_outputs as out_CT

from pyCATHY.cathy_utils import change_x2date, MPa2m, kPa2m
from pyCATHY import cathy_tools

import numpy as np
import pandas as pd
import matplotlib as mpl
# set some default plotting parameters for nicer looking plots
mpl.rcParams.update({"axes.grid":True, 
                     "grid.color":"gray", 
                     "grid.linestyle":'--',
                     'figure.figsize':(10,10)}
                    )
import utils

from scenarii_Majadas_DA_ET import load_scenario 
import argparse
import matplotlib.dates as mdates
import Majadas_utils

import shutil
import glob
from pathlib import Path
import xarray as xr
import rioxarray as rxr
import geopandas as gpd

#%%

def get_cmd():
    parser = argparse.ArgumentParser(description='ProcessDA')
    # parser.add_argument('sc', type=int, help='scenario nb')
    #  #Feddes_irr, ref_scenario_irr_het_soil_f5, hetsoil_irr, Archie_Pert_scenarii, freq
    # -------------------------------------------------------------------------------------------------------
    parser.add_argument('-study','--study', type=str, 
                        help='study selection', 
                        required=False, 
                        # default='ET'
                        default='ZROOT'
                        )  
    parser.add_argument('-sc','--sc', type=int,
                        help='scenario nb', 
                        required=False, 
                        default=1
                        )
    parser.add_argument('-nens','--nens', type=int, 
                        help='nb of ensemble', 
                        required=False,
                        default=4
                        )
    parser.add_argument('-DAtype',type=str, 
                        help='type of DA',
                        default='enkf_Evensen2009'
                        # default='enkf_Evensen2009_Sakov'
                        )
    parser.add_argument('-DA_OBS_loc',type=int, 
                        help='DA observation COV localisation',
                        default=0
                        )
    parser.add_argument('-DA_loc_domain',type=str, 
                        help='DA domain localisation',
                        # default='nodes'
                        # default='zones'
                        default='veg_map'
                        # default=None
                        )
    parser.add_argument('-dataErr',type=float, 
                        help='error data',
                        default=1e-12
                        )
    parser.add_argument('-refModel',type=str, 
                        help='name of the ref model',
                        # default='MajadasBuffer_5000'                       
                        default='prj_name_Majadas_AOI_Buffer_100_WTD_2.0_short_0_SCF_1.0_OMEGA_1'                       
                        # default='Majadas_2024_WTD2'                       
                        )   
    parser.add_argument('-AOI', type=str, help='Area of Interest',
                        default='Buffer_100', 
                        # default='Buffer_5000', 
                        # default='H2_Bassin', 
                        required=False
                        ) 
    parser.add_argument('-short', type=int, 
                        help='select only one part of the dataset',
                        default=0, required=False) 
    parser.add_argument('-rainSeasonDA', type=int, 
                        help='select only one part of the dataset',
                        default=0, required=False) 
    args = parser.parse_args()

    return(args)

rootpath = Path(os.getcwd()).parent
prepoEOPath = Path('/run/media/z0272571a/SENET/iberia_daily/E030N006T6')

#%%
# ----------------------------------------------------------------------------
#  Build project name
# ----------------------------------------------------------------------------

args = get_cmd()

results_df, matching_index = utils.backup_simulog_DA(args,
                                                    filename='DA_ET_Majadas_log.csv'
                                                    )
prj_name_DA = 'DA_ET_Majadas' + str(matching_index)

#%%
expected_value = 1e-7  #expected mean value of ETa
percentage_error = args.dataErr 
absolute_error = expected_value * (percentage_error / 100)

#%% Import input data 
majadas_aoi = Majadas_utils.get_Majadas_aoi(buffer=5000)

file_pattern = '*ET-gf*.tif'
ET_0_filelist = list(prepoEOPath.glob(file_pattern))
crs_ET = rxr.open_rasterio(ET_0_filelist[0]).rio.crs

ds_analysis_EO = utils.read_prepo_EO_datasets(fieldsite='Majadas',
                                              AOI=args.AOI,
                                              crs=crs_ET
                                              )
ds_analysis_EO.rio.write_crs(majadas_aoi.crs)
ds_analysis_EO.rio.crs 

ds_analysis_EO['Elapsed_Time_s'] = (ds_analysis_EO.time - ds_analysis_EO.time[0]).dt.total_seconds()
# Check for NaN values in the ETp variable
nan_times = ds_analysis_EO['ETp'].isnull()
nan_times_per_time = nan_times.any(dim=('x', 'y'))
times_without_nan = ds_analysis_EO['time'].where(~nan_times_per_time, drop=True)
ds_analysis_EO = ds_analysis_EO.where(times_without_nan, drop=True)

if args.short==True:
    cutoffDate = ['01/01/2023','01/03/2024']
    start_time, end_time = pd.to_datetime(cutoffDate[0]), pd.to_datetime(cutoffDate[1])
    mask_time = (ds_analysis_EO.time >= start_time) & (ds_analysis_EO.time <= end_time)
    # Filter the DataArrays using the mask
    ds_analysis_EO = ds_analysis_EO.sel(time=mask_time)

#%% 
# ----------------------------------------------------------------------------
# Init CATHY model: copy reference model input files into DA folder
# ----------------------------------------------------------------------------

path2prjSol = rootpath / "WB_FieldModels"  # add your local path here
path2prj = rootpath / "WB_FieldModels/DA_ET"  # add your local path here

simu_ref = cathy_tools.CATHY(dirName=path2prjSol, 
                             prj_name=args.refModel
                             )
src_path = os.path.join(simu_ref.workdir,simu_ref.project_name)
dst_path = os.path.join(path2prj,prj_name_DA)

# Find .vtk files and output files to exclude
vtk_files = glob.glob(os.path.join(src_path, '**', '*.vtk'), recursive=True)
out_files = glob.glob(os.path.join(src_path, 'output', '**'), recursive=True)

# Combine both lists of files to exclude
exclude_files = vtk_files + out_files

# Define a function to ignore vtk and output files
def ignore_files(dir, files):
    excluded = []
    for file in files:
        # If the file's full path is in the exclude list, add it to the ignore list
        full_path = os.path.join(dir, file)
        if any(full_path.startswith(excl_file) for excl_file in exclude_files):
            excluded.append(file)
    return set(excluded)


shutil.copytree(src_path, dst_path, dirs_exist_ok=True, ignore=ignore_files)
simu = DA(dirName=path2prj, 
          prj_name=prj_name_DA
          )

#%% Get areas and point of interest 

gdf_AOI_POI_Majadas = Majadas_utils.get_AOI_POI_Majadas(crs_ET)
majadas_aoi = gpd.read_file('../data/AOI/majadas_aoi.geojson')
majadas_POIs, POIs_coords = Majadas_utils.get_Majadas_POIs()

is_point = gdf_AOI_POI_Majadas.geometry.geom_type == 'Point'
DEM, hd_DEM = simu_ref.read_inputs('dem')

#%% Get Nodes of interest (1 in each zone)

gdf_AOI_POI_Majadas['meshNode'] = None
for geom, indexPOI in zip(gdf_AOI_POI_Majadas[is_point].geometry,gdf_AOI_POI_Majadas[is_point].index):
    x, y = geom.coords[0]
    meshNodePOI, closest = simu_ref.find_nearest_node([x,y, np.max(DEM)])
    gdf_AOI_POI_Majadas.loc[indexPOI,'meshNode'] = meshNodePOI[0]

x, y =  gdf_AOI_POI_Majadas.iloc[4].geometry.coords[0]
# depths = [5,50,100,200]
depths = [5,100,400]

grid3d = simu_ref.read_outputs('grid3d')

meshNodes2plot = []
for d in depths:
    meshNodePOI, closest = simu_ref.find_nearest_node([x,y, np.max(DEM)-d/100])
    meshNodes2plot.append(meshNodePOI)
    
#%% actual ET 1D
dtcoupling = simu_ref.read_outputs('dtcoupling')

fig, ax = plt.subplots()
dtcoupling.plot(y='Atmact-vf', ax=ax, color='k', linestyle='--')
ax.set_xlabel('Time (s)')
ax.set_ylabel('ET (m)')
plt.tight_layout()

#%% Plot ATMBC
# --------------------
fig, ax = plt.subplots(figsize=(6,2))
simu_ref.show_input('atmbc',ax=ax)
plt.tight_layout()
fig.savefig(os.path.join(simu.workdir,simu.project_name,'atmbc.png'),
            dpi=350,
            bbox_inches='tight'
            )

#%% Plot spatial ATMBC
# --------------------
try:
    v_atmbc = simu_ref.atmbc['atmbc_df'].set_index('time').loc[simu_ref.atmbc['atmbc_df'].index[1]]
    v_atmbc_mat = np.reshape(v_atmbc,[21,21])
    fig, ax = plt.subplots(figsize=(4,4))
    img = ax.imshow(v_atmbc_mat)
    plt.colorbar(img)
    fig.savefig(os.path.join(simu.workdir,
                              simu.project_name,
                              'spatial_ETp.png')
                )
except:
    pass

#%% Initial conditions
# --------------------
df_psi = simu_ref.read_outputs('psi')
simu.run_preprocessor(verbose=True)
simu.run_processor(IPRT1=3)
# simu.update_ic(INDP=1,
#                 pressure_head_ini=list(df_psi.iloc[0])
#                 )

#%%
# ----------------------------------------------------------------------------
# Read DA scenario
# ----------------------------------------------------------------------------
root_map_nveg = len(np.unique(simu_ref.read_inputs('root_map')[0]))

scenarii = load_scenario(study=args.study,
                         nzones=root_map_nveg
                         )
scenario =  scenarii[list(scenarii)[args.sc]]


#%% perturbated variable 

_, FP = simu.read_inputs('soil')
simu.update_soil(PMIN=-1e+35)
simu.read_inputs('atmbc')
# ss
list_pert = perturbate.perturbate(simu, 
                                  scenario, 
                                  args.nens
                                  )   

#%% PAD ETa

np.shape(DEM)
ETa_meshnodes = Majadas_utils.xarraytoDEM_pad(ds_analysis_EO['ETa'])

nnod = simu_ref.read_outputs('grid3d')['nnod']
simu.grid3d
ETa_meshnodes

#%% Read observations = actual ET
data_measure = {}

simu_ref.atmbc
df_atmbc_ref = simu_ref.read_inputs('atmbc')
startDate = ETa_meshnodes.time.min()
print(startDate,ETa_meshnodes.time.max())


df_atmbc_ref_dates = [startDate.values + pd.Timedelta(atmbc_timei,unit='seconds') for atmbc_timei in df_atmbc_ref.time.unique()]
ETa_meshnodes_filterDates = ETa_meshnodes.sel(time=slice(np.vstack(df_atmbc_ref_dates).min(),
                                                         np.vstack(df_atmbc_ref_dates).max())
                                              )
# Define summer months
if args.rainSeasonDA==1:
    summer_months = [4, 5, 6, 7, 8, 9]
    non_summer_mask = ~ETa_meshnodes_filterDates['time.month'].isin(summer_months)
    ETa_meshnodes_filterDAperiod = ETa_meshnodes.where(non_summer_mask, drop=True)
    # elapsedTimeSec = (ETa_meshnodes_filterDAperiod.time - startDate).astype('timedelta64[s]').values
    elapsedTimeSec = (ETa_meshnodes_filterDAperiod.time - startDate).dt.total_seconds().values
else:
    elapsedTimeSec = (ETa_meshnodes_filterDates.time - startDate).dt.total_seconds().values
    ETa_meshnodes_filterDAperiod = ETa_meshnodes_filterDates

np.max(ETa_meshnodes_filterDAperiod)

for i, tobsi in enumerate(ETa_meshnodes_filterDAperiod.time):  
    for node_obsi in range(int(nnod)):
        data_measure = read_observations(
                                        data_measure,
                                        ETa_meshnodes_filterDAperiod.sel(time=tobsi).values.flatten()[node_obsi]*(1e-3/86400), 
                                        data_type = 'ETact', 
                                        data_err = absolute_error, # instrumental error
                                        show=True,
                                        tA=elapsedTimeSec[i],
                                        obs_cov_type='data_err', #data_err
                                        mesh_nodes=node_obsi, 
                                        datetime=pd.to_datetime(ETa_meshnodes_filterDAperiod.time[i].values),
                                        ) 
#%%
data_measure_df = dictObs_2pd(data_measure) 
data_measure_df.index
data_measure_df.iloc[0]

#%% Add covariance localisation
# ede

#%%
measurements = data_measure_df['data'].unstack(level='assimilation time').to_numpy()
errors2d = np.array([measurements]*int(simu.grid3d['nnod']))
nobs = len(data_measure_df['sensor_name'].unique())
stacked_data_cov = []
for i in range(len(data_measure_df.index.get_level_values(1).unique())):
    matrix = np.zeros([nobs,nobs])
    np.fill_diagonal(matrix, args.dataErr)
    stacked_data_cov.append(matrix)
np.shape(stacked_data_cov)

simu.stacked_data_cov = stacked_data_cov


#%% Parameters perturbation
# stop
var_per_dict_stacked = {}
for dp in list_pert:
    savefig = os.path.join(
                            simu.workdir,
                            simu.project_name,
                            simu.project_name + dp['savefig']
                            )
    np.random.seed(1)
    # need to call perturbate_var as many times as variable to perturbate
    # return a dict merging all variable perturbate to parse into prepare_DA
    var_per_dict_stacked = perturbate.perturbate_parm(
                                                    var_per_dict_stacked,
                                                    parm=dp, 
                                                    type_parm = dp['type_parm'], # can also be VAN GENUCHTEN PARAMETERS
                                                    mean =  dp['mean'],
                                                    sd =  dp['sd'],
                                                    sampling_type =  dp['sampling_type'],
                                                    ensemble_size =  dp['ensemble_size'], # size of the ensemble
                                                    per_type= dp['per_type'],
                                                    savefig=savefig
                                                    )
    
#%%
try:
    utils.plot_atmbc_pert(simu,var_per_dict_stacked)
except:
    pass

#%%
# import pyCATHY.DA.localisation as localisation
# localisation
# # Example usage
# mapped_grid = map_cells_to_nodes(simu.veg_map, (21,21))
# np.shape(mapped_grid)
# ss
# fig, ax = plt
# plt.imshow(mapped_grid)
# plt.imshow(simu.veg_map)
simu.update_zone()
simu.update_veg_map()

simu.read_inputs('zone')
simu_ref.read_inputs('root_map')
simu.read_inputs('root_map')

simu.read_inputs('atmbc')
simu.atmbc

# zone
#%% Run DA sequential

# DTMIN = 1e-2
# DELTAT = 1e-1

DTMIN = 1e-2
DELTAT = 1e3
DTMAX = 1e4

# stop
simu.run_DA_sequential(
                          VTKF=2,
                          TRAFLAG=0,
                          DTMIN=DTMIN,
                          DELTAT=DELTAT,
                          DTMAX=DTMAX,
                          parallel=True,    
                          dict_obs= data_measure,
                          list_assimilated_obs='all', # default
                          list_parm2update=scenarii[list(scenarii)[args.sc]]['listUpdateParm'],
                          DA_type=args.DAtype, #'pf_analysis', # default
                          dict_parm_pert=var_per_dict_stacked,
                          open_loop_run=False,
                          threshold_rejected=80,
                          localisation=args.DA_loc_domain,
                          # localisation='veg_map',
                          )

# args.parallel
