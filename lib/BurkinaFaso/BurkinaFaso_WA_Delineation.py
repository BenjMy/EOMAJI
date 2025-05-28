# Copyright (c) 2021 The Centum Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Centum project
#
"""

Burkina Faso
----------------------------------------------
EOMAJI WA
- Load prepro dataset (netcdf)

"""

import pooch
import xarray as xr
import rioxarray as rxr
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
from centum.delineation import ETAnalysis
import numpy as np
from centum import plotting as pltC
from centum import utils
import geopandas as gpd
from IPython.display import HTML
import sys
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

import EOMAJI_utils
import BurkinaFaso_utils
import os 

cwd = os.getcwd()
rootPathData= Path('/home/z0272571a@CAMPUS.CSIC.ES/Nextcloud/BenCSIC/Codes/Tech4agro_org/EOMAJI/data/prepro/BurkinaFaso/')
figPath= Path('/home/z0272571a@CAMPUS.CSIC.ES/Nextcloud/BenCSIC/Codes/Tech4agro_org/EOMAJI/figures/BurkinaFaso/')
rootPrepoPath = Path(cwd) / '../../data/prepro/' 

print('loading datasets')

BurkinaFaso_ETa_dataset = rootPathData/'ETa_BurkinaFaso.netcdf'
BurkinaFaso_ETp_dataset = rootPathData/'ETp_BurkinaFaso.netcdf'
# BurkinaFaso_CLC_dataset = rootPathData/'CLCover_BurkinaFaso.netcdf'
# BurkinaFaso_rain_dataset = rootPathData/'RAIN_BurkinaFaso.netcdf'
# BurkinaFaso_CLC_gdf = gpd.read_file(rootPathData/'BassinH2_BurkinaFaso_corrected.shp')

ETa_ds = xr.load_dataset(BurkinaFaso_ETa_dataset)
ETp_ds = xr.load_dataset(BurkinaFaso_ETp_dataset)
ETp_ds.rio.crs
crs_ET = EOMAJI_utils.get_crs_ET_fromfile()
ETp_ds = ETp_ds.rio.write_crs(crs_ET)
# CLC = xr.load_dataset(BurkinaFaso_CLC_dataset)  # Load the CLC dataset


ds_analysis_EO, RAIN = EOMAJI_utils.read_prepo_EO_datasets(fieldsite='BurkinaFaso',
                                                           # AOI=args.AOI,
                                                           rootPath=rootPrepoPath/'BurkinaFaso',
                                                           crs=crs_ET
                                                           )
_, rain_aligned = xr.align(ETa_ds, RAIN, 
                           join='inner')  # or 'left'/'right'/'outer'
rain_avg = rain_aligned['RAIN'].mean(dim=["x", "y"], skipna=True)

plt.figure(figsize=(12, 5))
plt.plot(rain_avg.time.values, rain_avg.values, lw=0.8)
plt.title("Rainfall Time Series")
plt.xlabel("Date")
plt.ylabel("Rainfall (units)")
plt.grid(True)
plt.tight_layout()
plt.show()




###############################################################################
# Step 3: Create an animated visualization of the ETa time series

# ETa_ds_selec = ETa_ds.isel(time=slice(0, 50))
ETa_ds_selec = ETa_ds
x_coords = ETa_ds_selec['ETa'].coords['x'].values
y_coords = ETa_ds_selec['ETa'].coords['y'].values


# fig, ax = plt.subplots(figsize=(8, 6))
# im = ax.imshow(ETa_ds_selec['ETa'].isel(time=0).values, 
#                cmap='coolwarm', 
#                origin='upper',
#                vmin=2,
#                vmax=8,
#                extent=[x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()])
# ax.set_title('ETa Time Series')  # Title for the time series plot
# ax.set_xlabel('X Coordinate')  # Label for the X axis
# ax.set_ylabel('Y Coordinate')  # Label for the Y axis
# ax.axis('square')  # Make the axis square for proper aspect ratio

# cbar = fig.colorbar(im, ax=ax, orientation='vertical', extend='both', label='ETa')

# def update(frame):
#     time = ETa_ds_selec.isel(time=frame).time.values
#     im.set_data(ETa_ds_selec['ETa'].isel(time=frame).values)
#     ax.set_title(f'ETa: {time}')
#     return [im]

# ani = FuncAnimation(fig, update, frames=len(ETa_ds_selec['time']), 
#                     interval=200,
#                     blit=True
#                     )
# # HTML(ani.to_jshtml())  # Show the animation in the notebook
# # ani.save('ETa_timeseries.gif', writer='pillow', dpi=150)
# ani.save(figPath/'ETa_timeseries.gif', writer='pillow', dpi=150)
#%%

CLC_BurkinaFaso_clipped = BurkinaFaso_utils.get_LandCoverMap()

CLC_root_depth = EOMAJI_utils.CLC_LookUpTable(rootPrepoPath/'../LandCover/BurkinaFaso_CGLOPS_LUT')

indexnoET = CLC_root_depth[CLC_root_depth['rootDepth']==1e-4].index.unique()
CLC_BurkinaFaso_clipped_resampled = CLC_BurkinaFaso_clipped.interp_like(ETa_ds.isel(time=0))


# CLC_root_depth = EOMAJI_utils.CLC_LookUpTable(rootPrepoPath/'../LandCover/BurkinaFaso_CGLOPS_LUT')
# use the lookup table CLC_root_depth to classified cultivated (index IGBP=12) against forest areas (columns with name containing forest)
# create mask for ETa_ds and plot 2d evolution of one land coover against the other taking the mean over each mask

# Identify cultivated class by IGBP code
cultivated_indices = CLC_root_depth[CLC_root_depth.index == 12].index

forest_indices = CLC_root_depth[
    CLC_root_depth['description'].str.contains('forest', case=False, na=False)
].index

# mask_cultivated = CLC_BurkinaFaso_clipped_resampled['landcover'].isin(cultivated_indices)
# mask_forest = CLC_BurkinaFaso_clipped_resampled.isin(forest_indices)

# mask_cultivated.plot.imshow()
# mask_forest['landcover'].plot.imshow()

#%%
# Mask areas with no ET
maskET = CLC_BurkinaFaso_clipped_resampled['landcover'].isin(indexnoET)
np.unique(CLC_BurkinaFaso_clipped_resampled['landcover'])
# maskET.plot.imshow()
# np.sum(maskET)
# maskET.sum().item()
# Apply the mask to the entire ETa_ds (over time) — set ETa to NaN where mask is True
# maskET = CLC_BurkinaFaso_clipped['landcover'] == 13

maskET = CLC_BurkinaFaso_clipped['landcover'].isin([0,1,11,13,15,16])
(CLC_BurkinaFaso_clipped['landcover'] == 13).plot.imshow()
ETa_ds_filter_urban = ETa_ds.where(~maskET)
# ET_analysis_ds.rio.crs

# CLC_BurkinaFaso_clipped.crs 

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(ETa_ds_filter_urban['ETa'].isel(time=0).values, 
               cmap='coolwarm', 
               origin='upper',
               vmin=2,
               vmax=8,
               extent=[x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()])
ax.set_title('ETa Time Series')  # Title for the time series plot
ax.set_xlabel('X Coordinate')  # Label for the X axis
ax.set_ylabel('Y Coordinate')  # Label for the Y axis
ax.axis('square')  # Make the axis square for proper aspect ratio
#%%

# cbar = fig.colorbar(im, ax=ax, orientation='vertical', extend='both', label='ETa')

# def update(frame):
#     time = ETa_ds_filter_urban.isel(time=frame).time.values
#     im.set_data(ETa_ds_filter_urban['ETa'].isel(time=frame).values)
#     ax.set_title(f'ETa: {time}')
#     return [im]

# ani = FuncAnimation(fig, update, frames=len(ETa_ds_filter_urban['time']), 
#                     interval=200,
#                     blit=True
#                     )
# # HTML(ani.to_jshtml())  # Show the animation in the notebook
# # ani.save('ETa_timeseries.gif', writer='pillow', dpi=150)
# ani.save(figPath/'ETa_timeseries_filterUrban.gif', writer='pillow', dpi=150)

#%%


# fig, ax = plt.subplots(figsize=(8, 6))
# im = ax.imshow(ETp_ds_selec['ETp'].isel(time=0).values, 
#                cmap='coolwarm', 
#                origin='upper',
#                vmin=2,
#                vmax=8,
#                extent=[x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()])
# ax.set_title('ETp Time Series')  # Title for the time series plot
# ax.set_xlabel('X Coordinate')  # Label for the X axis
# ax.set_ylabel('Y Coordinate')  # Label for the Y axis
# ax.axis('square')  # Make the axis square for proper aspect ratio

# cbar = fig.colorbar(im, ax=ax, 
#                     orientation='vertical', 
#                     extend='both', 
#                     label='ETp'
#                     )

# def update(frame):
#     time = ETp_ds_selec.isel(time=frame).time.values
#     im.set_data(ETp_ds_selec['ETp'].isel(time=frame).values)
#     ax.set_title(f'ETp: {time}')
#     return [im]

# ani = FuncAnimation(fig, update, 
#                     frames=len(ETa_ds_selec['time']), 
#                     interval=200,
#                     blit=True
#                     )
# # HTML(ani.to_jshtml())  # Show the animation in the notebook
# ani.save(figPath/'ETp_timeseries.gif', writer='pillow', dpi=150)

# ss
#%%
print('running delineation')
threshold_local = 0.25
threshold_regional = 0.25
time_window = 100

irr_analysis_usingET = ETAnalysis()


ET_analysis_ds = ETa_ds.copy()
ET_analysis_ds = ET_analysis_ds.sortby("time")
if 'spatial_ref' in ETp_ds.coords:
    ETp_ds = ETp_ds.drop_vars('spatial_ref')
ET_analysis_ds['ETp'] = ETp_ds['ETp'].sortby("time")


# window_size_x = (ET_analysis_ds.x.max() - ET_analysis_ds.x.min())/10
window_size_x = (ET_analysis_ds.x.max() - ET_analysis_ds.x.min()) /10
window_size_x = window_size_x.values
# Run the irrig

ET_analysis_ds = ET_analysis_ds.isel(time=slice(0, 365))

#%%

BurkinaFaso_Satelitte = rxr.open_rasterio(rootPrepoPath/"BurkinaFaso/satellite.tif")
BurkinaFaso_Satelitte = BurkinaFaso_Satelitte.rio.reproject("EPSG:27701")
# BurkinaFaso_Satelitte = BurkinaFaso_Satelitte.rio.reproject("EPSG:4326")
BurkinaFaso_Satelitte.plot.imshow()




#%%

BurkinaFaso_utils.plot_various_resolutions(ETa_ds_filter_urban,
                                           window_size_x,
                                           BurkinaFaso_Satelitte
                                           )
#%%
run_delineation = True
if run_delineation:
    (decision_ds, 
     event_type) = irr_analysis_usingET.irrigation_delineation(
                                            ET_analysis_ds,
                                            threshold_local=threshold_local,
                                            threshold_regional=threshold_regional,
                                            time_window=time_window,
                                            window_size_x=window_size_x,
                                        )
         
    decision_ds = decision_ds.rio.write_crs(crs_ET)
    event_type = event_type.rio.write_crs(crs_ET)
    
    decision_ds.to_netcdf(rootPrepoPath/'BurkinaFaso/WA_decision_ds')
    event_type.to_netcdf(rootPrepoPath/'BurkinaFaso/WA_event_type_ds')
    
    # event_type = event_type.rio.write_crs(crs_ET)
    
    for var in decision_ds.variables:
        if 'grid_mapping' in decision_ds[var].encoding:
            del decision_ds[var].encoding['grid_mapping']
            
    decision_ds.to_netcdf(rootPrepoPath/'BurkinaFaso/WA_decision_ds.netcdf')
    
    
    for var in decision_ds.variables:
        if 'grid_mapping' in decision_ds[var].encoding:
            del decision_ds[var].encoding['grid_mapping']
            
    event_type.to_netcdf(rootPrepoPath/'BurkinaFaso/WA_event_type_ds.netcdf')

else:
    # decision_ds = rxr.open_rasterio(rootPrepoPath/'BurkinaFaso/WA_decision_ds.netcdf')
    decision_ds = xr.open_dataset(rootPrepoPath/'BurkinaFaso/WA_decision_ds.netcdf')
    # event_type = rxr.open_rasterio(rootPrepoPath/'BurkinaFaso/WA_event_type_ds.netcdf')

#%%
# Create a list of point names and corresponding DataArray objects
points = {
    'forest_point': decision_ds.sel(x=2797301.1, y=6417911.7, method="nearest"),
    'crop_point_SE': decision_ds.sel(x=2796822.22, y=6415909.50, method="nearest"),
    'crop_point_N': decision_ds.sel(x=2795304.4, y=6418402.2, method="nearest"),
    'crop_point_SW': decision_ds.sel(x=2793281.29, y=6416048.54, method="nearest")
}
plt.figure(figsize=(10, 5))

# Plot ETa for both points
points['forest_point']['ETa'].plot(label='Forest', linestyle='-', marker='o')
points['crop_point_SE']['ETa'].plot(label='CropSE', linestyle='--', marker='s')
points['crop_point_N']['ETa'].plot(label='CropN', linestyle='--', marker='s')
points['crop_point_SW']['ETa'].plot(label='CropSW', linestyle='--', marker='s')

# Add labels and title
plt.title('ETa Time Series for Forest and Crop Points')
plt.xlabel('Time')
plt.ylabel('ETa (mm/day)')  # Update units if needed

# Add legend
plt.legend()

# Optional: Improve layout
plt.grid(True)
plt.tight_layout()
plt.show()


fig.savefig(figPath/'ETa_multiple_POI.png', 
            dpi=300
            )

#%%
ratio_ETap_local = points['crop_point_SW']['ratio_ETap_local_time_avg']
# ratio_ETap_local_diff = crop_point_SW['ratio_ETap_local_diff']
ratio_ETap_regional_spatial_avg =  points['crop_point_SW']['ratio_ETap_regional_spatial_avg_time_avg']
# ratio_ETap_local_spatial_avg = crop_point_SW['ratio_ETap_local']
# ratio_ETap_regional_spatial_avg = crop_point_SW['ratio_ETap_regional_spatial_avg']
condRain = points['crop_point_SW']['condRain']
condIrrigation = points['crop_point_SW']['condIrrigation']

import matplotlib.pyplot as plt

fig, (ax1, ax2, ax3) = plt.subplots(3,1,
                                    figsize=(8, 6),
                                    sharex=True)

ax1.bar(rain_avg.time.values, 
        rain_avg.values, width=1, 
        color='blue', edgecolor='blue')
ax1.set_title("Daily Average Rainfall")
ax1.set_xlabel("Date")
ax1.set_ylabel("Rainfall (mm/day)")
# fig.autofmt_xdate()
# plt.tight_layout()
# plt.show()

lnETp = points['crop_point_SW']['ETp'].plot(ax=ax2, label='ETp (CropSW)', 
                                linestyle='--', 
                                marker='s',
                                markersize=2,
                                color='tab:green')
ax2.set_ylabel('ETp (mm/day)', color='tab:blue')
ax2.tick_params(axis='y', labelcolor='tab:blue')

# Primary axis: ETa
ln1 = points['crop_point_SW']['ETa'].plot(ax=ax2,
                                          label='ETa (CropSW)', 
                                          linestyle='--', 
                                          marker='s', 
                                          markersize=2,
                                          color='tab:blue'
                                          )
ax2.set_ylabel('ET (a/p) (mm/day)', color='tab:blue')
ax2.tick_params(axis='y', labelcolor='tab:blue')

# Create secondary axis for ratios
ax2twin = ax2.twinx()
# ln2 = ratio_ETap_local.plot(ax=ax2, label='Local ETa/ETp', linestyle='-', marker='o', color='tab:orange')
# ln3 = ratio_ETap_local_diff.plot(ax=ax2, 
#                                  label='ETa/ETp local diff', linestyle='--',
#                                  marker='s', color='tab:green')
ln_Local_ratio_ETap = ratio_ETap_local.plot(ax=ax2twin, 
                                           label='Local ETa/ETp',
                                           linestyle='-',
                                           linewidth=2,
                                           # marker='^', 
                                           color='tab:purple')

ln4 = ratio_ETap_regional_spatial_avg.plot(ax=ax2twin, 
                                           label='Regional ETa/ETp',
                                           linestyle='--',
                                           linewidth=2,
                                           # marker='^', 
                                           color='tab:red')


ax2twin.set_ylabel('ETa / ETp Ratio', color='tab:gray')
ax2twin.tick_params(axis='y', labelcolor='tab:gray')
ax2twin.axhline(1.0, color='gray', linestyle='--', linewidth=0.8)  # Reference line

# Combine legends
lines = lnETp + ln1  + ln4 + ln_Local_ratio_ETap
labels = [l.get_label() for l in lines]
ax2.legend(lines, labels, loc='upper left')

ln_condRain = condRain.plot(ax=ax3, 
                            label='condRain',
                            # linestyle='--',
                            linewidth=5,
                            marker='.', 
                            color='tab:blue')

ln_condIrrigation = condIrrigation.plot(ax=ax3, 
                                        label='condIrrigation',
                                        # linestyle='--',
                                        linewidth=5,
                                        marker='.', 
                                        color='tab:red')

lines = ln_condRain + ln_condIrrigation
labels = [l.get_label() for l in lines]
ax3.legend(lines, labels, loc='upper left')

ax2.title('')
ax3.title('')


# Title and formatting
plt.title('ETa and ETa/ETp Ratios for Crop Point SW')
ax1.set_xlabel('Time')
ax1.grid(True)
plt.tight_layout()
plt.show()

fig.savefig(figPath/'ETaETp_SW.png', 
            dpi=300
            )


# forest_point.data_vars

#%%
# print('plotting mask')

# from pyproj import CRS
# crs = CRS.from_wkt('PROJCS["unknown",GEOGCS["WGS 84",DATUM["World Geodetic System 1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]]],PROJECTION["Azimuthal_Equidistant"],PARAMETER["latitude_of_center",53],PARAMETER["longitude_of_center",24],PARAMETER["false_easting",5837287.81977],PARAMETER["false_northing",2121415.69617],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH]]')

# CLC = CLC.rio.write_crs(crs.to_wkt())
# ETa_ds = ETa_ds.rio.write_crs(crs.to_wkt())
# CLC_reproj = CLC.rio.reproject_match(ETa_ds)

# Majadas_CLC_gdf_reproj = Majadas_CLC_gdf.to_crs(crs.to_wkt())


# fig, ax = plt.subplots(1, 1, figsize=(10, 5))
# CLC_reproj['Code_18'].plot.imshow(ax=ax, cmap='viridis')
# ax.set_title("Majadas de Tietar Corine Land Cover")
# plt.show()


# mask_CLC_IrrLand = CLC_reproj['Code_CLC']==212
# mask_CLC_Agroforestry = CLC_reproj['Code_CLC']==244

# np.sum(mask_CLC_IrrLand)
# np.sum(mask_CLC_Agroforestry)

# clc_codes = utils.get_CLC_code_def()


# fig, ax = plt.subplots(1, 1, figsize=(10, 5))
# mask_CLC_IrrLand.plot.imshow(ax=ax, cmap='viridis')
# ax.set_title("Majadas de Tietar Corine Land Cover")
# plt.show()

# fig, ax = plt.subplots(1, 1, figsize=(10, 5))
# mask_CLC_Agroforestry.plot.imshow(ax=ax, cmap='viridis')
# ax.set_title("Majadas de Tietar Corine Land Cover")
# plt.show()


#%%
# print('plotting time serie')

# start_date = "2023-08-01"
# end_date = "2023-08-15"  # Changed to cover the entire summer
    

# # decision_ds_filtered = decision_ds.sel(time=slice(start_date,end_date))
# decision_ds_filtered = decision_ds


# decision_ds_IrrLand = (
#     decision_ds_filtered
#     .where(mask_CLC_IrrLand)
#     .mean(['x', 'y'])
# )


# decision_ds_Agroforestry = (
#     decision_ds_filtered
#     .where(mask_CLC_Agroforestry)
#     .mean(['x', 'y'])
# )

# decision_ds_IrrLand.data_vars

# fig, axs = plt.subplots(2,1)

# decision_ds_IrrLand['ETa'].plot(x='time',label='ETa',ax=axs[0])
# decision_ds_IrrLand['ratio_ETap_regional_spatial_avg_time_avg'].plot(x='time',
#                                                                      label='ratio_ETap_regional',
#                                                                      ax=axs[1])
# decision_ds_IrrLand['ratio_ETap_local'].plot(x='time',
#                                             label='ratio_ETap_local',
#                                             ax=axs[1])

# # decision_ds_IrrLand['condRain'].plot(x='time',
# #                                         label='condRain',
# #                                         ax=axs[1])
# plt.legend()

#%%

def plot_decision_schedule(event_type,time_steps,fig,axes):
        
    axes = axes.flatten()  # Flatten to easily iterate over
    cmap = plt.cm.colors.ListedColormap([
                                         # 'white',
                                         'green', 
                                         'red'
                                         ])
    x_values = event_type['x'].values
    y_values = event_type['y'].values
    extent = [x_values.min(), x_values.max(), y_values.min(), y_values.max()]
    for i, ax in enumerate(axes):
        if i < time_steps:  # Only plot if there is corresponding data
            data = event_type.isel(time=i).values  # or event_type.sel(time=...) if using labels
            img = ax.imshow(data, 
                            cmap=cmap, 
                            vmin=0, 
                            vmax=1, 
                            extent=extent,
                            origin='lower'
                            )
            ax.set_xlabel('x')  # Label for the x-axis
            ax.set_ylabel('y')  # Label for the y-axis
        else:
            ax.axis('off')  # Turn off empty subplots
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, 
                                              norm=plt.Normalize(vmin=0, 
                                                                 vmax=1)
                                              ), 
                        ax=axes, 
                        orientation='horizontal', 
                        fraction=0.02, pad=0.04)  # Adjust placement
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['True', 'False'])
    
    pass

#%%
print('plotting decision rain')

# decision_ds_filtered = decision_ds.sel(time=slice(start_date,end_date))
time_steps = decision_ds.time.size
ncols = 4
nrows = int(np.ceil(time_steps / ncols))  # Number of rows needed

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, 
                         figsize=(15, nrows * 3)
                         )
plot_decision_schedule(decision_ds['condRain'],
                       time_steps,
                       fig,axes)

# for axi in axes.flatten():
#     Majadas_CLC_gdf_reproj.plot(ax=axi, 
#                                 edgecolor="black", 
#                                 facecolor="none", 
#                                 # alpha=1,
#                                 )


# Majadas_CLC_gdf_reproj.plot()

#%%
        
# event_type_sorted = event_type.sortby("time")
# event_type_filtered = event_type_sorted.sel(time=slice(start_date,end_date))


# event_type_IrrLand = (
#     event_type_filtered
#     .where(mask_CLC_IrrLand)
#     # .mean(['x', 'y'])
# )


# event_type_Agroforestry = (
#     event_type_filtered
#     .where(mask_CLC_Agroforestry)
#     # .mean(['x', 'y'])
# )


# np.sum(event_type)
# np.sum(event_type_Agroforestry)

# ncols = 4
# time_steps = event_type_IrrLand.time.size
# nrows = int(np.ceil(time_steps / ncols))  # Number of rows needed
# fig, axes = plt.subplots(nrows=nrows, ncols=ncols, 
#                          figsize=(15, nrows * 3)
#                          )
# pltC.plot_irrigation_schedule(event_type_IrrLand,
#                               time_steps,
#                               fig,
#                               axes
#                               )


# fig, axes = plt.subplots(nrows=nrows, ncols=ncols, 
#                          figsize=(15, nrows * 3)
#                          )
# pltC.plot_irrigation_schedule(event_type_Agroforestry,
#                               time_steps,
#                               fig,
#                               axes
#                               )
    

    

# # event_type_node_IN = (
# #     event_type_filtered
# #     .where(mask_CLCselected_land_covers{lci+1}"])
# #     .mean(['x', 'y'])
# # )


