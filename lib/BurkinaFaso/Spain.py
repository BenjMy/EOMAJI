# Copyright (c) 2021 The Centum Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Centum project
#
"""

Tree-Grass ecosystem (Spain, Majadas de Tietar)
----------------------------------------------
This example demonstrates how to load and visualize datasets related to
evapotranspiration (ETa) and land cover classification (CLC) over the Majadas
region in Spain. The datasets are loaded using Pooch for remote file management.
Two visualizations are created: one showing land cover classification (CLC) and
another displaying the time series of evapotranspiration (ETa).


"""

import pooch
import xarray as xr 
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from pathlib import Path
from centum.delineation import ETAnalysis
import numpy as np
from centum import plotting as pltC
from centum import utils 
import geopandas as gpd


# rootPathData= Path('/home/ben/Nextcloud/BenCSIC/Codes/tech4agro/test_Majadas_centum_dataset/')
rootPathData= Path('/home/z0272571a@CAMPUS.CSIC.ES/Nextcloud/BenCSIC/Codes/Tech4agro_org/test_Majadas_centum_dataset/')
###############################################################################
# Step 1: Download datasets using Pooch
# Pooch is used to manage the dataset downloads from the specified GitHub repository.

# pooch_Majadas = pooch.create(
#     path=pooch.os_cache("Majadas_project"),
#     base_url="https://github.com/BenjMy/test_Majadas_centum_dataset/raw/refs/heads/main/",
#     registry={
#         "20200403_LEVEL2_ECMWF_TPday.tif": None,
#         "ETa_Majadas.netcdf": None,
#         "ETp_Majadas.netcdf": None,
#         "CLCover_Majadas.netcdf": None,
#     },
# )

# pooch_Majadas = pooch.create(
#     path=pooch.os_cache("Majadas_project"),
#     base_url="https://github.com/BenjMy/test_Majadas_centum_dataset/raw/refs/heads/main/",
#     registry={
#         "ETa_Majadas.netcdf": None,
#         "ETp_Majadas.netcdf": None,
#         "CLC_Majadas_clipped.shp": None,
#         "CLC_Majadas_clipped.shx": None,
#         "CLC_Majadas_clipped.dbf": None,
#     },
# )

# Majadas_ETa_dataset = pooch_Majadas.fetch('ETa_Majadas.netcdf')
# Majadas_ETp_dataset = pooch_Majadas.fetch('ETp_Majadas.netcdf')
# Majadas_CLC_dataset = pooch_Majadas.fetch('CLCover_Majadas.netcdf')

print('loading datasets')

Majadas_ETa_dataset = rootPathData/'ETa_Majadas_H2Bassin.netcdf'
Majadas_ETp_dataset = rootPathData/'ETp_Majadas_H2Bassin.netcdf'
Majadas_CLC_dataset = rootPathData/'CLCover_Majadas.netcdf'
Majadas_rain_dataset = rootPathData/'RAIN_Majadas_H2Bassin.netcdf'

Majadas_CLC_gdf = gpd.read_file(rootPathData/'BassinH2_Majadas_corrected.shp')

ETa_ds = xr.load_dataset(Majadas_ETa_dataset)
ETa_ds = ETa_ds.rename({"__xarray_dataarray_variable__": "ETa"})  # Rename the main variable to 'ETa'

ETp_ds = xr.load_dataset(Majadas_ETp_dataset)
ETp_ds = ETp_ds.rename({"__xarray_dataarray_variable__": "ETp"})  # Rename the main variable to 'ETa'

CLC = xr.load_dataset(Majadas_CLC_dataset)  # Load the CLC dataset

###############################################################################
# Step 2: Corine Land Cover Visualization

x_coords_CLC = CLC['Code_CLC'].coords['x'].values
y_coords_CLC = CLC['Code_CLC'].coords['y'].values

fig, ax = plt.subplots()
im1 = ax.imshow(CLC['Code_CLC'].values, cmap='viridis', aspect='auto', origin='upper',
                    extent=[x_coords_CLC.min(), x_coords_CLC.max(), y_coords_CLC.min(), y_coords_CLC.max()])
ax.set_title('CLC Code') 
ax.set_xlabel('X Coordinate') 
ax.set_ylabel('Y Coordinate') 
ax.axis('square')  #

plt.tight_layout()
plt.show()

###############################################################################
# Step 3: Create an animated visualization of the ETa time series

ETa_ds_selec = ETa_ds.isel(time=slice(0, 25))
x_coords = ETa_ds_selec['ETa'].coords['x'].values
y_coords = ETa_ds_selec['ETa'].coords['y'].values

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(ETa_ds_selec['ETa'].isel(band=0).isel(time=0).values, cmap='coolwarm', origin='upper',
               extent=[x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()])
ax.set_title('ETa Time Series')  # Title for the time series plot
ax.set_xlabel('X Coordinate')  # Label for the X axis
ax.set_ylabel('Y Coordinate')  # Label for the Y axis
ax.axis('square')  # Make the axis square for proper aspect ratio

cbar = fig.colorbar(im, ax=ax, orientation='vertical', extend='both', label='ETa')

def update(frame):
    im.set_data(ETa_ds_selec['ETa'].isel(band=0).isel(time=frame).values)
    ax.set_title(f'ETa Time Step: {frame}')
    return [im]

ani = FuncAnimation(fig, update, frames=len(ETa_ds_selec['time']), interval=200, blit=True)
HTML(ani.to_jshtml())  # Show the animation in the notebook

#%%
print('running delineation')
threshold_local = 0.25
threshold_regional = 0.25
time_window = 1

irr_analysis_usingET = ETAnalysis()


ET_analysis_ds = ETa_ds.isel(band=0)
ET_analysis_ds = ET_analysis_ds.sortby("time")
ET_analysis_ds['ETp'] = ETp_ds['ETp'].isel(band=0)

window_size_x = (ET_analysis_ds.x.max() - ET_analysis_ds.x.min())/10

# Run the irrigation delineation
decision_ds, event_type = irr_analysis_usingET.irrigation_delineation(
                                                                        ET_analysis_ds,
                                                                        threshold_local=threshold_local,
                                                                        threshold_regional=threshold_regional,
                                                                        time_window=time_window,
                                                                        window_size_x=window_size_x,
                                                                    )

# event_type

#%%
print('plotting mask')

from pyproj import CRS
crs = CRS.from_wkt('PROJCS["unknown",GEOGCS["WGS 84",DATUM["World Geodetic System 1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]]],PROJECTION["Azimuthal_Equidistant"],PARAMETER["latitude_of_center",53],PARAMETER["longitude_of_center",24],PARAMETER["false_easting",5837287.81977],PARAMETER["false_northing",2121415.69617],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH]]')

CLC = CLC.rio.write_crs(crs.to_wkt())
ETa_ds = ETa_ds.rio.write_crs(crs.to_wkt())
CLC_reproj = CLC.rio.reproject_match(ETa_ds)

Majadas_CLC_gdf_reproj = Majadas_CLC_gdf.to_crs(crs.to_wkt())


fig, ax = plt.subplots(1, 1, figsize=(10, 5))
CLC_reproj['Code_18'].plot.imshow(ax=ax, cmap='viridis')
ax.set_title("Majadas de Tietar Corine Land Cover")
plt.show()


mask_CLC_IrrLand = CLC_reproj['Code_CLC']==212
mask_CLC_Agroforestry = CLC_reproj['Code_CLC']==244

np.sum(mask_CLC_IrrLand)
np.sum(mask_CLC_Agroforestry)

clc_codes = utils.get_CLC_code_def()


fig, ax = plt.subplots(1, 1, figsize=(10, 5))
mask_CLC_IrrLand.plot.imshow(ax=ax, cmap='viridis')
ax.set_title("Majadas de Tietar Corine Land Cover")
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
mask_CLC_Agroforestry.plot.imshow(ax=ax, cmap='viridis')
ax.set_title("Majadas de Tietar Corine Land Cover")
plt.show()


#%%
print('plotting time serie')

start_date = "2023-08-01"
end_date = "2023-08-15"  # Changed to cover the entire summer
    

# decision_ds_filtered = decision_ds.sel(time=slice(start_date,end_date))
decision_ds_filtered = decision_ds


decision_ds_IrrLand = (
    decision_ds_filtered
    .where(mask_CLC_IrrLand)
    .mean(['x', 'y'])
)


decision_ds_Agroforestry = (
    decision_ds_filtered
    .where(mask_CLC_Agroforestry)
    .mean(['x', 'y'])
)

decision_ds_IrrLand.data_vars

fig, axs = plt.subplots(2,1)

decision_ds_IrrLand['ETa'].plot(x='time',label='ETa',ax=axs[0])
decision_ds_IrrLand['ratio_ETap_regional_spatial_avg_time_avg'].plot(x='time',
                                                                     label='ratio_ETap_regional',
                                                                     ax=axs[1])
decision_ds_IrrLand['ratio_ETap_local'].plot(x='time',
                                            label='ratio_ETap_local',
                                            ax=axs[1])

# decision_ds_IrrLand['condRain'].plot(x='time',
#                                         label='condRain',
#                                         ax=axs[1])
plt.legend()

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

decision_ds_filtered = decision_ds.sel(time=slice(start_date,end_date))
time_steps = decision_ds_filtered.time.size
ncols = 4
nrows = int(np.ceil(time_steps / ncols))  # Number of rows needed

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, 
                         figsize=(15, nrows * 3)
                         )
plot_decision_schedule(decision_ds_filtered['condRain'],
                       time_steps,
                       fig,axes)

for axi in axes.flatten():
    Majadas_CLC_gdf_reproj.plot(ax=axi, 
                                edgecolor="black", 
                                facecolor="none", 
                                # alpha=1,
                                )


# Majadas_CLC_gdf_reproj.plot()

#%%
        
event_type_sorted = event_type.sortby("time")
event_type_filtered = event_type_sorted.sel(time=slice(start_date,end_date))


event_type_IrrLand = (
    event_type_filtered
    .where(mask_CLC_IrrLand)
    # .mean(['x', 'y'])
)


event_type_Agroforestry = (
    event_type_filtered
    .where(mask_CLC_Agroforestry)
    # .mean(['x', 'y'])
)


np.sum(event_type)
np.sum(event_type_Agroforestry)

ncols = 4
time_steps = event_type_IrrLand.time.size
nrows = int(np.ceil(time_steps / ncols))  # Number of rows needed
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, 
                         figsize=(15, nrows * 3)
                         )
pltC.plot_irrigation_schedule(event_type_IrrLand,
                              time_steps,
                              fig,
                              axes
                              )


fig, axes = plt.subplots(nrows=nrows, ncols=ncols, 
                         figsize=(15, nrows * 3)
                         )
pltC.plot_irrigation_schedule(event_type_Agroforestry,
                              time_steps,
                              fig,
                              axes
                              )
    

    

# event_type_node_IN = (
#     event_type_filtered
#     .where(mask_CLCselected_land_covers{lci+1}"])
#     .mean(['x', 'y'])
# )

            