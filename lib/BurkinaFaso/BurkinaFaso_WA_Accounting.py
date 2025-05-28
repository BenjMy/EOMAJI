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
from centum import accounting
from pyCATHY import CATHY
from pyCATHY.importers import cathy_inputs as in_CT
from pyCATHY.plotters import cathy_plots as cplt

import os 

cwd = os.getcwd()
rootPathData= Path('/home/z0272571a@CAMPUS.CSIC.ES/Nextcloud/BenCSIC/Codes/Tech4agro_org/EOMAJI/data/prepro/BurkinaFaso/')
figPath= Path('/home/z0272571a@CAMPUS.CSIC.ES/Nextcloud/BenCSIC/Codes/Tech4agro_org/EOMAJI/figures/BurkinaFaso/')
rootPrepoPath = Path(cwd) / '../../data/prepro/' 

#%%
print('loading datasets TSEB MODEL')

BurkinaFaso_ETa_dataset = rootPathData/'ETa_BurkinaFaso.netcdf'
BurkinaFaso_ETp_dataset = rootPathData/'ETp_BurkinaFaso.netcdf'
ETa_ds = xr.load_dataset(BurkinaFaso_ETa_dataset)
ETp_ds = xr.load_dataset(BurkinaFaso_ETp_dataset)
ETp_ds.rio.crs
crs_ET = EOMAJI_utils.get_crs_ET_fromfile()
ETp_ds = ETp_ds.rio.write_crs(crs_ET)
# CLC = xr.load_dataset(BurkinaFaso_CLC_dataset)  # Load the CLC dataset

ds_analysis_EO, RAIN = EOMAJI_utils.read_prepo_EO_datasets(fieldsite='BurkinaFaso',
                                                           # AOI=args.AOI,
                                                           rootPath=rootPrepoPath/'BurkinaFaso',
                                                           crs="EPSG:27701"
                                                           )

ds_analysis_EO = ds_analysis_EO.rio.reproject("EPSG:4326")

#%%
print('loading datasets CATHY MODEL')

prjname = 'prj_name_BurkinaFaso_AOI__dayMax_5_WTD_2_SCF_1.0'

hydro_BurkinaFaso = CATHY(
                        dirName='../../WB_FieldModels/BurkinaFaso/',
                        prj_name=prjname
                      )
BurkinaFasoPath = Path(hydro_BurkinaFaso.workdir) / hydro_BurkinaFaso.project_name

ds_EO_resampled = xr.load_dataset(os.path.join(hydro_BurkinaFaso.workdir,
                                              hydro_BurkinaFaso.project_name,
                                              'ds_EO_resampled.netcdf'
                                              )
                                )
ds_EO_resampled_nodes_Rain = EOMAJI_utils.xarraytoDEM_pad(ds_EO_resampled['RAIN'])
ds_EO_resampled_nodes_ETp = EOMAJI_utils.xarraytoDEM_pad(ds_EO_resampled['ETp_fill0'])
ds_EO_resampled_nodes_validmask = EOMAJI_utils.xarraytoDEM_pad(ds_EO_resampled['valid_mask']).astype(bool)

# Read and prepare ET data
ds_ET_baseline = hydro_BurkinaFaso.read_outputs('ET')
ds_ET_baseline = ds_ET_baseline.drop_duplicates()
ds_ET_baseline = ds_ET_baseline.set_index(['time', 'X', 'Y']).to_xarray()
ds_ET_baseline['ACT. ETRA_mmday'] = ds_ET_baseline['ACT. ETRA'] * 1000 * 86400

# Convert ElapsedTime to datetime
start_date = np.datetime64('2018-01-01')
elapsed = ds_ET_baseline.time
new_times = start_date + elapsed.values
# Assign new time coordinate
ds_ET_baseline = ds_ET_baseline.assign_coords(time=("time", new_times))
ds_ET_baseline = ds_ET_baseline.swap_dims({"time": "time"})
ds_EO_resampled_nodes_validmask = ds_EO_resampled_nodes_validmask.rename({'x': 'X', 'y': 'Y'})
mask = ds_EO_resampled_nodes_validmask.broadcast_like(ds_ET_baseline['ACT. ETRA_mmday'])

# Apply the mask
ds_ET_baseline['ACT. ETRA_mmday'] = ds_ET_baseline['ACT. ETRA_mmday'].where(mask)
ds_ET_baseline['ACT. ETRA_mmday'].isel(time=1).plot.imshow()

#%%
hydro_BurkinaFaso.show_input('root_map')

#%%
import matplotlib.pyplot as plt
import imageio
import os
import numpy as np
import pandas as pd

os.makedirs("gif_frames", exist_ok=True)

data = ds_ET_baseline['ACT. ETRA_mmday']  # ETa data: dims (time, X, Y)
rain_avg = ds_EO_resampled['RAIN'].mean(dim=['x','y'], skipna=True)  # Your 1D Rain_avg time series
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
    # absolute_date = start_date + data.time[i].values  # timedelta64 + datetime64
    time_label = pd.to_datetime(data.time[i].values).strftime('%Y-%m-%d')
    ax1.set_title(f"ETa (mm/day) - Time: {time_label}")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    plt.colorbar(im, ax=ax1, label='mm/day')

    # 2) Rain_avg time series plot
    ax2.plot(rain_avg.time.values, rain_avg.values, label='Rain Avg')
    # Find index in rain_avg.time closest to current absolute_date
    idx = np.searchsorted(rain_avg.time.values, data.time[i].values)
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




#%% MONTHLY VOLUME

monthly_volume_ET_EO = accounting.compute_water_accounting(ds_analysis_EO)
total_volume_ET_EO = monthly_volume_ET_EO.sum(dim=['y', 'x'])


monthly_volume_ET_baseline = accounting.compute_water_accounting(ds_ET_baseline)
total_volume_ET_baseline = monthly_volume_ET_baseline.sum(dim=['y', 'x'])

fig, ax = plt.subplots()
total_volume_ET_EO.plot(ax=ax,marker='o',label='ET_EO')
total_volume_ET_baseline.plot(ax=ax,marker='o',label='ET_baseline')
plt.ylabel('ETa Volume (m³)')
plt.title('Total ETa Volume Over Time')
plt.grid(True)
plt.show()

#%%
# Select a time slice, e.g., January 2018
time_slice_ET_EO = monthly_volume_ET_EO.sel(time='2018-01')
time_slice_ET_baseline = monthly_volume_ET_baseline.sel(time='2018-01')


time_slice_ET_EO.plot(cmap='Blues')
plt.title('ETa Volume (m³) - January 2018')

plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.show()

#%%
seasonal_mean = monthly_volume.groupby('time.season').mean(dim='time')
seasonal_mean.sel(season='DJF').plot(cmap='OrRd')  # e.g., winter season
plt.title('Average ETa Volume (m³) - Summer (DJF)')
plt.show()

#%%
time_slice = monthly_volume.sel(time='2018-01')
plt.hist(time_slice.values.flatten(), bins=50, color='skyblue')
plt.xlabel('ETa Volume (m³)')
plt.ylabel('Pixel Count')
plt.title('Histogram of ETa Volume (January 2018)')
plt.show()

#%%

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

fig, ax = plt.subplots(figsize=(8,6))

# Prepare the data for the first frame
data0 = monthly_volume.isel(time=0)
vmin = monthly_volume.min().item()
vmax = monthly_volume.max().item()

# Plot the initial image without colorbar
im = ax.imshow(data0, cmap='Blues', vmin=vmin, vmax=vmax, origin='lower')

# Create colorbar once
cbar = fig.colorbar(im, ax=ax)
cbar.set_label('ETa Volume (m³)')

def update(frame):
    im.set_data(monthly_volume.isel(time=frame))
    current_time = monthly_volume.time.values[frame]
    ax.set_title(f'ETa Volume - {np.datetime_as_string(current_time, unit="M")}')
    return [im]

ani = animation.FuncAnimation(fig, update, frames=len(monthly_volume.time), blit=True, repeat=False)

ani.save('ETa_volume_animation.gif', writer='pillow', fps=2)

plt.close(fig)
