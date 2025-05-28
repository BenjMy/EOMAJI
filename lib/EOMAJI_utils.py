#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 16:17:07 2024
"""
import numpy as np
import xarray as xr
import geopandas as gpd
import rioxarray as rxr
import pandas as pd
from shapely.geometry import Point
from centum import utils
# import utils
from pathlib import Path
from shapely.geometry import box
from shapely.geometry import mapping
from scipy import stats
from datetime import datetime
import matplotlib.pyplot as plt

#%%

def get_crs_ET_fromfile(path='/run/media/z0272571a/SENET/iberia_daily/E030N006T6/20190205_LEVEL4_300M_ET_0-gf.tif'):
    return rxr.open_rasterio(path).rio.crs

def get_crs_ET():
    crs_ET = '''CRS.from_wkt('PROJCS["Azimuthal_Equidistant",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]],PROJECTION["Azimuthal_Equidistant"],PARAMETER["latitude_of_center",8.5],PARAMETER["longitude_of_center",21.5],PARAMETER["false_easting",5621452.01998],PARAMETER["false_northing",5990638.42298],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH]]')'''
    return crs_ET

def read_prepo_EO_datasets(fieldsite='Majadas',
                           # AOI='Buffer_5000',
                           crs=None,
                           rootPath = Path('../../')
                           ):

    ETa_ds = xr.open_dataset(rootPath/f'ETa_{fieldsite}.netcdf',
                             )
    ETp_ds = xr.open_dataset(rootPath/f'ETp_{fieldsite}.netcdf',
                             # engine='scipy'
                             )
    RAIN_ds = xr.open_dataset(rootPath/f'RAIN_{fieldsite}.netcdf',
                              # engine='scipy'
                              )
    ds_analysis_EO = ETa_ds.drop_vars('spatial_ref', errors='ignore')
    ds_analysis_EO['ETp'] = ETp_ds.to_dataarray().sel(variable='ETp')
    # ds_analysis_EO['RAIN'] = RAIN_ds.to_dataarray().sel(variable='RAIN')

    # CLC_ds = CLC_ds.drop_vars('spatial_ref', errors='ignore')
    # ds_analysis_EO['CLC_code18'] = CLC_ds.Code_18
    ds_analysis_EO = ds_analysis_EO.sortby('time')
    ds_analysis_EO = ds_analysis_EO.rio.write_crs(crs)
    nulltimeETa = np.where(ds_analysis_EO.ETa.isnull().all())[0]
    valid_mask = ~ds_analysis_EO.time.isin(ds_analysis_EO.time[nulltimeETa])

    if len(nulltimeETa)>1:
        print('times with null ETa values!!')
    ds_analysis_EO = ds_analysis_EO.isel(time=valid_mask)

    # Determine the overlapping time range
    start_time = max(RAIN_ds.time.min(), ds_analysis_EO['ETp'].time.min())
    end_time = min(RAIN_ds.time.max(), ds_analysis_EO['ETp'].time.max())

    # Create a mask for the common time range
    # mask_time = (ds_analysis_EO['RAIN'].time >= start_time) & (ds_analysis_EO['RAIN'].time <= end_time)
    mask_time = (RAIN_ds.time >= start_time) & (RAIN_ds.time <= end_time)
    mask_time2 = (ds_analysis_EO['ETp'].time >= start_time) & (ds_analysis_EO['ETp'].time <= end_time)

    # Filter the DataArrays using the mask
    # ds_analysis_EO = ds_analysis_EO.sel(time=mask_time)
    ds_analysis_EO = ds_analysis_EO.sel(time=mask_time2)

    return ds_analysis_EO, RAIN_ds

def xarraytoDEM_pad(data_array, dims=['time', 'y', 'x']):
    # Extract the affine transform to get pixel sizes
    transform = data_array.rio.transform()

    pixel_size_x = transform.a
    pixel_size_y = -transform.e  # negative because y decreases in north-up coords

    # Number of coords along y and x
    ny = data_array.sizes[dims[1]]
    nx = data_array.sizes[dims[2]]

    # Define padding pixels
    pad_pixels_y = 1
    pad_pixels_x = 1

    # Pad the numpy array (same as you did)
    pad_width = ((0, 0), (pad_pixels_y, 0), (0, pad_pixels_x))  # (time, y, x)
    padded_array_np = np.pad(data_array.values,
                             pad_width,
                             mode='edge')

    # Build new coordinate arrays:
    # For y: generate from (first_coord - pad_pixels_y*pixel_size_y) up to (last_coord + pad_pixels_y*pixel_size_y)
    y_start = data_array[dims[1]].values[0] - pad_pixels_y * pixel_size_y
    y_end = data_array[dims[1]].values[-1]
    new_y = np.linspace(y_start, y_end, ny + pad_pixels_y)

    # For x: from first_coord to (last_coord + pad_pixels_x*pixel_size_x)
    x_start = data_array[dims[2]].values[0]
    x_end = data_array[dims[2]].values[-1] + pad_pixels_x * pixel_size_x
    new_x = np.linspace(x_start, x_end, nx + pad_pixels_x)

    # time coords unchanged
    time_coords = data_array[dims[0]].values

    padded_data_array = xr.DataArray(
        padded_array_np,
        dims=dims,
        coords={
            dims[0]: time_coords,
            dims[1]: new_y,
            dims[2]: new_x,
        },
        attrs=data_array.attrs
    )

    return padded_data_array

def clip_rioxarray(
                   fieldsite,
                   ET_filelist,
                   ET_0_filelist,
                   rain_filelist,
                   field_aoi,
                   prepoEOPath,
                   addtilename
                   ):



    for m in ET_filelist:
        etai = rxr.open_rasterio(m)
        # clipped_etai = etai.rio.clip_box(
        #                                   minx=majadas_aoi.bounds['minx'],
        #                                   miny=majadas_aoi.bounds['miny'],
        #                                   maxx=majadas_aoi.bounds['maxx'],
        #                                   maxy=majadas_aoi.bounds['maxy'],
        #                                   crs=majadas_aoi.crs,
        #                                 )

        clipped_etai = etai.rio.clip(
                                        field_aoi.geometry.values,
                                        crs=field_aoi.crs,
                                        )
        clipped_etai['time']=extract_filedate(m)
        
        tile_name = ''
        if addtilename==True:
            tile_name = [part for part in m.parts if part.startswith("X") and "_" in part][0]
        output_path = prepoEOPath/tile_name
        output_path.mkdir(parents=True, exist_ok=True)
        clipped_etai.rio.to_raster(output_path/m.name)


    for m in ET_0_filelist:
        etrefi = rxr.open_rasterio(m)
        clipped_etrefi = etrefi.rio.clip(
                                            field_aoi.geometry.values,
                                            crs=field_aoi.crs,
                                            )
        clipped_etrefi['time']=extract_filedate(m)
        
        tile_name = ''
        if addtilename==True:
            tile_name = [part for part in m.parts if part.startswith("X") and "_" in part][0]
        output_path = prepoEOPath/tile_name
        output_path.mkdir(parents=True, exist_ok=True)
        clipped_etrefi.rio.to_raster(output_path/m.name)

    for m in rain_filelist:
        raini = rxr.open_rasterio(m)
        clipped_raini = raini.rio.clip(
                                            field_aoi.geometry.values,
                                            crs=field_aoi.crs,
                                            )

        clipped_raini['time']=extract_filedate(m)
        clipped_raini.rio.to_raster(prepoEOPath/m.name)

    return clipped_etai, clipped_etrefi, clipped_raini


def extract_filedate(file_path):
    file_name = file_path.name
    date_str = file_name.split('_')[0]
    return datetime.strptime(date_str, '%Y%m%d')



def export_tif2netcdf(
                    pathTif2read='../../../prepro/Majadas/',
                    fieldsite='Majadas',                  
                    tile='',                  
                    ):

    file_pattern = '*ET-gf*.tif'
    ET_clipped_filelist = list(Path(pathTif2read/tile).glob(file_pattern))

    file_pattern = '*ET_0-gf*.tif'
    ET_0_clipped_filelist = list(Path(pathTif2read/tile).glob(file_pattern))

    file_pattern = '*TPday*.tif'
    rain_clipped_filelist = list(Path(pathTif2read).glob(file_pattern))

    ETa_l = []
    ETa_dates = []
    for m in ET_clipped_filelist:
        ETafi = rxr.open_rasterio(m)
        ETafi['time']=extract_filedate(m)
        ETa_l.append(ETafi)
        ETa_dates.append(ETafi['time'])


    ETp_l = []
    ETp_dates = []
    for m in ET_0_clipped_filelist:
        ETpfi = rxr.open_rasterio(m)
        ETpfi['time']=extract_filedate(m)
        ETp_l.append(ETpfi)
        ETp_dates.append(ETpfi['time'])

    rain_l = []
    rain_dates = []
    for m in rain_clipped_filelist:
        rainfi = rxr.open_rasterio(m)
        rainfi['time']=extract_filedate(m)
        rain_l.append(rainfi)
        rain_dates.append(rainfi['time'])
    
    ETp = xr.concat(ETp_l,dim='time')
    ETa = xr.concat(ETa_l,dim='time')
    RAIN = xr.concat(rain_l,dim='time')

    return ETp, ETa, RAIN

    # ETp.to_netcdf(f'../../prepro/{fieldsite}/ETp_{fieldsite}.netcdf')
    # RAIN.to_netcdf(f'../../prepro/{fieldsite}/RAIN_{fieldsite}.netcdf')
    # ETa.to_netcdf(f'../../prepro/{fieldsite}/ETa_{fieldsite}.netcdf')
    

from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import Rbf

def interpolate_vars_on_surface_nodes_rbf(ds, mesh3d_nodes, 
                                          surface_node_indices=None
                                          ):
    """
    Interpolate EO variables on the surface nodes of a mesh using RBF with extrapolation.
    NaNs in source data are skipped.

    Parameters:
    -----------
    ds : xarray.Dataset
        The EO dataset with shape (time, y, x) and coords `x`, `y`, `time`.

    mesh3d_nodes : np.ndarray
        Array of shape (n_nodes, 3) with [x, y, z] node positions.

    surface_node_indices : array-like or None
        Indices of surface nodes. If None, selects highest z nodes.

    Returns:
    --------
    ds_surface : xarray.Dataset
        Interpolated dataset on surface nodes.
    """
    if surface_node_indices is None:
        z_vals = mesh3d_nodes[:, 2]
        surface_node_indices = np.where(z_vals >= np.max(z_vals))[0]

    surface_nodes = mesh3d_nodes[surface_node_indices]
    x_target = surface_nodes[:, 0]
    y_target = surface_nodes[:, 1]

    times = ds.time.values
    x_coords = ds.x.values
    y_coords = ds.y.values
    X, Y = np.meshgrid(x_coords, y_coords)

    vars_to_interp = ['ETa', 'ETp', 'RAIN']
    data_vars = {}

    for varname in vars_to_interp:
        if varname not in ds:
            continue

        interpolated = np.full((len(times), len(surface_node_indices)), np.nan)

        for i, t in enumerate(times):
            data_2d = ds[varname].sel(time=t).values
            X_flat = X.flatten()
            Y_flat = Y.flatten()
            Z_flat = data_2d.flatten()

            # Mask NaNs
            mask = ~np.isnan(Z_flat)
            if not np.any(mask):
                continue

            try:
                rbf = Rbf(X_flat[mask], Y_flat[mask], Z_flat[mask], function='linear')
                interpolated[i, :] = rbf(x_target, y_target)
            except Exception as e:
                print(f"Skipping time {t} due to error: {e}")
                continue

        data_vars[f"{varname}_surface_nodes"] = (("time", "surface_node"), interpolated)

    ds_surface = xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": times,
            "surface_node": surface_node_indices,
            "x_surface": ("surface_node", x_target),
            "y_surface": ("surface_node", y_target),
            "z_surface": ("surface_node", surface_nodes[:, 2]),
        }
    )

    if 'ETp_surface_nodes' in ds_surface and 'RAIN_surface_nodes' in ds_surface:
        rain = ds_surface['RAIN_surface_nodes']
        etp = ds_surface['ETp_surface_nodes']
        net_atmbc = (rain - etp) * (1e-3 / 86400)
        ds_surface['net_atmbc'] = net_atmbc.fillna(0)

    return ds_surface

# import xarray as xr
# import numpy as np
# import pandas as pd
# from scipy.interpolate import Rbf
import numpy as np
import xarray as xr
import pandas as pd
from scipy.interpolate import griddata

def interpolate_vars_on_surface_nodes_vectorized(ds: xr.Dataset,
                                                  mesh3d_nodes: np.ndarray,
                                                  var_names=["ETa", "ETp", "RAIN"]
                                                  ) -> xr.Dataset:
    # Step 1: Identify surface nodes
    df_nodes = pd.DataFrame(mesh3d_nodes, columns=["x", "y", "z"])
    df_surface = df_nodes.sort_values("z", ascending=False).drop_duplicates(subset=["x", "y"])
    surface_xy = df_surface[["x", "y"]].values
    surface_z = df_surface["z"].values
    surface_idx = np.arange(len(surface_xy))

    # Get coordinates
    x_coords = ds.x.values
    y_coords = ds.y.values
    time_coords = ds.time.values

    # Create 2D grid of input coordinates
    xx, yy = np.meshgrid(x_coords, y_coords)
    points_grid = np.column_stack((xx.ravel(), yy.ravel()))

    interpolated_vars = {}

    for var_name in var_names:
        print(var_name)
        if var_name not in ds:
            raise ValueError(f"Variable '{var_name}' not found in dataset.")

        var = ds[var_name].values  # shape (time, y, x)
        time_steps, ny, nx = var.shape
        var = var.reshape(time_steps, -1)  # shape (time, ny*nx)

        # Identify valid points based on the first timestep
        valid_mask = ~np.isnan(var[0])
        interp_result = np.full((time_steps, len(surface_xy)), np.nan)

        if np.any(valid_mask):
            valid_points = points_grid[valid_mask]
            for t in range(time_steps):
                z_valid = var[t, valid_mask]
                interp_vals = griddata(
                    valid_points,
                    z_valid,
                    surface_xy,
                    method='linear',
                    fill_value=np.nan
                )
                interp_result[t] = interp_vals

        # Create DataArray
        data_array = xr.DataArray(
            interp_result,
            dims=("time", "surface_node"),
            coords={"time": time_coords, "surface_node": surface_idx},
            name=var_name + "_surface_nodes"
        )
        interpolated_vars[data_array.name] = data_array

    # Final dataset
    ds_surface = xr.Dataset(interpolated_vars)
    ds_surface = ds_surface.assign_coords({
        "x_surface": ("surface_node", surface_xy[:, 0]),
        "y_surface": ("surface_node", surface_xy[:, 1]),
        "z_surface": ("surface_node", surface_z),
    })

    return ds_surface

from joblib import Parallel, delayed
from scipy.interpolate import Rbf
import numpy as np
import pandas as pd
import xarray as xr


def interpolate_one_timestep(x_coords, y_coords, data_2d, surface_xy):
    # Mask NaNs
    valid_mask = ~np.isnan(data_2d)
    if not np.any(valid_mask):
        return np.full(len(surface_xy), np.nan)

    xx, yy = np.meshgrid(x_coords, y_coords)
    x_valid = xx[valid_mask].ravel()
    y_valid = yy[valid_mask].ravel()
    z_valid = data_2d[valid_mask].ravel()

    try:
        rbf = Rbf(x_valid, y_valid, z_valid, function='linear')
        return rbf(surface_xy[:, 0], surface_xy[:, 1])
    except Exception:
        return np.full(len(surface_xy), np.nan)


def interpolate_vars_on_surface_nodes(ds: xr.Dataset,
                                      mesh3d_nodes: np.ndarray,
                                      var_names=["ETa", "ETp", "RAIN"],
                                      n_jobs=-1  # Use all cores by default
                                      ) -> xr.Dataset:
   
    
    df_nodes = pd.DataFrame(mesh3d_nodes, columns=["x", "y", "z"])
    df_surface = df_nodes.sort_values("z", ascending=False).drop_duplicates(subset=["x", "y"])
    surface_xy = df_surface[["x", "y"]].values
    surface_z = df_surface["z"].values
    surface_idx = np.arange(len(surface_xy))

    x_coords = ds.x.values
    y_coords = ds.y.values
    time_coords = ds.time.values

    interpolated_vars = {}

    for var_name in var_names:
        print(f"Interpolating {var_name}...")

        if var_name not in ds:
            raise ValueError(f"Variable '{var_name}' not found in dataset.")

        var = ds[var_name]

        # Parallel processing
        results = Parallel(n_jobs=n_jobs)(
            delayed(interpolate_one_timestep)(x_coords, 
                                              y_coords, 
                                              var.isel(time=t).values,
                                              surface_xy
                                              )
            for t in range(var.shape[0])
        )

        data_array = xr.DataArray(
            np.array(results),
            dims=("time", "surface_node"),
            coords={"time": time_coords, "surface_node": surface_idx},
            name=var_name + "_surface_nodes"
        )
        interpolated_vars[data_array.name] = data_array

    ds_surface = xr.Dataset(interpolated_vars)
    ds_surface = ds_surface.assign_coords({
        "x_surface": ("surface_node", surface_xy[:, 0]),
        "y_surface": ("surface_node", surface_xy[:, 1]),
        "z_surface": ("surface_node", surface_z),
    })

    return ds_surface


# def interpolate_vars_on_surface_nodes(ds: xr.Dataset, 
#                                       mesh3d_nodes: np.ndarray, 
#                                       var_names=["ETa", "ETp", "RAIN"]
#                                       ) -> xr.Dataset:
#     """
#     Interpolates selected variables from an xarray.Dataset to surface mesh nodes using Rbf (scattered interpolation).

#     Parameters
#     ----------
#     ds : xr.Dataset
#         Dataset with shape (time, y, x) and coordinate variables x and y.
#     mesh3d_nodes : np.ndarray
#         Array of shape (n_nodes, 3) containing x, y, z coordinates.
#     var_names : list of str
#         List of variable names to interpolate from the dataset.

#     Returns
#     -------
#     xr.Dataset
#         Dataset with interpolated variables at surface nodes: dimensions (time, surface_node).
#     """

#     # Step 1: Identify surface nodes (highest z at each (x, y))
#     df_nodes = pd.DataFrame(mesh3d_nodes, columns=["x", "y", "z"])
#     df_surface = df_nodes.sort_values("z", ascending=False).drop_duplicates(subset=["x", "y"])
#     surface_xy = df_surface[["x", "y"]].values
#     surface_z = df_surface["z"].values
#     surface_idx = np.arange(len(surface_xy))

#     x_coords = ds.x.values
#     y_coords = ds.y.values
#     time_coords = ds.time.values

#     interpolated_vars = {}

#     for var_name in var_names:
#         print(var_name)
#         if var_name not in ds:
#             raise ValueError(f"Variable '{var_name}' not found in dataset.")

#         var = ds[var_name]
#         var_interp = []

#         for t in range(var.shape[0]):
#             print(t)
#             data_2d = var.isel(time=t).values

#             # Mask NaNs
#             valid_mask = ~np.isnan(data_2d)
#             if not np.any(valid_mask):
#                 interp_vals = np.full(len(surface_xy), np.nan)
#             else:
#                 xx, yy = np.meshgrid(x_coords, y_coords)
#                 x_valid = xx[valid_mask].ravel()
#                 y_valid = yy[valid_mask].ravel()
#                 z_valid = data_2d[valid_mask].ravel()

#                 # Fit RBF interpolator
#                 try:
#                     rbf = Rbf(x_valid, y_valid, z_valid, function='linear')  # or 'multiquadric', 'inverse', etc.
#                     interp_vals = rbf(surface_xy[:, 0], surface_xy[:, 1])
#                 except Exception:
#                     interp_vals = np.full(len(surface_xy), np.nan)

#             var_interp.append(interp_vals)

#         data_array = xr.DataArray(
#             np.array(var_interp),
#             dims=("time", "surface_node"),
#             coords={"time": time_coords, "surface_node": surface_idx},
#             name=var_name + "_surface_nodes"
#         )
#         interpolated_vars[data_array.name] = data_array

#     # Step 3: Create the output dataset
#     ds_surface = xr.Dataset(interpolated_vars)
#     ds_surface = ds_surface.assign_coords({
#         "x_surface": ("surface_node", surface_xy[:, 0]),
#         "y_surface": ("surface_node", surface_xy[:, 1]),
#         "z_surface": ("surface_node", surface_z),
#     })

#     return ds_surface


# def interpolate_vars_on_surface_nodes(ds: xr.Dataset, mesh3d_nodes: np.ndarray, var_names=["ETa", "ETp", "RAIN"]) -> xr.Dataset:
#     """
#     Interpolates selected variables from an xarray.Dataset to surface mesh nodes using RegularGridInterpolator.

#     Parameters
#     ----------
#     ds : xr.Dataset
#         Dataset with shape (time, y, x) and coordinate variables x and y.
#     mesh3d_nodes : np.ndarray
#         Array of shape (n_nodes, 3) containing x, y, z coordinates.
#     var_names : list of str
#         List of variable names to interpolate from the dataset.

#     Returns
#     -------
#     xr.Dataset
#         Dataset with interpolated variables at surface nodes: dimensions (time, surface_node).
#     """

#     # Step 1: Identify surface nodes (highest z at each (x, y))
#     df_nodes = pd.DataFrame(mesh3d_nodes, columns=["x", "y", "z"])
#     df_surface = df_nodes.sort_values("z", ascending=False).drop_duplicates(subset=["x", "y"])
#     surface_xy = df_surface[["x", "y"]].values
#     surface_z = df_surface["z"].values
#     surface_idx = np.arange(len(surface_xy))

#     x_coords = ds.x.values
#     y_coords = ds.y.values
#     time_coords = ds.time.values

#     interpolated_vars = {}

#     for var_name in var_names:
#         if var_name not in ds:
#             raise ValueError(f"Variable '{var_name}' not found in dataset.")

#         var = ds[var_name]
#         var_interp = []

#         for t in range(var.shape[0]):
#             data_2d = var.isel(time=t).values

#             # Fill NaNs using nearest-neighbor interpolation in 2D
#             if np.isnan(data_2d).any():
#                 data_filled = fillna_nearest_2d(data_2d)
#             else:
#                 data_filled = data_2d

#             f = RegularGridInterpolator(
#                 (y_coords, x_coords),
#                 data_filled,
#                 bounds_error=False,
#                 fill_value=np.nan  # Outside original extent will still be NaN
#             )

#             interp_vals = f(surface_xy[:, [1, 0]])  # (y, x) order
#             var_interp.append(interp_vals)

#         data_array = xr.DataArray(
#             np.array(var_interp),
#             dims=("time", "surface_node"),
#             coords={"time": time_coords, "surface_node": surface_idx},
#             name=var_name + "_surface_nodes"
#         )
#         interpolated_vars[data_array.name] = data_array

#     # Step 3: Create the output dataset
#     ds_surface = xr.Dataset(interpolated_vars)
#     ds_surface = ds_surface.assign_coords({
#         "x_surface": ("surface_node", surface_xy[:, 0]),
#         "y_surface": ("surface_node", surface_xy[:, 1]),
#         "z_surface": ("surface_node", surface_z),
#     })

#     return ds_surface


def fillna_nearest_2d(arr):
    """Fill NaNs in 2D array using nearest-neighbor method."""
    from scipy.ndimage import distance_transform_edt

    mask = np.isnan(arr)
    if not np.any(mask):
        return arr

    # Get indices of nearest non-NaN values
    idx = np.indices(arr.shape)
    dist, (inds_y, inds_x) = distance_transform_edt(mask, return_indices=True)
    return arr[inds_y, inds_x]



# def interpolate_vars_on_surface_nodes(ds: xr.Dataset, mesh3d_nodes: np.ndarray, var_names=["ETa", "ETp", "RAIN"]) -> xr.Dataset:
#     """
#     Interpolates selected variables from an xarray.Dataset to surface mesh nodes.

#     Parameters
#     ----------
#     ds : xr.Dataset
#         Dataset with shape (time, y, x) and coordinate variables x and y.
#     mesh3d_nodes : np.ndarray
#         Array of shape (n_nodes, 3) containing x, y, z coordinates.
#     var_names : list of str
#         List of variable names to interpolate from the dataset.

#     Returns
#     -------
#     xr.Dataset
#         Dataset with interpolated variables at surface nodes: dimensions (time, surface_node).
#     """
#     # Step 1: Identify surface nodes (highest z at each (x, y))
#     df_nodes = pd.DataFrame(mesh3d_nodes, columns=["x", "y", "z"])
#     df_surface = df_nodes.sort_values("z", ascending=False).drop_duplicates(subset=["x", "y"])
#     surface_xy = df_surface[["x", "y"]].values
#     surface_z = df_surface["z"].values
#     surface_idx = np.arange(len(surface_xy))

#     # Step 2: Sort x and y coordinates
#     x_coords = ds.x.values
#     y_coords = ds.y.values
#     time_coords = ds.time.values

#     x_sort_idx = np.argsort(x_coords)
#     y_sort_idx = np.argsort(y_coords)
#     x_sorted = x_coords[x_sort_idx]
#     y_sorted = y_coords[y_sort_idx]

#     interpolated_vars = {}

#     # Step 3: Interpolate each selected variable
#     for var_name in var_names:
#         if var_name not in ds:
#             raise ValueError(f"Variable '{var_name}' not found in dataset.")

#         var = ds[var_name]
#         var_interp = []

#         for t in range(var.shape[0]):
#             data_2d = var.isel(time=t).values

#             # Sort data_2d to match sorted x and y
#             data_sorted = data_2d[np.ix_(y_sort_idx, x_sort_idx)]

#             # Create interpolator with sorted axes
#             f = RegularGridInterpolator(
#                 (y_sorted, x_sorted), 
#                 data_sorted,
#                 bounds_error=False,
#                 fill_value=None
#             )

#             # Interpolate at surface (y, x)
#             interp_vals = f(surface_xy[:, [1, 0]])  # order: (y, x)
#             var_interp.append(interp_vals)

#         data_array = xr.DataArray(
#             np.array(var_interp),  # shape: (time, surface_node)
#             dims=("time", "surface_node"),
#             coords={"time": time_coords, "surface_node": surface_idx},
#             name=var_name + "_surface_nodes"
#         )

#         interpolated_vars[data_array.name] = data_array

#     # Step 4: Create the output dataset
#     ds_surface = xr.Dataset(interpolated_vars)
#     ds_surface = ds_surface.assign_coords({
#         "x_surface": ("surface_node", surface_xy[:, 0]),
#         "y_surface": ("surface_node", surface_xy[:, 1]),
#         "z_surface": ("surface_node", surface_z),
#     })

#     return ds_surface


# def interpolate_vars_on_surface_nodes(ds: xr.Dataset, mesh3d_nodes: np.ndarray, var_names=["ETa", "ETp", "RAIN"]) -> xr.Dataset:
#     """
#     Interpolates selected variables from an xarray.Dataset to surface mesh nodes.

#     Parameters
#     ----------
#     ds : xr.Dataset
#         Dataset with shape (time, y, x) and coordinate variables x and y.
#     mesh3d_nodes : np.ndarray
#         Array of shape (n_nodes, 3) containing x, y, z coordinates.
#     var_names : list of str
#         List of variable names to interpolate from the dataset.

#     Returns
#     -------
#     xr.Dataset
#         Dataset with interpolated variables at surface nodes: dimensions (time, surface_node).
#     """
#     # Step 1: Identify surface nodes (highest z at each (x, y))
#     df_nodes = pd.DataFrame(mesh3d_nodes, columns=["x", "y", "z"])
#     df_surface = df_nodes.sort_values("z", ascending=False).drop_duplicates(subset=["x", "y"])
#     surface_xy = df_surface[["x", "y"]].values
#     surface_z = df_surface["z"].values
#     surface_idx = np.arange(len(surface_xy))

#     x_coords = ds.x.values
#     y_coords = ds.y.values
#     time_coords = ds.time.values

#     interpolated_vars = {}

#     # Step 2: Interpolate each selected variable
#     for var_name in var_names:
#         if var_name not in ds:
#             raise ValueError(f"Variable '{var_name}' not found in dataset.")

#         var = ds[var_name]
#         var_interp = []

#         for t in range(var.shape[0]):
#             data_2d = var.isel(time=t).values
#             f = RegularGridInterpolator((y_coords, x_coords), 
#                                         data_2d, 
#                                         bounds_error=False, 
#                                         fill_value=None
#                                         )
#             interp_vals = f(surface_xy[:, [1, 0]])  # (y, x) order
#             var_interp.append(interp_vals)

#         data_array = xr.DataArray(
#             np.array(var_interp),  # shape: (time, surface_node)
#             dims=("time", "surface_node"),
#             coords={"time": time_coords, "surface_node": surface_idx},
#             name=var_name + "_surface_nodes"
#         )

#         interpolated_vars[data_array.name] = data_array

#     # Step 3: Create the output dataset
#     ds_surface = xr.Dataset(interpolated_vars)
#     ds_surface = ds_surface.assign_coords({
#         "x_surface": ("surface_node", surface_xy[:, 0]),
#         "y_surface": ("surface_node", surface_xy[:, 1]),
#         "z_surface": ("surface_node", surface_z),
#     })

#     return ds_surface




def center_to_node_grid_all_vars(ds: xr.Dataset) -> xr.Dataset:
    """
    Convert all cell-centered variables in a structured rioxarray.Dataset
    to a mesh-node-based grid using linear interpolation.
    
    Parameters
    ----------
    ds : xr.Dataset
        Input dataset with coordinates 'x' and 'y' representing cell centers.
    
    Returns
    -------
    xr.Dataset
        New dataset with all variables interpolated to node-based coordinates.
    """
    # Extract center coordinates
    x_center = ds.x.values
    y_center = ds.y.values

    # Compute uniform spacing
    dx = np.diff(x_center).mean()
    dy = np.diff(y_center).mean()

    # Create node coordinates
    x_nodes = np.concatenate(([x_center[0] - dx / 2], x_center + dx / 2))
    y_nodes = np.concatenate(([y_center[0] - dy / 2], y_center + dy / 2))

    # Initialize new dataset
    ds_nodes = xr.Dataset(coords={
        "x_node": x_nodes,
        "y_node": y_nodes,
        "time": ds.coords.get("time", None)
    })

    # Interpolate each data variable with x and y dims
    for var in ds.data_vars:
        dims = ds[var].dims
        if 'x' in dims and 'y' in dims:
            # Perform interpolation
            interpolated = ds[var].interp(x=x_nodes, y=y_nodes, method="linear")
            # Rename dimensions to node coordinates
            interpolated = interpolated.rename({'x': 'x_node', 'y': 'y_node'})
            ds_nodes[var + "_nodes"] = interpolated

    return ds_nodes



import numpy as np
from scipy.ndimage import binary_dilation
from scipy.ndimage import binary_erosion
from scipy.ndimage import gaussian_filter

def raise_boundary_elevation(DEM: np.ndarray, 
                             mask: np.ndarray, 
                             boundary_raise: float = 1.0) -> np.ndarray:
    """
    Raise elevation values at the boundary cells of the valid mask area.
    Boundary cells are valid cells adjacent to at least one invalid cell.

    Parameters:
    - DEM: 2D array of elevations
    - mask: boolean 2D array, True where data is valid
    - boundary_raise: amount to add to boundary cells' elevation

    Returns:
    - DEMnew: copy of DEM with raised boundary elevations
    """

    DEMnew = DEM.copy()

    # Dilate mask: True where neighbor is valid or invalid
    dilated_mask = binary_dilation(mask)

    # Boundary = valid cells that have at least one invalid neighbor
    boundary_mask = mask & (~dilated_mask | ~mask)
    # Above line is a bit complex, better do:
    # boundary_mask = mask & (~binary_erosion(mask))

    # Actually, simpler and more precise:
    from scipy.ndimage import binary_erosion
    eroded_mask = binary_erosion(mask)
    boundary_mask = mask & (~eroded_mask)  # valid cells that disappear after erosion are boundaries

    # Raise DEM elevation at boundary cells
    DEMnew[boundary_mask] += boundary_raise

    return DEMnew




def find_boundary_outlet(mask: np.ndarray) -> tuple[int, int]:
    """
    Find an arbitrary outlet cell located at the boundary of the valid mask.
    The outlet is any valid cell adjacent to an invalid cell.

    Parameters:
    - mask: 2D boolean array, True where data is valid

    Returns:
    - outlet_coord: tuple (row, col) of outlet cell
    """

    eroded_mask = binary_erosion(mask)
    boundary_mask = mask & (~eroded_mask)
    boundary_indices = np.argwhere(boundary_mask)

    if boundary_indices.size == 0:
        raise ValueError("No boundary cells found in the mask.")

    return tuple(boundary_indices[0])  # first boundary cell found


def masked_gaussian_filter(data: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """
    Smooth data with Gaussian filter ignoring invalid data outside the mask.

    Parameters:
    - data: 2D array with data (np.nan for invalid)
    - mask: boolean array, True for valid data
    - sigma: Gaussian sigma

    Returns:
    - Smoothed data with same shape as input, nodata cells remain unchanged
    """
    data_filled = np.where(mask, data, 1e-5)
    weight = mask.astype(float)

    smooth_data = gaussian_filter(data_filled, sigma=sigma)
    smooth_weight = gaussian_filter(weight, sigma=sigma)

    with np.errstate(invalid='ignore', divide='ignore'):
        smoothed = smooth_data / smooth_weight
    smoothed[~mask] = np.nan  # keep invalid as nan

    return smoothed

def create_gently_sloped_dem(
    DEM: np.ndarray,
    nodata_value: float = -9999,
    base_elevation: float = 1e-1,
    slope_magnitude: float = 1e-5,
    boundary_raise: float = 1e-3,
    smooth_sigma: float = 2
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    
    

    mask = DEM != nodata_value
    
    DEMraisedb = raise_boundary_elevation(DEM, 
                                          mask, 
                                          boundary_raise=1e-5
                                          ) 
    
    if not np.any(mask):
        raise ValueError("DEM has no valid data cells.")

    DEMnew = np.full_like(DEMraisedb, np.nan, dtype=float)
    DEMnew[mask] = base_elevation

    # Raise boundary valid cells
    from scipy.ndimage import binary_erosion
    eroded_mask = binary_erosion(mask)
    boundary_mask = mask & (~eroded_mask)
    DEMnew[boundary_mask] += boundary_raise

    outlet_coord = find_boundary_outlet(mask)
    DEMnew[outlet_coord] = DEMnew[mask].min() - slope_magnitude * 500

    y_idx, x_idx = np.indices(DEM.shape)
    dist = np.sqrt((y_idx - outlet_coord[0])**2 + (x_idx - outlet_coord[1])**2)
    dist_masked = dist[mask]
    dist_norm = (dist - dist_masked.min()) / np.ptp(dist_masked)

    DEMnew[mask] -= slope_magnitude * dist_norm[mask]

    # Smooth with mask-aware gaussian filter
    DEM_smoothed = masked_gaussian_filter(DEMnew, mask, sigma=smooth_sigma)
    DEMnew[mask] = DEM_smoothed[mask]

    DEMnew[outlet_coord] = DEMnew[mask].min() - slope_magnitude * 10

    fig, ax = plt.subplots()
    im = ax.imshow(DEMnew, 
                   cmap="terrain",
                   vmin=DEMnew[mask].min(),
                   vmax=DEMnew[mask].max()
                   )
    ax.scatter(outlet_coord[1], outlet_coord[0], color="red", label="Outlet")
    fig.colorbar(im, ax=ax, label="Elevation")
    ax.legend()
    ax.set_title("Gently Sloped DEM with Unique Outlet")

    return DEMnew, mask, outlet_coord


def CLC_LookUpTable(path2file):
    
    CLC_lookup = pd.read_csv(path2file,
                             # delim_whitespace=True
                             delimiter='\t'
                             ).set_index('IGBP')

    def assign_root_depth(desc):
        if 'forest' in desc.lower():
            return 3
        elif 'herbaceous' in desc.lower():
            return 0.5
        elif 'herbaceous' in desc.lower():
            return 0.5
        elif 'sparse vegetation' in desc.lower():
            return 0.5
        elif 'Cultivated' in desc.lower():
            return 0.5
        else:
            return 1e-4

    CLC_lookup['rootDepth'] = CLC_lookup['description'].apply(assign_root_depth)
    
    return CLC_lookup
    

    
