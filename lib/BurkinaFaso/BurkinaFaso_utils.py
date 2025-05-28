#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 16:17:07 2024
"""
import geopandas as gpd
import rioxarray as rxr
import xarray as xr
# import utils
from pathlib import Path


#%%

    
def get_AOI_POI_BurkinaFaso(AOI,crs_ET):
    BurkinaFaso_aoi = gpd.read_file(f'../../data/AOI/{AOI}.geojson')
    BurkinaFaso_aoi_reproj = BurkinaFaso_aoi.to_crs(crs_ET)
    return BurkinaFaso_aoi_reproj


def get_LandCoverMap(crs=None):
    if crs is None:
        crs = get_crs_ET()

    tilesCLC = ['BurkinaFaso_X0028_Y0029_IGBP', 'BurkinaFaso_X0027_Y0029_IGBP']
    pathCLC = Path('/home/z0272571a@CAMPUS.CSIC.ES/Nextcloud/BenCSIC/Codes/Tech4agro_org/EOMAJI/data/LandCover')
    
    CLC = []
    for tile in tilesCLC:
        da = rxr.open_rasterio(pathCLC / f"{tile}.tif", masked=True)
        # Convert DataArray to Dataset with a proper name
        ds = da.to_dataset(name="landcover")
        CLC.append(ds)

    # Now you can safely merge the Datasets
    CLC_BurkinaFaso = xr.merge(CLC)
    BurkinaFaso_aoi_reproj = get_AOI_POI_BurkinaFaso('burkina_faso_aoi',crs)
        
    CLC_BurkinaFaso_clipped = CLC_BurkinaFaso.rio.clip(
                                                BurkinaFaso_aoi_reproj.geometry.values,
                                                BurkinaFaso_aoi_reproj.crs,
                                                drop=True,
                                                invert=False
                                            )
                                                
    
    return CLC_BurkinaFaso_clipped.isel(band=0)

def get_crs_ET(path='/run/media/z0272571a/SENET/iberia_daily/E030N006T6/20190205_LEVEL4_300M_ET_0-gf.tif'):
    return rxr.open_rasterio(path).rio.crs
    

def get_BurkinaFaso_root_map_from_CLC():
    pass


def clip_ET_withLandCover():
    pass


import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_various_resolutions(ds,window_size_x,
                             BurkinaFaso_Satelitte
                            ):
    
    # Extraer el dato principal
    eta = ds['ETa'].isel(time=200)
    
    # Coordenadas del punto de interés
    crop_point = ds.sel(x=2793281.29, y=6416048.54, method="nearest")
    x_point = crop_point.x.values
    y_point = crop_point.y.values
    
    # Tamaño de la ventana de zoom en metros
    half_window = window_size_x /2 
    

    # Crear figura con tres subplots
    fig, (ax_main, ax_zoom, ax_zoom2) = plt.subplots(1, 3, figsize=(14, 6), constrained_layout=True)
    
    # Mapa completo (subplot principal)
    im = eta.plot.imshow(ax=ax_main, vmin=2, vmax=5, cmap="viridis", add_colorbar=False)
    ax_main.plot(x_point, y_point, 'ro', markersize=6, label='crop_point_SW')
    
    # Recuadro de zoom
    rect = patches.Rectangle((x_point - half_window, y_point - half_window),
                             window_size_x, window_size_x,
                             linewidth=1.5, edgecolor='red', facecolor='none')
    ax_main.add_patch(rect)
    ax_main.set_title("ETa (full map)")
    ax_main.set_aspect('equal')
    
    # Zoom regional
    eta.plot.imshow(ax=ax_zoom,
                    vmin=2, 
                    vmax=5, 
                    cmap="viridis", 
                    alpha=1,
                    add_colorbar=False
                    )
    ax_zoom.set_xlim(x_point - half_window, x_point + half_window)
    ax_zoom.set_ylim(y_point - half_window, y_point + half_window)
    ax_zoom.plot(x_point, y_point, 'ro', markersize=4)
    ax_zoom.set_title(f"Zoom ±{int(half_window)}m")
    BurkinaFaso_Satelitte.plot.imshow(ax=ax_zoom,alpha=0.6)
    ax_zoom.set_aspect('equal')
    
    # Zoom al píxel
    divideby= 15
    eta.plot.imshow(ax=ax_zoom2, vmin=2, vmax=5, cmap="viridis", add_colorbar=False)
    ax_zoom2.set_xlim(x_point - half_window/divideby, x_point + half_window/divideby)
    ax_zoom2.set_ylim(y_point - half_window/divideby, y_point + half_window/divideby)
    ax_zoom2.plot(x_point, y_point, 'ro', markersize=4)
    ax_zoom2.set_title("Zoom al píxel")
    BurkinaFaso_Satelitte.plot.imshow(ax=ax_zoom2,alpha=0.6)
    ax_zoom2.set_aspect('equal')
    
    # Añadir colorbar única
    # cbar = fig.colorbar(im.get_images()[0], 
    #                     ax=[ax_main, ax_zoom, ax_zoom2], 
    #                     orientation='vertical',
    #                     shrink=0.8)
    # cbar.set_label("ETa (mm/día)")
    
    plt.show()
    


