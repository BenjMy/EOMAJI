#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:50:23 2023
@author: ben
"""

import geopandas as gpd
import shapely as shp
import leafmap
import contextily as cx
import geoplot
import rioxarray as rxr

#%%

# sites = gpd.read_file('../data/AOI/sites.qgz')


#%% Read AOI points and plots
south_africa_aoi = gpd.read_file('../data/AOI/south_africa_aoi.geojson')
majadas_aoi = gpd.read_file('../data/AOI/majadas_aoi.geojson')
guateng_province = gpd.read_file('../data/AOI/guateng_province.geojson')
burkina_faso_aoi = gpd.read_file('../data/AOI/burkina_faso_aoi.geojson')

sites = [
        south_africa_aoi,
        majadas_aoi,
        guateng_province,
        burkina_faso_aoi   
        ]

sites_names = ['south_africa','majadas','guateng','burkina_faso']

#%%
south_africa_aoi.crs
south_africa_aoi_wm = south_africa_aoi.to_crs(epsg=3857)
ax = south_africa_aoi_wm.plot(figsize=(10, 10), alpha=0.5, edgecolor="k")
cx.add_basemap(ax)
ax.set_axis_off()

#%%

for si, sni in zip(sites,sites_names):
    bbox = si.total_bounds
    print(si.area)
    
    # bbox = south_africa_aoi.buffer(5e-4).total_bounds
    leafmap.map_tiles_to_geotiff('../figures/satellite_' + sni + '.tif', 
                                 list(bbox), 
                                 zoom=12, 
                                 source='Esri.WorldImagery'
                                 )
    
#%%
south_africa_xr = rxr.open_rasterio('../figures/satellite.tif', engine='rasterio')
south_africa_xr.plot.imshow()
# south_africa_aoi.boundary.plot()

