#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 17:17:11 2024
"""

#%% Tuto vector file to xarray raster!

# https://corteva.github.io/geocube/stable/examples/categorical.html
import cf_xarray as cfxr
import geopandas as gpd

ssurgo_data = gpd.read_file("../soil_data_group.geojson")




ssurgo_data[ssurgo_data.hzdept_r==15].plot(column='sandtotal_r')
# this is only a subset of all of the classes
ssurgo_data.drclassdcd.drop_duplicates().values.tolist()

drclasses_complete = [
    'Poorly drained',
    'Somewhat poorly drained',
    'Excessively drained',
    'Subaqueous',
    'Well drained',
    'Somewhat excessively drained',
    'Very poorly drained',
    'Moderately well drained'
]

from geocube.api.core import make_geocube

categorical_enums = {'drclassdcd': drclasses_complete}
out_grid = make_geocube(
    vector_data=ssurgo_data,
    output_crs="epsg:32615",
    group_by='hzdept_r',
    resolution=(-100, 100),
    categorical_enums=categorical_enums
)


drclassdcd_string = out_grid['drclassdcd_categories'][out_grid['drclassdcd'].astype(int)]\
    .drop('drclassdcd_categories')
    
out_grid['drclassdcd'] = drclassdcd_string
out_grid

drclassdcd_slice = out_grid.drclassdcd.sel(hzdept_r=15)
drclassdcd_slice.where(drclassdcd_slice!=out_grid.drclassdcd.rio.nodata).plot()

# clay_slice = out_grid.claytotal_r.sel(hzdept_r=15)
# clay_slice.where(clay_slice!=out_grid.claytotal_r.rio.nodata).plot()

# drclassdcd_slice = out_grid.drclassdcd.sel(hzdept_r=15)
# drclassdcd_slice.where(drclassdcd_slice!=out_grid.drclassdcd.rio.nodata).plot()