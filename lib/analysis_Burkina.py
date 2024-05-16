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

#%% Read AOI points and plots
burkina_faso_aoi = gpd.read_file('../data/AOI/burkina_faso_aoi.geojson')

