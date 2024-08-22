#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 10:55:50 2024

"""
from pathlib import Path
import geopandas as gpd
import Majadas_utils
import matplotlib.pyplot as plt
import leafmap
import contextily as cx

#%%
HydroPath_path = Path('../data/Copernicus_95732/Results/EU-Hydro.shp/??')
