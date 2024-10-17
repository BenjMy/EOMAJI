#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 14:05:06 2024
"""

import xarray as xr
from centum.delineation import ETAnalysis

# Create an instance of ETAnalysis
analysis = ETAnalysis()

# Assuming you have an xarray Dataset named 'ds_analysis'
ds_analysis = xr.Dataset()  # Replace this with your actual dataset

# Use the methods of the ETAnalysis dataclass
ds_analysis = analysis.compute_ratio_ETap_regional(ds_analysis)
ds_analysis = analysis.define_decision_thresholds(ds_analysis)

# Continue with your analysis...
