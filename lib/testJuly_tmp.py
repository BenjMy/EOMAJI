#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 17:46:22 2024

@author: z0272571a
"""

import numpy as np
import matplotlib.pyplot as plt
import july
from july.utils import date_range

dates = date_range("2020-01-01", "2020-12-31")
data = np.random.randint(0, 14, len(dates))

july.heatmap(dates, data, title='Github Activity', cmap="github")
