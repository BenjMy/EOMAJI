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
import rioxarray as rxr


figPath = Path('../figures/')

#%%
SWC_path = Path('../data/TDR/')
SWC_coords = Majadas_utils.import_SWC_pos(path=SWC_path/'Majadas_coord_SWC_sensors_Benjamin.csv')[0]
SWC_coords.columns
SWC_coords[['X_ETRS89', 'Y_ETRS89']]
SWC_coords[['longwgs84', 'latwgs84']]

#%%

clc_codes = {
    "111": "Continuous urban fabric",
    "112": "Discontinuous urban fabric",
    "121": "Industrial or commercial units",
    "122": "Road and rail networks and associated land",
    "123": "Port areas",
    "124": "Airports",
    "131": "Mineral extraction sites",
    "132": "Dump sites",
    "133": "Construction sites",
    "141": "Green urban areas",
    "142": "Sport and leisure facilities",
    "211": "Non-irrigated arable land",
    "212": "Permanently irrigated land",
    "213": "Rice fields",
    "221": "Vineyards",
    "222": "Fruit trees and berry plantations",
    "223": "Olive groves",
    "231": "Pastures",
    "241": "Annual crops associated with permanent crops",
    "242": "Complex cultivation patterns",
    "243": "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "244": "Agro-forestry areas",
    "311": "Broad-leaved forest",
    "312": "Coniferous forest",
    "313": "Mixed forest",
    "321": "Natural grasslands",
    "322": "Moors and heathland",
    "323": "Sclerophyllous vegetation",
    "324": "Transitional woodland-shrub",
    "331": "Beaches, dunes, sands",
    "332": "Bare rocks",
    "333": "Sparsely vegetated areas",
    "334": "Burnt areas",
    "335": "Glaciers and perpetual snow",
    "411": "Inland marshes",
    "412": "Peat bogs",
    "421": "Salt marshes",
    "422": "Salines",
    "423": "Intertidal flats",
    "511": "Water courses",
    "512": "Water bodies",
    "521": "Coastal lagoons",
    "522": "Estuaries",
    "523": "Sea and ocean"
}


#%%
CLC_path = Path('../data/95732/Results/U2018_CLC2018_V2020_20u1.shp/U2018_CLC2018_V2020_20u1.shp')

CLC_Majadas = gpd.read_file(CLC_path)
# CLC_Majadas.plot()
CLC_Majadas.crs

# Check the current CRS
print("Original CRS:", CLC_Majadas.crs)



crsET = Majadas_utils.get_crs_ET()
AOI = Majadas_utils.get_Majadas_aoi(crs=CLC_Majadas.crs)
AOI_reprojected = AOI.to_crs(epsg=4326)

# AOI.crs 

# Reproject to WGS84 (EPSG:4326)
CLC_Majadas_reprojected = CLC_Majadas.to_crs(epsg=4326)
# CRSError: Invalid projection: EPSG:6326: (Internal Proj Error: proj_create: crs not found)

# Clip the CLC shapefile using the AOI
CLC_clipped = gpd.clip(CLC_Majadas_reprojected, AOI)
CLC_clipped['Land_Cover_Name'] = CLC_clipped['Code_18'].map(clc_codes)
# CLC_clipped
# Plot the clipped CLC data
fig, ax = plt.subplots(figsize=(10, 10))
CLC_clipped.plot(column='Land_Cover_Name', ax=ax, legend=True,
                 # label=CLC_clipped['Land_Cover_Name'].values,
                  alpha=0.2
                  )
# Add the AOI boundary to the plot for reference
# ax.legend(CLC_clipped.Land_Cover_Name.values)
# AOI.boundary.plot(ax=ax, color='red', linewidth=2)

# Majadas = rxr.open_rasterio('../data/Majadas.tif')
# Majadas.plot()



# cx.add_basemap(ax, crs=db.crs)

# # Create a map
# m = leafmap.Map(center=(AOI.geometry.centroid.y.mean(), AOI.geometry.centroid.x.mean()), 
#                 zoom=12, 
#                 basemap="HYBRID")

# # Add the clipped CLC data to the map
# m.add_gdf(CLC_clipped, column='Code_18', layer_name='Clipped CLC')

# # Add the AOI boundary to the map
# m.add_gdf(AOI, layer_name='AOI', style={"color": "red", "weight": 2})

# Show the plot
fig.savefig(figPath/'CLCover_map_Majadas.png', dpi=300)
