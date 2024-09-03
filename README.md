[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/BenjMy/EOMAJI/main)

# EOMAJI
This project aims to implement a prototype for irrigation mapping and crop yield estimation using inputs from the scientific ECOSTRESS and PRISMA missions.

> **Hypothesis**: Irrigated agricultural areas can be distinguished from adjacent agricultural parcels or natural areas through sudden increase in soil moisture and actual evapotranspiration which cannot be explained by
other factors (e.g. change in weather or vegetation cover).

> **Note**
> This project aims to implement a prototype that applies the TSEB model evaluated under Sen-ET with inputs from the scientific ECOSTRESS and PRISMA missions as an exploration of the capabilities for future operational Copernicus missions (LSTM+CHIME) to estimate ET

> **Methology**: 
- Irrigation delineation
![EO-MAJI-IrrNet](./figures/EO-MAJI-IrrDelineation.png)
- Irrigation net
![EO-MAJI-IrrNet](./figures/EO-MAJI-IrrNet.png)


## Synthetic TWIN water accounting models

### Simple scenario (10 days)
We conducted experiments involving synthetic scenarios to assess their detectability with: 
- varying sizes and levels of irrigation
- different frequencies of evapotranspiration (EO)
- atmospheric boundary conditions, including rain events occurring between irrigation cycles. 

The results indicate that establishing thresholds based on the **ratio of actual evapotranspiration Eta/p** facilitated the identification of localized changes in soil moisture associated with rain and irrigation practices. 
Future work: 
- introducing heterogeneities into the soil structure, which will be addressed using a data assimilation approach to calibrate the model during rainfed seasons. 
This calibration will help characterize soil properties and serve as a baseline model for simulations during the irrigation season.

### Scenario based on historical data 

Using the Climate Data Store (CDS) API: https://cds-beta.climate.copernicus.eu/how-to-api

- ERA5 hourly data on single levels from 1940 to present
- Crop productivity and evapotranspiration indicators from 2000 to present derived from satellite observations


`pip install cdsapi´



## Real field sites water accounting
- Burkina Faso grids:
- Botswana grids:
- SPAIN Majadas grids: S3/Meteo = E030N006T6 and S2/Landast = X0033_Y0044 

**DTM**
- Majadas: MDT02-ETRS89-HU30-0624-1-COB2.tif (from https://centrodedescargas.cnig.es/CentroDescargas/index.jsp)
The datasets are accessible in the OSF: https://osf.io/zxuy5/

**Land Cover Map**
https://land.copernicus.eu/en/cart-downloads
CORINE Land Cover 2018 (vector/raster 100 m), Europe, 6-yearly - Shapefile (NUTS: ES432)

**Watershed**
https://land.copernicus.eu/en/cart-downloads
EU-Hydro River Network Database 2006-2012 (vector), Europe - Shapefile (NUTS: ES432)
EU-Hydro River Network Database 2006-2012 (vector), Europe - Shapefile (Bounding Box)


### Example: Spain irrigation accounting 
- 4 tiles to cover Spain
- format= *.vrt (virtual raster tile)
- Resolution = 300m at daily scale
- Range from Oct. 2008 to Dec. 2022
- Nomenclature:
  - TPday: daily total precipitation (mm)
  - ET0: potential ET (mm/day)
  - ET: actual ET (mm/day)
  - *GS*: gap filled
- Interesting irrigation districts:
  - Majadas de Tietar
  - Leida/Balaguer
  - Astorga/Leon/Benavente (Duero Bassin)
 
 
### !Irrigation accounting quick check!
Pick at least two point of interest (POI): an agricultural area with irrigation VS non irrigated area (example forest ecosystem)
- plot time serie of ETp: is there differences between irr. vs non irrigated points? 
- plot time serie of ETa: is there differences between irr. vs non irrigated points? 
- plot 1:1 between ETa from WB model and ETa from EB model

 
## To go further
### New satellite missions

- [ECOSTRESS](https://ecostress.jpl.nasa.gov/) thermal sensor: can provide robust measurements of Land Surface Temperature at high spatial and temporal resolutions
- [PRISMA hyperspectral sensor](https://www.eoportal.org/satellite-missions/prisma-hyperspectral): data for accurate modelling of such biophysical parameters as fractions of green and woody/senescent vegetation broadband albedo or leaf area index
- [Copernicus Sentinel Expansion missions](https://www.esa.int/Applications/Observing_the_Earth/Copernicus/Copernicus_Sentinel_Expansion_missions) - LSTM and CHIME

## Contributors
- CSIC (H. Nieto, M. P. Martín, V. Burchard)
- DHI (R. Guzinski,M. Munk)
- University of Leicester (D. Ghent)




