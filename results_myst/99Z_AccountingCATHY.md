---
title: Majadas water accounting
subtitle: 
thumbnail: ./img/EOAFRICA-logo.png
date: 04/22/2024
authors:
  - name: Hector Nieto
    affiliations:
      - ICA-CSIC
    orcid: 0000-0003-4250-6424
  - name: Benjamin Mary
    affiliations:
      - ICA-CSIC
    orcid: 0000-0001-7199-2885
  - name: Vicente Burchard-Levine
    affiliations:
      - ICA-CSIC
    orcid: 0000-0003-0222-8706
export: 
  - format: pdf
keywords: DA
---

# Water accounting water balance modelling + EO theory
Fluxes: 
- Infiltration/Exfiltration
- Runoff
- Boundary flux 
- Evaporation/Transpiration
```
ETa Earth Observations - ETa baseline
```
This means that the differences of water out is only due to ET. 
Water storing/depletion is **neglected**

## Steps to water accouting

- event delineation ratio ETa/ETp (from EO) 
- event classification using ratio ETa/ETp (from EO) --> rain or irrigation 
- resample to daily for analysis
- net = ETa (EO) - ETa (baseline)
- volume net = net * surface irrigated 

# water accounting testing using CATHY outputs
# Rain/irrigation event
Mesh descrition: 
- 3*3 raster with 1m*13 cell resolution
- 15 layers, up to 3m depth
- Surface cell = 1m2
- nnod: 16.0, nnod3: 256.0, nel: 810.0,
 
Rain event: 
- 1h rain (uniform), from time 0 to time 3600s
- 1e-6 m/s rate = 3.6 mm/day on each mesh nodes 
V=rain rate×time duration×surface area per node×number of nodes*1000 (m3 to L)

Results (dtcoupling):
```
Cum. Sum ACT volume * DURATION (TIME DELTA CHOOSEN for Cummulative SUM) * surface area per node
```
Results (mbeconv):
```
VIN : rain/iirgation * DELTAT · area
CUMVIN : VIN cumulative sum
```


# ET event





