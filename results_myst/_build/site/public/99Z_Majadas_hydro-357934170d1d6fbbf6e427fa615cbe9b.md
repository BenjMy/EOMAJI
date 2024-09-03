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

# Background and objectives

This work focus on the Majadas field site (Spain) water accounting. 

## Field site description
- Located Spain
- In total there is 10 land cover identified in the study area. The 2 mains are **savanna type ecosystem** + irrigated land
- Eddy Covariance Towers are installed and were previously used to validate the TSEB outputs 
- Soil water content sensors 

## WB Model description
- From October 2018 to Dec 2022 at daily scale
- DEM is simplified with a flat surface, cell resolution is given by the resolution of the ETp i.e. 300m
- Atmospheric boundary conditions are given using ETp computed from TSEB (using  FAO Penman- Monteith equation?) with a spatial resolution of 300m
- Rain (ERAnet5?) with a spatial resolution of 300m?
- Soil is homogeneous (type?)
- Root depth is defined according to Land cover type (arbitrary)
- Initial conditions are estimated from SMC sensors (?)

## DA Model description
- Objective: calibrate soil + root physical properties (during rainfed season)
- Observations to assimilate: SMC, ETa (derived from TSEB)

