---
title: AQUACROP irrigation accounting model
subtitle: Water balance model approach ETa/ETp
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

# Abstract




# Background and objectives


# Description of the synthetic models

## Hydrological model inputs
- The **regional domain size** is 100x100m square, flat. **Irrigation local size** is a 30x30m square centered on the regional domain. 
- The vegetation is uniform on all the regional domain size, with **root depth** of 1m typical from herbaceous crop. 

```{figure} ../figures/scenario_AquaCrop0/root_map.png
:class: bg-primary
:width: 600px
:alt: root_map

root_map
```

- The soil is homogeneous all over the regional domain.

- The irrigation consist in a ?? events at (based on Aquacrop outputs). 
- The irrigation rate is based on Aquacrop outputs as well.
- The rain events is based on Aquacrop outputs as well.

```{figure} ../figures/scenario_AquaCrop0/july_rain_irr.png
:class: bg-primary
:width: 600px
:alt: july_rain_irr

AQUACROP outputs for rain and irrigation management
```




- The **potential ETp** is the mean of the potential ETp extracted from ERA
- **At time 0**, the soil is dry with an initial pressure head of -30m that is equivalent to a saturation level of 0.3?.
- **No flow** boundary conditions are imposed outside the regional domain.

## Earth observations
Earth Observations are available at a **daily frequency**. To mimick EO we used a hydrological model including the irrigation.










