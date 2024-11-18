---
title: Synthetic irrigation delineation with DA
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

This work focus on introducing heterogeneities into the soil and vegetation structure, which will be addressed using a data assimilation (DA) approach to calibrate the model. 
This calibration will help characterize soil/plant properties.

```{figure} ../figures/scenario_AquaCrop_sc0_weather_reference/root_map.png
:class: bg-primary
:width: 600px
:alt: root_map

root_map
```


:::{error}
:class: myclass1 myclass2
:name: Warming
- **Root depths** are rarely known so perturbated
:::

:::{note}
:class: myclass1 myclass2
:name: Note
- We consider errors on observations ETa (evaluated using pyTSEB) to 1mm/day.
:::


