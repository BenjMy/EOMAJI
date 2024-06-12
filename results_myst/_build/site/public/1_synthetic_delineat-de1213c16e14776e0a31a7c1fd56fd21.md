---
title: Synthetic irrigation delineation
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

In this study, to compute the **baseline ET** (ET without irrigation scheme), we use the CATHY code (Catchment Hydrology, Camporese et al., 2010) for modeling soil surface-subsurface water flow. We conducted experiments involving synthetic scenarios to assess their detectability with:
- varying sizes and levels of irrigation
- different frequencies of evapotranspiration (EO) 
- atmospheric boundary conditions, including rain events occurring between irrigation cycles. 


# Description of the synthetic models

The **regional domain size** is 100x100m square, flat. **Irrigation local size** is a 30x30m square centered on the regional domain. 
The vegetation is uniform on all the regional domain size, with root depth of 


```{figure} ../figures/EOMAJI_mesh.png
:class: bg-primary
:width: 600px
:alt: EOMAJI_mesh

EO-MAJI-IrrDelineation
```

```{figure} ../figures/vtksaturation.gif
:class: bg-primary
:width: 600px
:alt: EOMAJI_mesh_vtksaturation

Saturation over the course of the irrigation
```

The irrigation rate is **5e-07** m/s during **21600** sec. This is equivalent to 

| Key               | S0                     | S1                     |
|-------------------|------------------------|------------------------|
| ETp               | -1e-07                 |                 	      |
| nb_days           | 10                     |                 	      |
| nb_hours_ET       | 8                      |                 	      |
| irr_time_index    | 3                      |                 	      |
| irr_length        | 21600                  |                 	      |
| irr_flow          | 5e-07                  |                 	      |
| EO_freq_days      | 1                      |                 	      |
| ETp_window_size_x | 10                     |                 	      |



# Results

## Scenario 0










