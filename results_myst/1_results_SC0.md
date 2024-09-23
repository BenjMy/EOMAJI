---
title: Scenario 0
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

# Scenario 0

```{figure} ../figures/EOMAJI_mesh.png
:class: bg-primary
:width: 600px
:alt: EOMAJI_mesh

EO-MAJI-IrrDelineation
```


## Model
- The **regional domain size** is 1000x1000m square, flat. **Irrigation local size** is a 100x100m square centered on the regional domain. 

**Soil and Vegetation**
- The vegetation is uniform on all the regional domain size, with **root depth** of 1m typical from herbaceous crop. 
- The soil is homogeneous all over the regional domain.
- **At time 0**, the soil is dry with an initial pressure head of -30m that is equivalent to a saturation level of 0.3?.



```{figure} ../figures/scenario0/root_map
:class: bg-primary
:width: 600px
:alt: root_map

Irrigation areas
```

**Atmospheric Boundary conditions**
- The irrigation consist in a single event at ?? sec (day ??). 
- The irrigation rate is **5e-07** m/s during **21600** sec. This is equivalent to 43.2 mm/day during 6hours. 
- The **potential ETp** is homogeneous all over the domain and with time, set to -1e-07 m/s.
- **No flow** boundary conditions are imposed outside the regional domain.

**Observations**
- Earth Observations are available at a **daily frequency**. 

```{figure} ../figures/vtksaturation.gif
:class: bg-primary
:width: 600px
:alt: EOMAJI_mesh_vtksaturation

Saturation over the course of the irrigation
```

## Irrigation delimitation

```{figure} ../figures/scenario0/plot_1d_evol_irrArea.png
:class: bg-primary
:width: 700px
:alt: scenario0

Scenario 0, where irrigation triggering and length is highlighted by the dashad red vertical lines (approx. 3days). The plot show respectively the net atmospheric boundary conditions (ETp-Rain), the soil saturation water and pressure head on a surface node centered in the irrigation area. The bottom subplot shows the ETp (input), and the variation of ETa for both the baseline and with irrigation scheme.   
```


```{figure} ../figures/scenario0/plot_1d_evol_outArea.png
:class: bg-primary
:width: 700px
:alt: scenario0

SAME AS previous figure but outside the irrigation area.  
```


## Irrigation accounting

```{figure} ../figures/scenario0/netIrr_spatial_plot.png
:class: bg-primary
:width: 700px
:alt: netIrr_spatial_plot

netIrr_spatial_plot
```







## Classification & Decision

The local and regional changes are then **compared to a number of thresholds** to try to detect if:
- a) There is no input of water into the soil (e.g. local ETa/p does not increase above a threshold)
- b) There is input of water into the soil but due to rainfall (e.g. increase in regional ETa/p is over a
threshold and larger or similar to increase in local Eta/p)
- c) There is input of water to the soil due to irrigation (e.g. increase in local ETa/p is over a
threshold and significantly larger than increase in regional ETa/p)

```{figure} ../figures/scenario0/ratioETap_regional_withIRR_spatial_plot.png
:class: bg-primary
:width: 700px
:alt: ratioETap_withIRR_spatial_plot

There is input of water into the soil but due to rainfall : increase in regional ETa/p is over a
threshold and larger or similar to increase in local Eta/p)
```



```{figure} ../figures/scenario0/classify_events.png
:class: bg-primary
:width: 700px
:alt: classify_events

classify_events
```



```{figure} ../figures/scenario0/events_calendar.png
:class: bg-primary
:width: 800px
:alt: events_calendar

events_calendar
```











