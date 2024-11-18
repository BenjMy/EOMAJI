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

```{figure} ../figures/scenario_AquaCrop_sc0_weather_reference/root_map.png
:class: bg-primary
:width: 600px
:alt: root_map

root_map
```

- The soil is homogeneous all over the regional domain.

**AQUACROP IO**:
- The irrigation events based on threshold SMC. 
- The irrigation rate.

```{figure} ../figures/scenario_AquaCrop_sc0_weather_reference/july_rain_irr.png
:class: bg-primary
:width: 600px
:alt: july_rain_irr

AQUACROP schedule for irrigation management
```

**Atmospheric boundary conditions**:
- The **potential ETp** is the mean of the potential ETp extracted from ERA5 collection



```{figure} ../figures/scenario_AquaCrop_sc0_weather_reference/scenario_inputs.png
:class: bg-primary
:width: 600px
:alt: scenario_inputs

Cloud of points for scenario inputs data SPAIN ERA5 singlelevel hourly resampled daily. 
```


- **At time 0**, the soil is dry with an initial pressure head of -30m that is equivalent to a saturation level of 0.3?.
- **No flow** boundary conditions are imposed outside the regional domain.

## Earth observations
Earth Observations are available at a **daily frequency**. To mimick EO we used a hydrological model including the irrigation.


```{figure} ../figures/scenario_AquaCrop_sc0_weather_reference/plot_1d_evol_irrArea.png
:class: bg-primary
:width: 1000px
:alt: plot_1d_evol_irrArea

From top to bottom: net atmospheric boundary conditions (= ETp - rain), saturation water (sw), soil pressure head (psi) and actual evapotranspiration (baseline and EO observed)
```

```{figure} ../figures/scenario_AquaCrop_sc0_weather_reference/plot_1d_evol_OUTirrArea.png
:class: bg-primary
:width: 1000px
:alt: plot_1d_evol_OUTirrArea

From top to bottom: net atmospheric boundary conditions (= ETp - rain), saturation water (sw), soil pressure head (psi) and actual evapotranspiration (baseline and EO observed)
```


```{figure} ../figures/scenario_AquaCrop_sc0_weather_reference/irrigation_detection.png
:class: bg-primary
:width: 1000px
:alt: irrigation_detection

irrigation_detection
```
:::{note}
:class: myclass1 myclass2
:name: Note
- More irrigation detected than actual number of irrigation events
- Irrigation NON detected we overimposed with a rain event
:::


```{figure} ../figures/scenario_AquaCrop_sc0_weather_reference/plot_1d_net_irrArea.png
:class: bg-primary
:width: 1000px
:alt: plot_1d_net_irrArea

plot_1d_net_irrArea
```

:::{error}
:class: myclass1 myclass2
:name: Error
- Error in net estimation (**underestimated**)
- when the soil is saturated and irrigation is active ETa EO - ETa baseline does not work!
:::


```{figure} ../figures/scenario_AquaCrop_sc0_weather_reference/plot_accounting_summary_analysis.png
:class: bg-primary
:width: 1000px
:alt: plot_accounting_summary_analysis

plot_accounting_summary_analysis
```




