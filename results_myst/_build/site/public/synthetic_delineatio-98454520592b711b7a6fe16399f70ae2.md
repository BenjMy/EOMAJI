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

<!--
comment
-->


The ratio of actual to potential ET (ETa/p) should be used in order to avoid changes in ET
due to changes in weather (e.g. increased wind speed) or crop cover (e.g. quick development of
leaves) being attributed to irrigation. This ratio is closely related to root-zone water availability and
therefore is mainly influenced by irrigation or rainfall events.

This is achieved by first calculating the change in ETa/p between the time on which irrigation is to be detect and most recent previous time on which ET estimates are available. This change is calculated both locally (i.e. at individual pixel level) and regionally (i.e. as an average change in all agricultural pixels within 10 km window). 

The local and regional changes are then compared to a number of thresholds to try to detect if:
- a) There is no input of water into the soil (e.g. local ETa/p does not increase above a threshold)
- b) There is input of water into the soil but due to rainfall (e.g. increase in regional ETa/p is over a
threshold and larger or similar to increase in local Eta/p)
c) There is input of water to the soil due to irrigation (e.g. increase in local ETa/p is over a
threshold and significantly larger than increase in regional ETa/p)

Detected irrigation events are further split into low, medium and high probability based on another set
of thresholds. Since irrigation is normally applied on a larger area, the raster map with per-pixel
irrigation events is cleaned up by removing isolated pixels in which irrigation was detected.




```{figure} ../img/EO-MAJI-IrrDelineation.png
:class: bg-primary
:width: 600px
:alt: EO-MAJI-IrrDelineation

EO-MAJI-IrrDelineation
```

# Description of the scenario

- Regional domain size = 100x100
- Irrigation local size = 30x30






