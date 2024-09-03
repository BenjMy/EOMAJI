# Work in Progress


## Processing

### Issues

**TWIN simulations** 
- [ ] July calendar only works for matplolib 3.7

- [x] Problem with calculation of net atmbc from xarray grid !! Only for irrigation simu (meaning the problem can be too high irrigation rate)
- [x] Problem when changing number of zones/ soil

- [ ] Issue with plot psi = f(time days). Values are shifted compare to real irrigations times.

### 💧 Water balance/ Land surface **Synthetic** modeling

- [x] Detecting event and classify!
- [ ] Water **quantification**: this can be done to a daily/weekly/monthly scale
- [ ] Plot depletion + runoff


- [ ] Long term synthetic water balance modelling (some months)
  - [ ] create and define how many rain events/ irrigation events
    - [ ] define **delineation errors** as the difference between (nb of events detected - nb of events)/nb of events
    - [ ] define **quantification errors** as the difference between (mm irr. detected - mm. irr applied)/nb of events
  - Use ERA5 to create relevant scenario of ETp (see paper Italian) - ERA5 xarray? ERA5 Land soil moisture simulator luca brocca: Simulator of ERA5 Land soil moisture in the 0-100 cm soil layer (results in the first comment) and I have created 3 scenario by using the climatology 1991-2020: +/- 20% of 		 precipitation, +25% air temperature (as observed in the last years). See: https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview  --> Mean potential evaporation rate	kg m-2 s-1
 
- [ ] Varying nb of earth observations --> impact on detection (plot in July)
- [ ] Varying EO resolutions --> impact on detection (plot in July)


- [x] Net irrigation accounting plot 1D
- [ ] July plot with event classification
- [ ] 5000*5000 domain pixels
- [x] convert units to mm/day

**Scenario i**
- [ ] varying **initial conditions**
- [ ] non homogeneous soil conditions:  This imply using simu.update_zones()
- [ ] non homogeneous vegetation type conditions (coupling with DAISY?)
  This imply using simu.update_root_map()
For both it is necessary to write the soil file accordingly

**Scenario i**
- [ ] simulating **microWaves**: only account for the 5cm topsoil moisture, using an adequate CATHY mesh to showcase the drawbacks!
  Same mesh but below 5 cm add boundary condition of fix neumann + dirichlet -->  psi = -200 (very dry soil no water) + >> flux to drive the water up?

**Scenario i**
- [ ] irrigation amount

### 💧 Water balance/ Land surface **Real field** modeling
- real data EOMAJI's field site:
  - [ ] **Spain Majadas**
    - [x]define root map with respect to corinne land cover
    - [ ] add EC towers as points of interest
    - [x] check simulation inputs
    - [ ] compare hydro simulated with TDR real data
    - [ ] **CLASSIFY** interpretation of ETap ratio local/regional VS rain/irrigation event
    - [ ] DA with soil and root update during rainfed season
    - [ ] Use/Implement Couvreur root water uptake model to account for compensation/competion
  - [ ] Burkina
    - [ ] 📌 Clip to AOI
  - [ ] Botswana
     - [ ] 📌 Clip to AOI

## Redaction
- [ ] **WEB**, publish results in myst pages
- [ ] **Poster** Frascati
- [ ] Peer review article:
  - [ ] **Title**: Water accounting in a semi-arid savanna type ecosystem
        - Geophysics + water balance at majadas field site
        - Journal of Advances in Modeling Earth Systems (JAMES) 
  - [ ] **Title**: Water accounting using WB, EB and DA
        - WB + EB + DA for water accounting in all field sites

## Meeting notes

**July**
- 5000*5000 domain pixels (standard)
- how sensitive is compared to the irrigation amount? try to decrease the irr.
- convert in mm/day
- what about percolation? Inet does not account for it!
- simulating microWaves as well (=only account for the 5cm topsoil moisture, using an adequate CATHY mesh?) to showcase the drawbacks!
- non homogeneous soil conditions! calibrate the hydrological model during the rain season

