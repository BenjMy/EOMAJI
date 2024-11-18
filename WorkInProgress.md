# Work in Progress


## Processing

 - [ ] improve centum 
   - make it works for AQUACROP inputs 
   - make it works for Majadas inputs
- [ ] agricultural plot delineation using https://github.com/fieldsoftheworld
- [ ] Daily probability can be sum up to monthly values to improve estimates!

### Issues

**TWIN simulations** 

- [ ] Daily probability can be sum up to monthly values to improve estimates!
- [ ] issue with AQUACROP --> why continue to irrigate while the plant is harvested??
- [ ] why observation of ET are not constraining that much the state and the model parameters? almost no effects are visible!
- [x] July calendar only works for matplolib 3.7: temporary solved using a specific branch with a fix: 
	pip install git+https://github.com/thoellrich/july.git@fix-mpl.cbook.MatplotlibDepreciationWarning


- [x] Problem with calculation of net atmbc from xarray grid !! Only for irrigation simu (meaning the problem can be too high irrigation rate)
- [x] Problem when changing number of zones/ soil

- [ ] Issue with plot psi = f(time days). Values are shifted compare to real irrigations times.

### 💧 Water balance/ Land surface **Synthetic** modeling

- [x] Detecting event and classify!
- [ ] Water **quantification**: this can be done to a daily/weekly/monthly scale
  - [ ] Plot depletion + runoff
  - [ ] create df_daily_waterbalance from mbconv (VIN, VOUT, CUMVIN, STORE1, DSTORE...)

- [ ] **Aquacrop** Long term synthetic water balance modelling (some months)
  - [x] create and define how many rain events/ irrigation events
    - [ ] define **delineation errors** as the difference between (nb of events detected - nb of events)/nb of events
    - [ ] define **quantification errors** as the difference between (mm irr. detected - mm. irr applied)/nb of events
    - [x] DA with soil and root update during rainfed season
      - [x] Create EO ref model with PERMX < Truth 
      - [ ] Plot how much event detected with DA/ without DA?
   (Not really needed)
  - [ ] !varying root depths over time in AquaCrop!
  - [ ] initial_water_content
   
  - [x] Use ERA5 to create relevant scenario of ETp (see paper Italian) - ERA5 xarray?
     See: https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview
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
    - [ ] Extend to full catchment
    - [x]define root map with respect to corinne land cover
    - [ ] add EC towers as points of interest
    - [x] check simulation inputs
    - [ ] compare with REAL dataset 
      - [ ] compare hydro simulated with TDR real data
      - [ ] compare hydro simulated with piezometers real data
    - [ ] **CLASSIFY** interpretation of ETap ratio local/regional VS rain/irrigation event
    - [ ] DA with soil and root update during rainfed season
      - [ ] Plot how much event detected with DA/ without DA?
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

**October**
- 1st irrigation accounting plot/results --> overestimated as for now
- 1st tentative of DA during rainfed season --> this should demonstrate how much improvement in events detections
- build a package called centum: https://github.com/BenjMy/centum/tree/main
- issue with AQUACROP --> why continue to irrigate while the plant is harvested??

**July**
- 5000*5000 domain pixels (standard)
- how sensitive is compared to the irrigation amount? try to decrease the irr.
- convert in mm/day
- what about percolation? Inet does not account for it!
- simulating microWaves as well (=only account for the 5cm topsoil moisture, using an adequate CATHY mesh?) to showcase the drawbacks!
- non homogeneous soil conditions! calibrate the hydrological model during the rain season

