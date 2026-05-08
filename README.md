# Beyond Residence: Mobility-Driven PM2.5 Exposure Disparity in the DMV

## Overview

This project examines whether daily mobility patterns systematically increase PM2.5 
exposure burden across the Washington DC metropolitan area (DMV), and which 
populations bear the greatest burden. Rather than asking where people live relative 
to pollution, this study asks where people go — computing a visitor-weighted 
mobility-driven exposure differential per census block group (CBG).

**Study area:** Washington DC, Montgomery County MD, Prince George's County MD, 
Arlington VA, Fairfax County VA, Alexandria VA, Fairfax City VA, Falls Church VA  
**Study period:** January 2018 – November 2021 (47 months)  
**Unit of analysis:** 2,548 Census Block Groups  

---

## Data Sources

### 1. GHAP GlobalHighPM2.5
- **What**: Gapless 1km monthly PM2.5 estimates for global land areas
- **Years used**: 2018–2021
- **Download**: https://zenodo.org/records/10800980
- **Citation**: Wei et al. (2023) Nature Communications 14, 8349
- **Place in**: `data/raw/ghap/`

### 2. SafeGraph Patterns
- **What**: Monthly POI visitor flows with origin CBG, visitor counts, dwell time
- **Years used**: 2018–2021
- **Access**: Requires SafeGraph academic license (https://www.safegraph.com)
- **Not included** due to privacy restrictions and licensing
- **Place in**: `data/raw/safegraph/YYYY/`

### 3. ACS 2019 5-Year Estimates
- **What**: CBG-level demographic characteristics
- **Tables used**:
  - B01003 — Total population
  - B03002 — Race and Hispanic origin
  - B19013 — Median household income
  - B17017 — Poverty status
  - B25044 — Vehicle availability
  - B15003 — Educational attainment
  - B25003 — Housing tenure (owner/renter)
- **Download**: Automatically via `pygris`/`tidycensus`, or manually from https://data.census.gov
- **Place in**: `data/raw/acs/`

---

## Pipeline
Run scripts in order. Scripts 01–04 are Python, 05–07 are R Markdown, 08 is a Jupyter notebook.

```
01_process_ghap.py             →  data/processed/ghap_pm25_monthly.csv
02_process_acs.py              →  data/processed/cbg_demographics.csv
03_process_safegraph_flows.py  →  data/processed/flow_map.csv
04_build_final_data.py         →  data/processed/final_data_monthly.csv
05_descriptive.Rmd             →  descriptive stats, temporal trends, quartile analysis
06_spatial.Rmd                 →  Moran's I, LISA clusters, choropleth maps
07_modeling.Rmd                →  OLS → SEM → interaction model → COVID comparison
08_flow_map.ipynb              →  data/processed/kepler_pre_covid.csv
                                   data/processed/kepler_covid.csv
```

---

## Requirements

### Python
```
pandas
geopandas
pygris
rasterio
rasterstats
xarray
numpy
```

### R
```
tidyverse
sf
spdep
spatialreg
tigris
sandwich
lmtest
patchwork
scales
```

## Note on SafeGraph Data

SafeGraph Patterns data is not included in this repository due to licensing restrictions.
Researchers wishing to replicate this analysis should apply for academic access through
SafeGraph's data-for-academics program. The data processing pipeline (Scripts 03 and 04)
is fully documented and can be applied to any SafeGraph Patterns download covering the
Washington-Baltimore-Arlington CBSA.
