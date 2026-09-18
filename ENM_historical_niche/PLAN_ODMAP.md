# Historical ENM/SDM plan under ODMAP

## Goal
Build reproducible historical species-distribution / realized-niche models for two focal species already selected in the repository:

- Plant: black spruce (*Picea mariana*)
- Animal: red fox (*Vulpes vulpes*)

For each common historical period, combine observed distribution information with historical WorldClim climate and Global Human Modification (GHM v3), screen predictors, tune multiple algorithms, select the best-supported model, and generate continuous and binary historical niche/distribution maps.

## ODMAP structure
This project follows the repository `ODMAP/` checklist and is organized as Overview → Data → Model → Assessment → Prediction.

### O — Overview
- Objective: estimate period-specific realized climatic + anthropogenic niche and mapped habitat suitability.
- Outputs: continuous suitability, TSS-threshold binary distribution, variable importance, response curves, model evaluation, clamping/novelty diagnostics, E-space niche summaries, and a completed ODMAP report.
- Common historical target years: 1990, 1995, 2000, 2005, 2010, 2015, 2020.
- Temporal matching:
  - climate: five-year windows centered on target year (1988–1992, 1993–1997, 1998–2002, 2003–2007, 2008–2012, 2013–2017, 2018–2022)
  - GHM: exact target-year layer
  - black spruce: exact target-year annual dominant-species map
  - red fox: occurrence records within the corresponding five-year window

### D — Data
#### Species data
Black spruce:
- Source: Natural Resources Canada Annual Tree Species 1984–2022 / Awesome-GEE mirror.
- Response: annual 30 m dominant-species classification; class 18 = *Picea mariana*.
- Modeling grid: aggregate/rasterize to the climate grid; retain observed prevalence rather than treating unobserved non-forest pixels as species absences without checking masks.

Red fox:
- Source: GBIF occurrence records.
- Filters: accepted taxon, valid lon/lat, year in target period, no obvious coordinate problems, terrestrial points, remove exact duplicates, one record per environmental grid cell per period, remove institutional/centroid-like records when detectable.
- Sampling-bias control: spatial thinning + buffered accessible-area background; sensitivity comparison against broader background.

#### Climate
- Source: WorldClim historical monthly weather, CRU-TS 4.09 downscaled with WorldClim 2.1 bias correction.
- Raw variables: monthly tmin, tmax, precipitation.
- Native historical resolution used here: 2.5 arc-min.
- Derive 19 BIOCLIM variables separately for each five-year period from monthly climatologies.

#### Human activity
- Source: Global Human Modification v3, overall modification (AA), 300 m, 1990–2020 in five-year steps.
- Resample/aggregate to the common 2.5 arc-min model grid using mean.

#### Harmonization
- CRS: EPSG:4326.
- Common grid: WorldClim 2.5 arc-min.
- Continuous predictors: bilinear/mean aggregation as appropriate.
- Categorical observed tree map: class extraction before aggregation; use proportion of black-spruce pixels / presence threshold documented in results.
- All processing is scripted; large raw rasters are not committed to Git.

### M — Model
#### Predictor screening
1. Remove zero/near-zero variance layers within accessible area.
2. Ecological pre-screen of all 19 BIO variables + GHM.
3. Pairwise Pearson |r| >= 0.7: retain the variable with stronger univariate spatial-CV performance / clearer ecological interpretation.
4. VIF target < 5.
5. Repeat screening by species; use a common selected predictor set across periods for each species to keep temporal comparisons interpretable.

#### Algorithms
Candidate models follow the repository ENM curriculum and `biomod2` logic:
- GLM
- GAM
- GBM
- Random Forest
- Maxnet / MaxEnt-style penalized presence-background model
- XGBoost when dependency is available

#### Tuning
- Spatial block cross-validation, not random k-fold.
- Maxnet: feature classes L/LQ/LQH/LQHP and regularization multipliers 0.5–4.
- RF: mtry, min node size / terminal node settings.
- GBM: trees, interaction depth, shrinkage, min observations.
- GAM: smooth complexity constrained to avoid overfit.
- XGBoost: depth, learning rate, subsample, column sampling, regularization.
- Reproducible fixed random seed.

### A — Assessment
Primary selection rule is validation performance under spatial CV, with overfitting penalty.

Metrics:
- ROC-AUC
- TSS
- sensitivity / specificity
- Boyce index for presence-background outputs where applicable
- omission rate
- calibration / Brier score when feasible
- AUC train–validation difference as an overfitting diagnostic

Model choice:
1. Exclude models with unstable folds or unacceptable omission.
2. Rank by mean spatial-CV TSS, then AUC/Boyce.
3. If performance is statistically/ practically tied, choose the simpler model.
4. Build an ensemble only if multiple model classes pass quality thresholds and the ensemble improves spatial-CV performance; otherwise report the single best model.

### P — Prediction
For each species × period:
- continuous suitability raster
- binary suitable/unsuitable raster using validation-derived max-TSS threshold
- clamping / environmental novelty mask
- observed records/range overlay
- publication-ready map

Additional temporal products:
- suitability change between adjacent periods
- stable / loss / gain maps
- range area and centroid shift
- PCA environmental-space realized-niche plot by period

## Directory layout
```
ENM_historical_niche/
  PLAN_ODMAP.md
  README.md
  config/
  scripts/
  data_raw/            # gitignored / download manifests only
  data_processed/      # compact tables, selected predictors
  models/              # lightweight summaries; large objects optional/LFS
  results/
    evaluation/
    variable_selection/
    maps/
    niche_space/
  odmap/
  logs/
```

## Reproducibility / acceptance criteria
- Every raw source has URL/DOI/version/access date and checksum where available.
- No period is modeled unless occurrence/range and predictors genuinely overlap in time.
- No random train/test split is used as the primary performance estimate.
- Predictor selection is performed without leaking validation folds where computationally feasible; otherwise the limitation is documented.
- Final maps are generated only from the selected/tuned model or validated ensemble.
- ODMAP protocol is filled from actual analysis settings/results, not generic placeholders.
- Any data-access limitation is recorded explicitly rather than replaced with simulated data.
