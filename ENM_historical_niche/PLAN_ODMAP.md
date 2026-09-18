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
- Study domain: Canada for both taxa, so the plant and animal are fitted on the same environmental/background domain.
- Common grid: 10 arc-min, EPSG:4326; this targets continental realized-niche patterns rather than stand-scale occupancy.
- Outputs: continuous suitability, TSS-threshold binary distribution, variable importance, response curves/model diagnostics, E-space niche summaries, temporal range statistics, and a completed ODMAP report.
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
- Processing: reproject/aggregate to the 10 arc-min common grid by mode. Class 18 is presence; other valid mapped dominant-tree classes are the modeled contrast/absence at the continental grid scale; nodata is excluded.

Red fox:
- Source: GBIF occurrence API, taxon key 5219243, country=CA.
- Filters: presence records only, valid lon/lat, no GBIF geospatial issue, within Canada polygon, fossil records excluded, coordinate uncertainty <=20 km when reported, duplicate environmental-grid cells removed per period.
- Sampling-bias control: one presence per 10 arc-min cell per period plus spatial block cross-validation.
- Background: period-matched valid Canadian environmental cells excluding presence cells; the broad Canada-wide background is explicit because the species is widespread and the study domain itself defines the accessible calibration region for this analysis.

#### Climate
- Source: WorldClim historical monthly weather, CRU-TS 4.09 downscaled with WorldClim 2.1 bias correction.
- Raw variables: monthly tmin, tmax, precipitation.
- Historical resolution used: 10 arc-min.
- Derive 19 BIOCLIM variables separately for each five-year period from monthly climatologies.

#### Human activity
- Source: Global Human Modification v3, overall modification (AA), 300 m, 1990–2020 in five-year steps.
- Aggregate/resample to the common 10 arc-min model grid using mean.

#### Harmonization
- CRS: EPSG:4326.
- Common grid: WorldClim 10 arc-min.
- Continuous predictors: bilinear/mean aggregation as appropriate.
- Categorical annual tree map: modal aggregation to the common grid.
- All processing is scripted; large raw rasters are not committed to Git.

### M — Model
#### Predictor screening
1. Start from dynamic BIO1–BIO19 + GHM for every period.
2. Remove zero/near-zero variance layers.
3. Pairwise Pearson |r| >= 0.7 pruning.
4. VIF target <=5.
5. Estimate permutation importance only on held-out spatial folds.
6. Rank predictors and test nested top-k subsets; select the smallest subset whose spatial-CV TSS is within 0.01 of the maximum.
7. Use one selected predictor set per species across all historical periods to preserve temporal comparability.

#### Algorithms
Candidate model classes are:
- GLM (regularized logistic regression)
- GAM-like spline logistic model
- GBM
- Random Forest
- XGBoost when the dependency is available

These cover parametric, smooth, bagged-tree and boosted-tree responses while remaining executable without a Java MaxEnt dependency. Model class is selected empirically rather than predetermined.

#### Tuning
- Spatial block cross-validation, not random k-fold.
- GLM: regularization strength.
- GAM-like spline model: spline knots/degree + regularization.
- RF: depth, minimum leaf size, feature sampling.
- GBM: number of trees, learning rate, depth, subsampling.
- XGBoost: depth, learning rate, subsampling, column sampling, regularization.
- Reproducible fixed random seed = 20260918.

### A — Assessment
Primary selection rule is validation performance under spatial CV, with an explicit overfitting penalty.

Metrics:
- ROC-AUC
- TSS
- sensitivity / specificity
- Brier score
- training–validation AUC difference
- leave-one-period-out temporal AUC/TSS as a transferability check

Model choice:
1. Exclude failed/unstable candidates.
2. Rank primarily by mean spatial-CV TSS, secondarily by ROC-AUC and smaller train–validation AUC gap.
3. Determine the parsimonious factor subset using the 0.01-TSS rule.
4. Recompare algorithms using the final factor set and retune the selected algorithm.
5. Use out-of-fold predictions to estimate the final max-TSS binary threshold.

### P — Prediction
For each species × period:
- continuous suitability GeoTIFF
- binary suitable/unsuitable GeoTIFF using the out-of-fold max-TSS threshold
- observed records/range overlay PNG
- suitable area and suitability centroid

Additional temporal products:
- seven-period map panel for each species
- period-wise range statistics
- PCA environmental-space realized-niche plot by period
- completed ODMAP value table and Markdown report

## Directory layout
```
ENM_historical_niche/
  PLAN_ODMAP.md
  README.md
  config/
  scripts/
  data_raw/            # gitignored / recreated automatically
  data_processed/      # harmonized rasters, clean points/model tables
  models/              # workflow artifact (large binary objects ignored by Git)
  results/
    evaluation/
    variable_selection/
    maps/
    niche_space/
  odmap/
  logs/
```

## Reproducibility / acceptance criteria
- Every source has a stable URL/DOI/version or API query definition.
- No period is modeled unless occurrence/range and predictors genuinely overlap in time.
- No random train/test split is used as the primary performance estimate.
- Final variables must pass the documented collinearity/importance/parsimony procedure.
- Final model class and hyperparameters must come from spatial-CV comparison.
- Final maps are generated only from the selected/tuned final model.
- ODMAP protocol is filled from actual analysis settings/results, not generic placeholders.
- Any failed data source or analysis step is logged explicitly; simulated replacement data are prohibited.
