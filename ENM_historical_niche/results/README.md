# Model results

This directory is tracked for outputs from the ODMAP-aligned historical ENM workflow.

Generated subdirectories:
- `evaluation/`: spatial-CV tuning, final model comparison, thresholds and temporal holdout metrics.
- `variable_selection/`: correlation/VIF screening, permutation importance and final predictor sets.
- `maps/`: continuous suitability, binary distribution GeoTIFFs, and publication-ready PNG maps for 1990–2020.
- `niche_space/`: environmental-space/PCA niche outputs.

Only results produced from real source data are committed. Failed runs must not populate fabricated placeholders.
