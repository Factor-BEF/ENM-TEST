def write_odmap(species:str,algo:str,params:dict,predictors:List[str],threshold:float,df:pd.DataFrame,finaltab:pd.DataFrame,temporal:pd.DataFrame)->None:
    common={
      "o_objective_1":"Estimate historical realized climatic + anthropogenic niche and habitat suitability in Canada",
      "o_objective_2":"Continuous suitability and validation-derived binary distribution",
      "o_location_1":"Canada",
      "o_scale_1":f"{XMIN},{YMIN},{XMAX},{YMAX}","o_scale_2":f"{RES} degrees (~10 arc-min)",
      "o_scale_3":"1988-2022 observations/predictors summarized at 1990-2020 five-year target steps",
      "o_scale_4":"5-year climate/occurrence windows; exact-year GHM and black-spruce range",
      "o_predictors_1":"climatic; anthropogenic",
      "o_algorithms_1":"GLM, GAM-like splines, Random Forest, GBM, XGBoost (if available)",
      "o_workflow_1":"clean data -> derive dynamic BIOCLIM -> harmonize GHM -> collinearity/VIF screen -> spatial block CV tuning -> CV permutation importance -> parsimonious factor subset -> final algorithm comparison -> final model -> historical projections",
      "o_software_1":f"Python {sys.version.split()[0]}; rasterio; scikit-learn; xgboost={HAS_XGB}",
      "d_pred_1":";".join(predictors),"d_pred_2":"WorldClim CRU-TS 4.09 downscaled with WorldClim 2.1; GHM v3 Zenodo 10.5281/zenodo.14449495",
      "d_pred_4":"10 arc-min common model grid","d_pred_5":"EPSG:4326","d_pred_6":"1988-2022 climate; 1990-2020 GHM",
      "m_preselect_1":"Remove near-zero variance; Pearson |r|>=0.7 pruning; VIF<=5; spatial-CV permutation importance; smallest top-k subset within 0.01 TSS of maximum",
      "m_multicol_1":"Pearson correlation threshold 0.7 followed by VIF threshold 5",
      "m_settings_1":json.dumps({"algorithm":algo,"params":params}),
      "m_estim_3":"Permutation importance measured as held-out spatial-fold AUC decrease",
      "m_selection_1":"Primary: mean spatial-block CV TSS; secondary AUC, AUC train-validation gap; prefer parsimonious factor set",
      "m_depend_1":f"GroupKFold using {CFG['spatial_block_deg']}-degree spatial blocks; same cell/block retained across periods",
      "m_threshold_1":f"Max-TSS threshold from out-of-fold predictions: {threshold:.6f}",
      "a_perform_2":"ROC-AUC, TSS, sensitivity, specificity, Brier score, training-validation AUC difference",
      "p_output_1":"Continuous suitability + binary distribution + observed overlay + temporal range statistics + PCA environmental-space plots"
    }
    if species=="black_spruce":
        common.update({"o_taxon_1":"Picea mariana","o_bio_1":"range map","o_bio_2":"presence/absence grid",
                       "d_bio_1":"Picea mariana","d_bio_4":"Natural Resources Canada Annual Tree Species 1984-2022 / Awesome-GEE; class 18",
                       "d_bio_8":"Annual 30-m categorical map reprojected by modal aggregation to 10 arc-min",
                       "d_bio_9":"Canada mask; annual class 18 retained as presence; other valid annual classes treated as absence for modeling"})
    else:
        common.update({"o_taxon_1":"Vulpes vulpes","o_bio_1":"citizen science; specimen; observation records","o_bio_2":"presence-only point occurrence with background",
                       "d_bio_1":"Vulpes vulpes","d_bio_2":"GBIF Backbone Taxonomy key 5219243","d_bio_4":"GBIF occurrence API, Canada, target five-year windows",
                       "d_bio_8":"one occurrence retained per 10-arcmin grid cell and period","d_bio_9":"valid coordinates; no GBIF geospatial issue; uncertainty <=20 km when reported; fossil records excluded; Canada polygon check",
                       "d_bio_11":"random background from temporally matched valid Canadian environmental cells, excluding presence cells"})
    pd.DataFrame([{"element_id":k,"value":v} for k,v in common.items()]).to_csv(ODMAP/f"{species}_ODMAP_values.csv",index=False)
    metric=finaltab.iloc[0].to_dict(); temps=temporal[["auc","tss"]].mean().to_dict() if len(temporal) else {}
    md=f"""# ODMAP report — {species}\n\n## Overview\n- Taxon: {common['o_taxon_1']}\n- Objective: {common['o_objective_1']}\n- Study domain: Canada; EPSG:4326; 10 arc-min common grid.\n- Historical target years: {', '.join(map(str,YEARS))}.\n\n## Data\n- Biodiversity data: {common['d_bio_4']}\n- Climate: WorldClim historical monthly weather (CRU-TS 4.09 downscaled with WorldClim 2.1), converted to period-specific BIO1-BIO19.\n- Human influence: Global Human Modification v3 overall layer for exact target years.\n- Model sample size: {len(df):,}; presence rows: {(df.response==1).sum():,}; contrast/background rows: {(df.response==0).sum():,}.\n\n## Model\n- Final predictors: {', '.join(predictors)}.\n- Final algorithm: {algo}.\n- Hyperparameters: `{json.dumps(params,sort_keys=True)}`.\n- Spatial autocorrelation control: {common['m_depend_1']}.\n- Threshold: {threshold:.4f} (OOF max TSS).\n\n## Assessment\n- Final spatial-CV mean TSS: {metric.get('tss_validation',float('nan')):.3f}.\n- Final spatial-CV mean ROC-AUC: {metric.get('auc_validation',float('nan')):.3f}.\n- Mean train-validation AUC gap: {metric.get('auc_gap',float('nan')):.3f}.\n- Leave-one-period-out mean TSS: {temps.get('tss',float('nan')):.3f}.\n- Leave-one-period-out mean ROC-AUC: {temps.get('auc',float('nan')):.3f}.\n\n## Prediction\n- Continuous suitability and thresholded binary maps are generated for every target year.\n- Observed occurrences/range are overlaid on map figures.\n- Predictions are restricted to the Canadian mask and finite period-matched predictor combinations.\n\n## Key limitations\n- Red fox is presence-background rather than true absence data; TSS therefore depends on the background design.\n- Black spruce annual maps represent dominant tree species, not all individual occurrences; modal aggregation to 10 arc-min targets continental-scale realized niche, not stand-scale occupancy.\n- GHM and climate are resampled/aggregated to the coarser common grid for temporal comparability.\n"""
    (ODMAP/f"{species}_ODMAP_report.md").write_text(md,encoding="utf-8")


def run_species(species:str,df:pd.DataFrame)->None:
    log(f"===== MODEL {species} rows={len(df)} =====")
    filtered=correlation_vif_screen(df,ALL_PREDICTORS,species)
    algo0,params0,prelim=tune_models(df,filtered,species,"preliminary")
    est0=estimator_from_choice(algo0,params0)
    imp=cv_permutation_importance(est0,df,filtered,species)
    ranked=imp.variable.tolist()
    chosen=choose_parsimonious_subset(est0,df,ranked,species)
    algo,params,finaltab=compare_algorithms_fixed(df,chosen,prelim,species)
    final_grid=[p for p in model_spaces()[algo][1]]
    rec=[]
    for i,p in enumerate(final_grid):
        try:
            e=set_params(model_spaces()[algo][0],p); summ,_,_=evaluate_cv(e,df[chosen],df.response.astype(int).to_numpy(),df.group.to_numpy())
            rec.append({"algorithm":algo,"candidate":i,"params":json.dumps(p,sort_keys=True),**summ})
        except Exception as ex: log(f"retune {species} {algo} failed: {ex}")
    if rec:
        rt=pd.DataFrame(rec).sort_values(["tss_validation","auc_validation","auc_gap"],ascending=[False,False,True]);rt.to_csv(EVAL/f"{species}_final_retune.csv",index=False)
        params=json.loads(rt.iloc[0].params); finaltab=rt.reset_index(drop=True)
    final_est=estimator_from_choice(algo,params)
    oof,threshold,folds=get_oof_and_threshold(final_est,df,chosen,species)
    temporal=temporal_holdout(final_est,df,chosen,species)
    final_est.fit(df[chosen],df.response.astype(int))
    joblib.dump({"model":final_est,"predictors":chosen,"threshold":threshold,"algorithm":algo,"params":params},MODELS/f"{species}_final_model.joblib",compress=3)
    area=cell_area_km2(); stats=[]
    for y in YEARS:
        s=predict_period(final_est,chosen,y); b=(s>=threshold).astype("uint8"); b[~np.isfinite(s)]=255
        write_raster(MAPS/f"{species}_{y}_suitability.tif",s,descriptions=["Suitability"])
        write_raster(MAPS/f"{species}_{y}_binary.tif",b,dtype="uint8",nodata=255,descriptions=["Suitable=1"])
        valid=b!=255; suit=b==1; ar=float(np.nansum(area[suit]))
        rr,cc=np.where(suit)
        if len(rr):
            lon,lat=grid_lonlat(rr,cc); w=area[rr,cc]; clon=float(np.average(lon,weights=w));clat=float(np.average(lat,weights=w))
        else: clon=clat=float("nan")
        stats.append({"year":y,"suitable_area_km2":ar,"centroid_lon":clon,"centroid_lat":clat,"n_suitable_cells":int(suit.sum()),"n_valid_cells":int(valid.sum())})
        plot_species_year(species,y,s,b,threshold)
    pd.DataFrame(stats).to_csv(RESULTS/f"{species}_range_statistics.csv",index=False)
    summary_panel(species);niche_pca(species,df,chosen)
    write_odmap(species,algo,params,chosen,threshold,df,finaltab,temporal)
    log(f"FINAL {species}: algo={algo}, predictors={chosen}, threshold={threshold:.3f}")


def source_manifest()->None:
    rows=[]
    for d in [
      {"dataset":"WorldClim historical monthly weather","version":CFG["worldclim_version"],"url":"https://www.worldclim.org/data/monthlywth.html","doi":"10.1002/joc.5086; CRU TS4 citation Harris et al. 2020"},
      {"dataset":"Global Human Modification v3 1990-2020","version":"v1 fixed Zenodo record","url":f"https://zenodo.org/records/{CFG['ghm_record']}","doi":CFG["ghm_doi"]},
      {"dataset":"Annual Tree Species 1984-2022","version":"Natural Resources Canada NTEMS","url":"https://open.canada.ca/data/en/dataset/17396c45-8100-4ce6-a5d6-147c3d566d2e","doi":"10.1016/j.foreco.2024.122313"},
      {"dataset":"GBIF Vulpes vulpes occurrences","version":"live occurrence API queried by fixed filters and period","url":"https://api.gbif.org/v1/occurrence/search","doi":"taxon key 5219243"}
    ]: rows.append(d)
    pd.DataFrame(rows).to_csv(PROC/"source_manifest.csv",index=False)


def write_readme_summary()->None:
    lines=["# Historical realized-niche analysis results","",f"Run generated on {time.strftime('%Y-%m-%d %H:%M:%S')} runner time.","","Target periods: "+", ".join(map(str,YEARS)),"Common domain: Canada; 10 arc-min; EPSG:4326.","","## Outputs","- `results/maps/`: continuous and binary GeoTIFFs plus PNG figures for both species and all periods.","- `results/evaluation/`: spatial-CV tuning and temporal holdout metrics.","- `results/variable_selection/`: correlation/VIF screen, permutation importance, parsimonious subset selection.","- `results/niche_space/`: PCA environmental-space figures/tables.","- `odmap/`: completed ODMAP values and report for each species.","- `models/`: serialized final model objects (workflow artifact; omitted from Git if large).","","All raw large downloads are excluded from Git and can be recreated by rerunning the workflow."]
    (ROOT/"README.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def main()->None:
    source_manifest()
    prepare_worldclim()
    prepare_ghm()
    prepare_black_spruce()
    prepare_red_fox()
    bs=build_black_spruce_table();rf=build_red_fox_table()
    run_species("black_spruce",bs)
    run_species("red_fox",rf)
    write_readme_summary()
    log("PIPELINE COMPLETE")


if __name__=="__main__":
    main()
