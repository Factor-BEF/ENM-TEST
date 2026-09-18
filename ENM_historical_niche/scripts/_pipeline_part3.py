def correlation_vif_screen(df: pd.DataFrame, predictors: List[str], species: str) -> List[str]:
    X=df[predictors].replace([np.inf,-np.inf],np.nan).dropna()
    keep=[v for v in predictors if X[v].std()>1e-8]
    while True:
        c=X[keep].corr().abs(); np.fill_diagonal(c.values,0)
        mx=float(c.to_numpy().max()) if len(keep)>1 else 0
        if mx < float(CFG["correlation_cutoff"]) or len(keep)<=2: break
        i,j=np.unravel_index(np.argmax(c.to_numpy()),c.shape)
        vi,vj=keep[i],keep[j]
        mi=float(c[vi].mean()); mj=float(c[vj].mean())
        drop=vi if mi>=mj else vj
        keep.remove(drop); log(f"{species}: corr prune {drop} (pair {vi}/{vj}, r={mx:.3f})")
    while len(keep)>2:
        corr=X[keep].corr().to_numpy()
        try: inv=np.linalg.pinv(corr); vifs=np.diag(inv)
        except Exception: break
        maxv=float(np.nanmax(vifs))
        if maxv <= float(CFG["vif_cutoff"]): break
        drop=keep[int(np.nanargmax(vifs))]; keep.remove(drop)
        log(f"{species}: VIF prune {drop} (VIF={maxv:.2f})")
    pd.DataFrame({"selected_after_collinearity":keep}).to_csv(VARSEL/f"{species}_collinearity_selected.csv",index=False)
    return keep


def max_tss_threshold(y: np.ndarray, p: np.ndarray) -> Tuple[float,float,float,float]:
    thresholds=np.unique(np.quantile(p,np.linspace(0.01,0.99,199)))
    best=(-999,0.5,0,0)
    for t in thresholds:
        yh=(p>=t).astype(int)
        tp=((yh==1)&(y==1)).sum(); fn=((yh==0)&(y==1)).sum(); tn=((yh==0)&(y==0)).sum(); fp=((yh==1)&(y==0)).sum()
        sens=tp/max(tp+fn,1); spec=tn/max(tn+fp,1); tss=sens+spec-1
        if tss>best[0]: best=(tss,float(t),sens,spec)
    return best[1],best[0],best[2],best[3]


def model_spaces() -> Dict[str, Tuple[Any,List[dict]]]:
    spaces={
        "GLM": (Pipeline([("scale",StandardScaler()),("model",LogisticRegression(max_iter=3000,class_weight="balanced",solver="liblinear",random_state=CFG["seed"]))]),
                [{"model__C":c} for c in [0.1,1.0,10.0]]),
        "GAM": (Pipeline([("spline",SplineTransformer(include_bias=False)),("scale",StandardScaler()),("model",LogisticRegression(max_iter=3000,class_weight="balanced",solver="liblinear",random_state=CFG["seed"]))]),
                [{"spline__n_knots":k,"spline__degree":d,"model__C":c} for k,d,c in [(4,2,1.0),(5,2,1.0),(5,3,0.5),(6,3,1.0)]]),
        "RF": (RandomForestClassifier(n_estimators=350,class_weight="balanced_subsample",n_jobs=-1,random_state=CFG["seed"]),
               [{"max_depth":d,"min_samples_leaf":leaf,"max_features":mf} for d,leaf,mf in [(10,5,"sqrt"),(18,3,"sqrt"),(None,5,"sqrt"),(18,8,0.6)]]),
        "GBM": (GradientBoostingClassifier(random_state=CFG["seed"]),
                [{"n_estimators":n,"learning_rate":lr,"max_depth":d,"subsample":0.8} for n,lr,d in [(150,0.05,2),(250,0.03,3),(250,0.05,2),(350,0.03,2)]])
    }
    if HAS_XGB:
        spaces["XGBOOST"]=(XGBClassifier(tree_method="hist",eval_metric="logloss",n_jobs=2,random_state=CFG["seed"]),
            [{"n_estimators":n,"max_depth":d,"learning_rate":lr,"subsample":0.8,"colsample_bytree":0.8,"reg_lambda":rl}
             for n,d,lr,rl in [(250,3,0.05,1),(350,3,0.03,5),(300,5,0.03,5),(250,5,0.05,10)]])
    return spaces


def set_params(est: Any, params: dict) -> Any:
    e=clone(est); e.set_params(**params); return e


def evaluate_cv(est: Any, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, n_splits:int|None=None) -> Tuple[dict,np.ndarray,List[dict]]:
    unique=np.unique(groups); ns=min(n_splits or int(CFG["cv_folds"]), len(unique))
    if ns<2: raise RuntimeError("Not enough spatial groups for CV")
    cv=GroupKFold(n_splits=ns); oof=np.full(len(y),np.nan); folds=[]
    for fold,(tr,te) in enumerate(cv.split(X,y,groups)):
        if len(np.unique(y[tr]))<2 or len(np.unique(y[te]))<2: continue
        m=clone(est); m.fit(X.iloc[tr],y[tr])
        ptr=m.predict_proba(X.iloc[tr])[:,1]; pte=m.predict_proba(X.iloc[te])[:,1]
        thr,_,_,_=max_tss_threshold(y[tr],ptr)
        _,tss,_,_=max_tss_threshold(y[te],pte)
        yh=(pte>=thr).astype(int)
        tp=((yh==1)&(y[te]==1)).sum(); fn=((yh==0)&(y[te]==1)).sum(); tn=((yh==0)&(y[te]==0)).sum(); fp=((yh==1)&(y[te]==0)).sum()
        sens_fixed=tp/max(tp+fn,1); spec_fixed=tn/max(tn+fp,1); tss_fixed=sens_fixed+spec_fixed-1
        auc_te=roc_auc_score(y[te],pte); auc_tr=roc_auc_score(y[tr],ptr)
        folds.append({"fold":fold,"auc_validation":auc_te,"auc_training":auc_tr,"auc_gap":auc_tr-auc_te,
                      "tss_validation":tss_fixed,"tss_validation_oracle":tss,"sensitivity":sens_fixed,"specificity":spec_fixed,
                      "brier":brier_score_loss(y[te],pte),"threshold_from_training":thr,"n_test":len(te)})
        oof[te]=pte
    if not folds: raise RuntimeError("All spatial CV folds invalid")
    fd=pd.DataFrame(folds)
    summary={k:float(fd[k].mean()) for k in ["auc_validation","auc_training","auc_gap","tss_validation","sensitivity","specificity","brier"]}
    summary["n_folds"]=len(fd)
    return summary,oof,folds


def tune_models(df: pd.DataFrame, predictors: List[str], species: str, suffix: str) -> Tuple[str,dict,pd.DataFrame]:
    X=df[predictors]; y=df["response"].astype(int).to_numpy(); g=df["group"].to_numpy()
    rec=[]
    for algo,(base,grid) in model_spaces().items():
        for i,params in enumerate(grid):
            try:
                est=set_params(base,params); summ,_,_=evaluate_cv(est,X,y,g)
                rec.append({"algorithm":algo,"candidate":i,"params":json.dumps(params,sort_keys=True),**summ})
                log(f"{species} {suffix} {algo} {i}: TSS={summ['tss_validation']:.3f} AUC={summ['auc_validation']:.3f} gap={summ['auc_gap']:.3f}")
            except Exception as e:
                log(f"{species} {suffix} {algo} {i} failed: {e}")
    tbl=pd.DataFrame(rec)
    if tbl.empty: raise RuntimeError(f"No model candidates succeeded for {species}")
    tbl=tbl.sort_values(["tss_validation","auc_validation","auc_gap"],ascending=[False,False,True]).reset_index(drop=True)
    tbl.to_csv(EVAL/f"{species}_model_tuning_{suffix}.csv",index=False)
    best=tbl.iloc[0]; return str(best.algorithm),json.loads(best.params),tbl


def estimator_from_choice(algo:str,params:dict)->Any:
    base=model_spaces()[algo][0]; return set_params(base,params)


def cv_permutation_importance(est:Any,df:pd.DataFrame,predictors:List[str],species:str)->pd.DataFrame:
    X=df[predictors]; y=df.response.astype(int).to_numpy(); g=df.group.to_numpy(); cv=GroupKFold(n_splits=min(int(CFG["cv_folds"]),len(np.unique(g))))
    vals=[]
    for fold,(tr,te) in enumerate(cv.split(X,y,g)):
        if len(np.unique(y[te]))<2: continue
        m=clone(est); m.fit(X.iloc[tr],y[tr])
        pi=permutation_importance(m,X.iloc[te],y[te],scoring="roc_auc",n_repeats=4,random_state=int(CFG["seed"])+fold,n_jobs=1)
        for v,mean,sd in zip(predictors,pi.importances_mean,pi.importances_std): vals.append({"fold":fold,"variable":v,"importance_auc_drop":mean,"sd":sd})
    raw=pd.DataFrame(vals); raw.to_csv(VARSEL/f"{species}_permutation_importance_folds.csv",index=False)
    summary=raw.groupby("variable")["importance_auc_drop"].agg(importance_mean="mean",importance_sd="std").reset_index()
    summary=summary.sort_values("importance_mean",ascending=False).reset_index(drop=True); summary.to_csv(VARSEL/f"{species}_permutation_importance.csv",index=False)
    return summary


def choose_parsimonious_subset(est:Any,df:pd.DataFrame,ranked:List[str],species:str)->List[str]:
    y=df.response.astype(int).to_numpy(); g=df.group.to_numpy(); rec=[]
    if len(ranked)<=2: return ranked
    for k in range(2,len(ranked)+1):
        vars_k=ranked[:k]
        try:
            summ,_,_=evaluate_cv(est,df[vars_k],y,g)
            rec.append({"k":k,"variables":";".join(vars_k),**summ})
            log(f"{species} top{k}: TSS={summ['tss_validation']:.3f} AUC={summ['auc_validation']:.3f}")
        except Exception as e: log(f"subset top{k} failed: {e}")
    tab=pd.DataFrame(rec); tab.to_csv(VARSEL/f"{species}_subset_selection.csv",index=False)
    mx=float(tab.tss_validation.max())
    eligible=tab[tab.tss_validation>=mx-0.01].sort_values(["k","auc_validation"],ascending=[True,False])
    chosen=str(eligible.iloc[0].variables).split(";")
    pd.DataFrame({"selected_predictor":chosen}).to_csv(VARSEL/f"{species}_final_predictors.csv",index=False)
    return chosen


def compare_algorithms_fixed(df:pd.DataFrame,predictors:List[str],prelim_table:pd.DataFrame,species:str)->Tuple[str,dict,pd.DataFrame]:
    rec=[]; y=df.response.astype(int).to_numpy();g=df.group.to_numpy()
    for algo in prelim_table.algorithm.unique():
        bestrow=prelim_table[prelim_table.algorithm==algo].sort_values(["tss_validation","auc_validation"],ascending=False).iloc[0]
        params=json.loads(bestrow.params); est=estimator_from_choice(algo,params)
        try:
            summ,_,_=evaluate_cv(est,df[predictors],y,g)
            rec.append({"algorithm":algo,"params":json.dumps(params,sort_keys=True),**summ})
        except Exception as e: log(f"final compare {species} {algo} failed: {e}")
    tab=pd.DataFrame(rec).sort_values(["tss_validation","auc_validation","auc_gap"],ascending=[False,False,True]).reset_index(drop=True)
    tab.to_csv(EVAL/f"{species}_final_algorithm_comparison.csv",index=False)
    b=tab.iloc[0]; return str(b.algorithm),json.loads(b.params),tab


def temporal_holdout(est:Any,df:pd.DataFrame,predictors:List[str],species:str)->pd.DataFrame:
    rec=[]
    for yhold in YEARS:
        tr=df.period!=yhold; te=df.period==yhold
        if te.sum()==0 or len(np.unique(df.loc[te,"response"]))<2: continue
        m=clone(est);m.fit(df.loc[tr,predictors],df.loc[tr,"response"].astype(int))
        ptr=m.predict_proba(df.loc[tr,predictors])[:,1]; p=m.predict_proba(df.loc[te,predictors])[:,1]
        thr,_,_,_=max_tss_threshold(df.loc[tr,"response"].to_numpy().astype(int),ptr)
        yy=df.loc[te,"response"].to_numpy().astype(int); yh=(p>=thr).astype(int)
        tp=((yh==1)&(yy==1)).sum();fn=((yh==0)&(yy==1)).sum();tn=((yh==0)&(yy==0)).sum();fp=((yh==1)&(yy==0)).sum()
        sens=tp/max(tp+fn,1);spec=tn/max(tn+fp,1)
        rec.append({"held_out_period":yhold,"auc":roc_auc_score(yy,p),"tss":sens+spec-1,"sensitivity":sens,"specificity":spec,"threshold":thr,"n":len(yy)})
    out=pd.DataFrame(rec); out.to_csv(EVAL/f"{species}_temporal_holdout.csv",index=False); return out


def get_oof_and_threshold(est:Any,df:pd.DataFrame,predictors:List[str],species:str)->Tuple[np.ndarray,float,pd.DataFrame]:
    summ,oof,folds=evaluate_cv(est,df[predictors],df.response.astype(int).to_numpy(),df.group.to_numpy())
    good=np.isfinite(oof);thr,tss,sens,spec=max_tss_threshold(df.response.to_numpy().astype(int)[good],oof[good])
    pd.DataFrame(folds).to_csv(EVAL/f"{species}_final_spatial_cv_folds.csv",index=False)
    pd.DataFrame([{"threshold":thr,"oof_tss":tss,"oof_sensitivity":sens,"oof_specificity":spec,**summ}]).to_csv(EVAL/f"{species}_final_metrics.csv",index=False)
    return oof,thr,pd.DataFrame(folds)


def predict_period(est:Any,predictors:List[str],year:int)->np.ndarray:
    d=load_predictor_stack(year); mask=CANADA_MASK.copy()
    for v in predictors: mask &= np.isfinite(d[v])
    idx=np.flatnonzero(mask); out=np.full((HEIGHT,WIDTH),np.nan,dtype="float32")
    if len(idx)==0: return out
    batch=50000
    for s in range(0,len(idx),batch):
        ids=idx[s:s+batch]
        X=pd.DataFrame({v:d[v].ravel()[ids] for v in predictors})
        out.ravel()[ids]=est.predict_proba(X)[:,1].astype("float32")
    return out


def cell_area_km2() -> np.ndarray:
    rows=np.arange(HEIGHT); _,lat=grid_lonlat(rows,np.zeros_like(rows)); R=6371.0088
    dlat=np.deg2rad(RES);dlon=np.deg2rad(RES)
    areas=(R**2)*dlat*dlon*np.cos(np.deg2rad(lat))
    return np.repeat(areas[:,None],WIDTH,axis=1)


def plot_species_year(species:str,year:int,suit:np.ndarray,binary:np.ndarray,threshold:float)->None:
    fig,ax=plt.subplots(figsize=(9,6))
    im=ax.imshow(suit,extent=[XMIN,XMAX,YMIN,YMAX],origin="upper",vmin=0,vmax=1,aspect="auto")
    try: ax.contour(binary.astype(float),levels=[0.5],extent=[XMIN,XMAX,YMIN,YMAX],origin="upper",linewidths=0.8)
    except Exception: pass
    if species=="red_fox":
        p=PROC/f"red_fox_occurrences_{year}.csv"
        if p.exists():
            d=pd.read_csv(p); ax.scatter(d.lon,d.lat,s=8,alpha=0.55,label="Observed GBIF cells")
    else:
        obs,_=read_raster(PROC/f"black_spruce_observed_{year}.tif")
        oo=np.where(obs[0]==1,1.0,np.nan)
        try: ax.contour(oo,levels=[0.5],extent=[XMIN,XMAX,YMIN,YMAX],origin="upper",linewidths=0.7)
        except Exception: pass
    ax.set_xlim(XMIN,XMAX);ax.set_ylim(YMIN,YMAX);ax.set_xlabel("Longitude");ax.set_ylabel("Latitude")
    ax.set_title(f"{species.replace('_',' ').title()} historical realized-niche suitability — {year}\ncontinuous suitability; binary boundary at TSS threshold={threshold:.3f}")
    fig.colorbar(im,ax=ax,label="Suitability")
    if species=="red_fox": ax.legend(loc="lower left",fontsize=8)
    fig.tight_layout();fig.savefig(MAPS/f"{species}_{year}_niche_map.png",dpi=180);plt.close(fig)


def summary_panel(species:str)->None:
    fig,axs=plt.subplots(2,4,figsize=(17,8.5),sharex=True,sharey=True);axs=axs.ravel()
    for ax,y in zip(axs,YEARS):
        a,_=read_raster(MAPS/f"{species}_{y}_suitability.tif");im=ax.imshow(a[0],extent=[XMIN,XMAX,YMIN,YMAX],origin="upper",vmin=0,vmax=1,aspect="auto");ax.set_title(str(y))
    axs[-1].axis("off");fig.suptitle(f"{species.replace('_',' ').title()} — historical realized niche in Canada")
    fig.colorbar(im,ax=axs.tolist(),shrink=0.75,label="Suitability");fig.savefig(MAPS/f"{species}_historical_panel.png",dpi=180,bbox_inches="tight");plt.close(fig)


def niche_pca(species:str,df:pd.DataFrame,predictors:List[str])->None:
    X=df[predictors].copy(); scaler=StandardScaler();Xs=scaler.fit_transform(X);pca=PCA(n_components=2,random_state=CFG["seed"]);pcs=pca.fit_transform(Xs)
    out=df[["period","response"]].copy();out["PC1"]=pcs[:,0];out["PC2"]=pcs[:,1];out.to_csv(NICHE/f"{species}_pca_scores.csv",index=False)
    fig,ax=plt.subplots(figsize=(8,6))
    for y in YEARS:
        sub=out[(out.period==y)&(out.response==1)]
        if len(sub): ax.scatter(sub.PC1,sub.PC2,s=8,alpha=.35,label=str(y))
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)");ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)");ax.set_title(f"{species.replace('_',' ').title()} occupied environmental space by period");ax.legend(ncol=2,fontsize=8)
    fig.tight_layout();fig.savefig(NICHE/f"{species}_environmental_space.png",dpi=180);plt.close(fig)
