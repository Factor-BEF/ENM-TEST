def prepare_worldclim() -> None:
    done = all((PROC / f"worldclim_bio_{y}.tif").exists() for y in YEARS)
    if done:
        log("WorldClim historical BIO rasters already present")
        return
    sums: Dict[str, Dict[int, List[np.ndarray]]] = {
        v: {y: [np.zeros((HEIGHT, WIDTH), dtype="float64") for _ in range(12)] for y in YEARS}
        for v in ["tmin", "tmax", "prec"]
    }
    counts = {v: {y: np.zeros(12, dtype=int) for y in YEARS} for v in ["tmin", "tmax", "prec"]}
    decades = ["1980-1989", "1990-1999", "2000-2009", "2010-2019", "2020-2024"]
    base = CFG["worldclim_base_url"]
    member_re = re.compile(r"(19|20)\d{2}[-_](0[1-9]|1[0-2])\.tif$", re.I)
    for var in ["tmin", "tmax", "prec"]:
        for decade in decades:
            url = f"{base}/wc2.1_cruts4.09_{CFG['worldclim_resolution']}_{var}_{decade}.zip"
            zpath = RAW / "worldclim" / Path(url).name
            download(url, zpath, min_bytes=100000)
            log(f"processing WorldClim {var} {decade} ({zpath.stat().st_size/1e6:.1f} MB)")
            with zipfile.ZipFile(zpath) as z:
                names = [n for n in z.namelist() if n.lower().endswith(".tif")]
                for name in names:
                    m = member_re.search(Path(name).name)
                    if not m:
                        continue
                    ymatch = re.search(r"((?:19|20)\d{2})[-_](0[1-9]|1[0-2])\.tif$", Path(name).name, re.I)
                    if not ymatch:
                        continue
                    yr = int(ymatch.group(1)); mo = int(ymatch.group(2))
                    targets = [ty for ty, (a,b) in WINDOWS.items() if a <= yr <= b]
                    if not targets:
                        continue
                    arr = reproject_memfile_to_grid(z.read(name), Resampling.bilinear)
                    if var in ("tmin", "tmax"):
                        arr *= detect_temp_scale(arr)
                    for ty in targets:
                        good = np.isfinite(arr)
                        sums[var][ty][mo-1][good] += arr[good]
                        counts[var][ty][mo-1] += 1
            zpath.unlink(missing_ok=True)
    for y in YEARS:
        monthly = {}
        for var in ["tmin", "tmax", "prec"]:
            layers = []
            for m in range(12):
                c = counts[var][y][m]
                if c == 0:
                    raise RuntimeError(f"Missing WorldClim months for {var} {y} month {m+1}")
                a = (sums[var][y][m] / c).astype("float32")
                a[~CANADA_MASK] = np.nan
                layers.append(a)
            monthly[var] = np.stack(layers)
        bio = compute_bioclim(monthly["tmin"], monthly["tmax"], monthly["prec"])
        out = PROC / f"worldclim_bio_{y}.tif"
        write_raster(out, bio, descriptions=BIO_NAMES)
        log(f"wrote {out.name}")


def ghm_remote_url(year: int) -> str:
    return f"https://zenodo.org/records/{CFG['ghm_record']}/files/HMv20240801_{year}c_AA.tif?download=1"


def prepare_ghm() -> None:
    for y in YEARS:
        out = PROC / f"ghm_{y}.tif"
        if out.exists():
            continue
        url = ghm_remote_url(y)
        log(f"reading remote GHM COG {y} from Zenodo")
        vsi = "/vsicurl/" + url
        env_opts = {
            "GDAL_HTTP_MULTIRANGE": "YES",
            "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
            "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
            "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
        }
        with rasterio.Env(**env_opts):
            with rasterio.open(vsi) as src:
                with WarpedVRT(src, crs=CRS, transform=TRANSFORM, width=WIDTH, height=HEIGHT,
                               resampling=Resampling.average, nodata=np.nan) as vrt:
                    arr = vrt.read(1, masked=True).filled(np.nan).astype("float32")
        arr[~CANADA_MASK] = np.nan
        arr[(arr < 0) | (arr > 1.01)] = np.nan
        write_raster(out, arr, descriptions=[f"GHM overall {y}"])
        log(f"wrote {out.name}")


def find_tif_in_zip(z: zipfile.ZipFile) -> str:
    tifs = [n for n in z.namelist() if n.lower().endswith((".tif", ".tiff"))]
    if not tifs:
        raise RuntimeError("No TIFF in black-spruce zip")
    tifs.sort(key=lambda n: z.getinfo(n).file_size, reverse=True)
    return tifs[0]


def prepare_black_spruce() -> None:
    base = CFG["black_spruce_base_url"]
    cls = int(CFG["black_spruce_class"])
    for y in YEARS:
        out = PROC / f"black_spruce_observed_{y}.tif"
        if out.exists():
            continue
        url = f"{base}/CA_Tree_Species_Classification_{y}.zip"
        zpath = RAW / "black_spruce" / Path(url).name
        download(url, zpath, min_bytes=100000)
        log(f"processing annual tree species {y} ({zpath.stat().st_size/1e6:.1f} MB)")
        extract_dir = RAW / "black_spruce" / f"tmp_{y}"
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zpath) as z:
            tif_name = find_tif_in_zip(z)
            z.extract(tif_name, extract_dir)
        tif_path = extract_dir / tif_name
        with rasterio.open(tif_path) as src:
            with WarpedVRT(src, crs=CRS, transform=TRANSFORM, width=WIDTH, height=HEIGHT,
                           resampling=Resampling.mode, nodata=src.nodata) as vrt:
                ma = vrt.read(1, masked=True)
                data = ma.filled(-9999)
                valid = np.logical_not(np.ma.getmaskarray(ma)) & CANADA_MASK
        obs = np.full((HEIGHT, WIDTH), 255, dtype="uint8")
        obs[valid] = (data[valid] == cls).astype("uint8")
        write_raster(out, obs, dtype="uint8", nodata=255, descriptions=[f"Picea mariana observed {y}"])
        shutil.rmtree(extract_dir, ignore_errors=True)
        zpath.unlink(missing_ok=True)
        log(f"wrote {out.name}: presence cells={(obs==1).sum()}, valid={(obs!=255).sum()}")


def gbif_fetch_period(start: int, end: int) -> pd.DataFrame:
    url = "https://api.gbif.org/v1/occurrence/search"
    rows = []
    offset = 0
    limit = 300
    while True:
        params = {
            "taxonKey": CFG["red_fox_taxon_key"], "country": CFG["gbif_country"],
            "year": f"{start},{end}", "hasCoordinate": "true",
            "hasGeospatialIssue": "false", "occurrenceStatus": "PRESENT",
            "limit": limit, "offset": offset,
        }
        r = requests.get(url, params=params, timeout=120)
        r.raise_for_status()
        js = r.json()
        res = js.get("results", [])
        rows.extend(res)
        if js.get("endOfRecords", False) or not res:
            break
        offset += len(res)
        if offset >= 100000:
            log(f"GBIF period {start}-{end} reached API pagination cap; stopping at {offset}")
            break
    keep = []
    for rec in rows:
        lon = rec.get("decimalLongitude"); lat = rec.get("decimalLatitude")
        if lon is None or lat is None:
            continue
        try:
            lon = float(lon); lat = float(lat)
        except Exception:
            continue
        if not (XMIN <= lon <= XMAX and YMIN <= lat <= YMAX):
            continue
        if abs(lon) < 1e-8 and abs(lat) < 1e-8:
            continue
        if not CANADA_PREP.covers(Point(lon, lat)):
            continue
        unc = rec.get("coordinateUncertaintyInMeters")
        if unc is not None:
            try:
                if float(unc) > float(CFG["max_coordinate_uncertainty_m"]):
                    continue
            except Exception:
                pass
        bor = str(rec.get("basisOfRecord", ""))
        if bor.upper() == "FOSSIL_SPECIMEN":
            continue
        keep.append({
            "gbifID": rec.get("key") or rec.get("gbifID"), "lon": lon, "lat": lat,
            "year": rec.get("year"), "basisOfRecord": bor,
            "coordinateUncertaintyInMeters": unc,
            "datasetKey": rec.get("datasetKey"), "scientificName": rec.get("scientificName"),
        })
    return pd.DataFrame(keep)


def point_to_rc(lon: np.ndarray, lat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    col = np.floor((lon - XMIN) / RES).astype(int)
    row = np.floor((YMAX - lat) / RES).astype(int)
    return row, col


def prepare_red_fox() -> None:
    all_rows = []
    for ty in YEARS:
        out = PROC / f"red_fox_occurrences_{ty}.csv"
        if out.exists():
            df = pd.read_csv(out)
        else:
            a,b = WINDOWS[ty]
            log(f"query GBIF Vulpes vulpes Canada {a}-{b}")
            df = gbif_fetch_period(a,b)
            if df.empty:
                raise RuntimeError(f"No red fox records for {a}-{b}")
            r,c = point_to_rc(df["lon"].to_numpy(), df["lat"].to_numpy())
            good = (r>=0)&(r<HEIGHT)&(c>=0)&(c<WIDTH)
            df = df.loc[good].copy(); r=r[good]; c=c[good]
            df["row"] = r; df["col"] = c
            df = df.drop_duplicates(subset=["row","col"]).sort_values(["row","col"]).reset_index(drop=True)
            df["period"] = ty
            df.to_csv(out, index=False)
            log(f"clean red fox {ty}: {len(df)} unique 10-arcmin cells")
        all_rows.append(df)
    pd.concat(all_rows, ignore_index=True).to_csv(PROC / "red_fox_occurrences_all.csv", index=False)


def load_predictor_stack(year: int) -> Dict[str, np.ndarray]:
    bio, desc = read_raster(PROC / f"worldclim_bio_{year}.tif")
    ghm, _ = read_raster(PROC / f"ghm_{year}.tif")
    d = {name: bio[i] for i,name in enumerate(BIO_NAMES)}
    d["ghm"] = ghm[0]
    return d


def grid_lonlat(rows: np.ndarray, cols: np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
    xs, ys = xy(TRANSFORM, rows, cols, offset="center")
    return np.asarray(xs), np.asarray(ys)


def spatial_group(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    b = float(CFG["spatial_block_deg"])
    gx = np.floor((lon + 180) / b).astype(int)
    gy = np.floor((lat + 90) / b).astype(int)
    return gx + 1000 * gy


def build_black_spruce_table() -> pd.DataFrame:
    out = PROC / "model_table_black_spruce.csv.gz"
    if out.exists():
        return pd.read_csv(out)
    rows_all = []
    for y in YEARS:
        pred = load_predictor_stack(y)
        obs, _ = read_raster(PROC / f"black_spruce_observed_{y}.tif")
        resp = obs[0]
        valid = CANADA_MASK & np.isfinite(resp)
        for v in ALL_PREDICTORS:
            valid &= np.isfinite(pred[v])
        pres = np.flatnonzero(valid & (resp == 1))
        absn = np.flatnonzero(valid & (resp == 0))
        npres = min(len(pres), int(CFG["max_presence_per_period_black_spruce"]))
        nabs = min(len(absn), int(CFG["max_absence_per_period_black_spruce"]))
        pres = RNG.choice(pres, size=npres, replace=False) if len(pres)>npres else pres
        absn = RNG.choice(absn, size=nabs, replace=False) if len(absn)>nabs else absn
        idx = np.concatenate([pres, absn]); yy = np.r_[np.ones(len(pres),dtype=int), np.zeros(len(absn),dtype=int)]
        rr, cc = np.unravel_index(idx, (HEIGHT,WIDTH)); lon,lat = grid_lonlat(rr,cc)
        dat = {"response":yy,"period":y,"row":rr,"col":cc,"lon":lon,"lat":lat}
        for v in ALL_PREDICTORS:
            dat[v] = pred[v].ravel()[idx]
        df = pd.DataFrame(dat); df["group"] = spatial_group(lon,lat)
        rows_all.append(df)
        log(f"black spruce modeling sample {y}: presence={len(pres)} absence={len(absn)}")
    ans = pd.concat(rows_all, ignore_index=True)
    ans.to_csv(out, index=False, compression="gzip")
    return ans


def build_red_fox_table() -> pd.DataFrame:
    out = PROC / "model_table_red_fox.csv.gz"
    if out.exists():
        return pd.read_csv(out)
    rows_all=[]
    for y in YEARS:
        pred = load_predictor_stack(y)
        occ = pd.read_csv(PROC / f"red_fox_occurrences_{y}.csv")
        pr = occ["row"].astype(int).to_numpy(); pc=occ["col"].astype(int).to_numpy()
        validp = (pr>=0)&(pr<HEIGHT)&(pc>=0)&(pc<WIDTH)
        pr=pr[validp]; pc=pc[validp]
        pidx = np.ravel_multi_index((pr,pc),(HEIGHT,WIDTH))
        valid = CANADA_MASK.copy()
        for v in ALL_PREDICTORS: valid &= np.isfinite(pred[v])
        valid.ravel()[pidx] = False
        bg_idx = np.flatnonzero(valid)
        nbg = min(len(bg_idx), max(1000, min(int(CFG["max_background_per_period_red_fox"]),
                                              int(CFG["background_multiplier_red_fox"])*len(pidx))))
        bg = RNG.choice(bg_idx, size=nbg, replace=False)
        idx=np.concatenate([pidx,bg]); yy=np.r_[np.ones(len(pidx),dtype=int),np.zeros(len(bg),dtype=int)]
        rr,cc=np.unravel_index(idx,(HEIGHT,WIDTH)); lon,lat=grid_lonlat(rr,cc)
        dat={"response":yy,"period":y,"row":rr,"col":cc,"lon":lon,"lat":lat}
        for v in ALL_PREDICTORS: dat[v]=pred[v].ravel()[idx]
        df=pd.DataFrame(dat); df["group"]=spatial_group(lon,lat)
        rows_all.append(df)
        log(f"red fox modeling sample {y}: presence={len(pidx)} background={len(bg)}")
    ans=pd.concat(rows_all,ignore_index=True)
    ans.to_csv(out,index=False,compression="gzip")
    return ans
