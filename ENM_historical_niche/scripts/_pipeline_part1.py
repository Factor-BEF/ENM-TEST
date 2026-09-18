from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import rasterio
from rasterio.features import rasterize
from rasterio.io import MemoryFile
from rasterio.transform import from_origin, xy
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.warp import reproject
from shapely.geometry import Point, shape
from shapely.prepared import prep
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "config.json"
RAW = ROOT / "data_raw"
PROC = ROOT / "data_processed"
RESULTS = ROOT / "results"
MAPS = RESULTS / "maps"
EVAL = RESULTS / "evaluation"
VARSEL = RESULTS / "variable_selection"
NICHE = RESULTS / "niche_space"
MODELS = ROOT / "models"
ODMAP = ROOT / "odmap"
LOGS = ROOT / "logs"

for d in [RAW, PROC, RESULTS, MAPS, EVAL, VARSEL, NICHE, MODELS, ODMAP, LOGS]:
    d.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    with (LOGS / "pipeline.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


CFG = load_config()
YEARS = [int(x) for x in CFG["target_years"]]
WINDOWS = {int(k): tuple(v) for k, v in CFG["period_windows"].items()}
XMIN, YMIN, XMAX, YMAX = map(float, CFG["extent"])
RES = float(CFG["resolution_deg"])
WIDTH = int(math.ceil((XMAX - XMIN) / RES))
HEIGHT = int(math.ceil((YMAX - YMIN) / RES))
TRANSFORM = from_origin(XMIN, YMAX, RES, RES)
CRS = CFG["crs"]
RNG = np.random.default_rng(int(CFG["seed"]))
BIO_NAMES = [f"bio{i}" for i in range(1, 20)]
ALL_PREDICTORS = BIO_NAMES + ["ghm"]


def sha256(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def download(url: str, path: Path, min_bytes: int = 1000, retries: int = 4) -> Path:
    """Robust segmented/resumable downloader for large ecological archives.

    GitHub-hosted runner benchmarking on 2026-09-18 showed that the WorldClim
    server supports HTTP range requests: aria2c with eight connections fetched
    the 126-MB 1990-1999 10-arcmin tmin archive in ~3 s and passed `unzip -t`.
    We therefore prefer aria2c over a single curl connection. Partial downloads
    are retained and resumed across attempts within a run.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size >= min_bytes:
        return path
    tmp = path.with_suffix(path.suffix + ".part")
    last = None
    for attempt in range(1, retries + 1):
        try:
            log(f"download {url} -> {path.name} (attempt {attempt}; segmented; resume={tmp.exists()})")
            if shutil.which("aria2c"):
                cmd = [
                    "aria2c", "-x", "8", "-s", "8", "-k", "1M",
                    "--continue=true", "--file-allocation=none",
                    "--connect-timeout=120", "--timeout=120",
                    "--max-tries=5", "--retry-wait=3",
                    "--allow-overwrite=true", "--auto-file-renaming=false",
                    "--dir", str(tmp.parent), "--out", tmp.name, url,
                ]
                p = subprocess.run(cmd, text=True, capture_output=True, timeout=1800)
                if p.returncode != 0:
                    raise RuntimeError(f"aria2c exit {p.returncode}: {p.stderr[-1200:]} {p.stdout[-1200:]}")
            else:
                # Portable fallback if aria2 is absent.
                cmd = [
                    "curl", "-fL", "--http1.1", "--retry", "5",
                    "--retry-delay", "5", "--retry-all-errors",
                    "--connect-timeout", "120", "--max-time", "1800",
                    "--continue-at", "-", "--output", str(tmp), url,
                ]
                p = subprocess.run(cmd, text=True, capture_output=True, timeout=1850)
                if p.returncode != 0:
                    if p.returncode == 33 and tmp.exists():
                        tmp.unlink()
                    raise RuntimeError(f"curl exit {p.returncode}: {p.stderr[-1200:]}")
            if not tmp.exists() or tmp.stat().st_size < min_bytes:
                raise RuntimeError(f"download too small: {tmp.stat().st_size if tmp.exists() else 0}")
            tmp.replace(path)
            log(f"download complete {path.name}: {path.stat().st_size/1e6:.1f} MB")
            return path
        except Exception as e:
            last = e
            log(f"download failed: {e}")
            time.sleep(attempt * 3)
    raise RuntimeError(f"Failed to download {url}: {last}")


def write_raster(path: Path, arr: np.ndarray, dtype: str = "float32", nodata: float | int = -9999,
                 descriptions: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    profile = {
        "driver": "GTiff", "height": HEIGHT, "width": WIDTH, "count": arr.shape[0],
        "dtype": dtype, "crs": CRS, "transform": TRANSFORM, "nodata": nodata,
        "compress": "DEFLATE", "predictor": 2 if "float" in dtype else 1,
        "tiled": True, "blockxsize": 256, "blockysize": 256,
    }
    out = np.asarray(arr).copy()
    if np.issubdtype(np.dtype(dtype), np.floating):
        out = out.astype(dtype)
        out[~np.isfinite(out)] = nodata
    else:
        out = out.astype(dtype)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(out)
        if descriptions:
            for i, desc in enumerate(descriptions, start=1):
                dst.set_band_description(i, desc)


def read_raster(path: Path) -> Tuple[np.ndarray, List[str]]:
    with rasterio.open(path) as src:
        a = src.read().astype("float32")
        nd = src.nodata
        if nd is not None:
            a[a == nd] = np.nan
        desc = [d or f"band{i}" for i, d in enumerate(src.descriptions, start=1)]
    return a, desc


def get_canada_geometry() -> Any:
    cache = RAW / "countries.geojson"
    url = "https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson"
    download(url, cache, min_bytes=100000)
    obj = json.loads(cache.read_text(encoding="utf-8"))
    matches = []
    for feat in obj["features"]:
        vals = [str(v).lower() for v in feat.get("properties", {}).values() if v is not None]
        if "canada" in vals:
            matches.append(shape(feat["geometry"]))
    if not matches:
        raise RuntimeError("Canada geometry not found in geo-countries")
    return matches[0]


def make_canada_mask() -> np.ndarray:
    out = PROC / "canada_mask.tif"
    if out.exists():
        arr, _ = read_raster(out)
        return arr[0] == 1
    geom = get_canada_geometry()
    mask = rasterize([(geom, 1)], out_shape=(HEIGHT, WIDTH), transform=TRANSFORM,
                     fill=0, dtype="uint8", all_touched=False).astype(bool)
    write_raster(out, mask.astype("uint8"), dtype="uint8", nodata=255, descriptions=["Canada mask"])
    return mask


CANADA_MASK = make_canada_mask()
CANADA_GEOM = get_canada_geometry()
CANADA_PREP = prep(CANADA_GEOM)


def reproject_memfile_to_grid(data: bytes, resampling: Resampling = Resampling.bilinear) -> np.ndarray:
    with MemoryFile(data) as mem:
        with mem.open() as src:
            dest = np.full((HEIGHT, WIDTH), np.nan, dtype="float32")
            reproject(
                source=rasterio.band(src, 1), destination=dest,
                src_transform=src.transform, src_crs=src.crs,
                src_nodata=src.nodata, dst_transform=TRANSFORM, dst_crs=CRS,
                dst_nodata=np.nan, resampling=resampling,
            )
    dest[~CANADA_MASK] = np.nan
    return dest


def detect_temp_scale(a: np.ndarray) -> float:
    q = np.nanpercentile(np.abs(a[CANADA_MASK]), 95)
    return 0.1 if q > 100 else 1.0


def rolling_quarters(x: np.ndarray, op: str = "mean") -> np.ndarray:
    qs = []
    for m in range(12):
        stack = np.stack([x[m], x[(m + 1) % 12], x[(m + 2) % 12]], axis=0)
        qs.append(np.nansum(stack, axis=0) if op == "sum" else np.nanmean(stack, axis=0))
    return np.stack(qs, axis=0)


def pick_by_index(values: np.ndarray, idx: np.ndarray) -> np.ndarray:
    flat_v = values.reshape(values.shape[0], -1)
    flat_i = idx.ravel()
    cols = np.arange(flat_i.size)
    out = flat_v[flat_i, cols]
    return out.reshape(idx.shape)


def compute_bioclim(tmin: np.ndarray, tmax: np.ndarray, prec: np.ndarray) -> np.ndarray:
    tmean = (tmin + tmax) / 2.0
    dtr = tmax - tmin
    bio1 = np.nanmean(tmean, axis=0)
    bio2 = np.nanmean(dtr, axis=0)
    bio5 = np.nanmax(tmax, axis=0)
    bio6 = np.nanmin(tmin, axis=0)
    bio7 = bio5 - bio6
    bio3 = np.where(np.abs(bio7) > 1e-8, 100.0 * bio2 / bio7, np.nan)
    bio4 = np.nanstd(tmean, axis=0, ddof=1) * 100.0
    tq = rolling_quarters(tmean, "mean")
    pq = rolling_quarters(prec, "sum")
    wet_q = np.nanargmax(np.where(np.isfinite(pq), pq, -np.inf), axis=0)
    dry_q = np.nanargmin(np.where(np.isfinite(pq), pq, np.inf), axis=0)
    warm_q = np.nanargmax(np.where(np.isfinite(tq), tq, -np.inf), axis=0)
    cold_q = np.nanargmin(np.where(np.isfinite(tq), tq, np.inf), axis=0)
    bio8 = pick_by_index(tq, wet_q)
    bio9 = pick_by_index(tq, dry_q)
    bio10 = pick_by_index(tq, warm_q)
    bio11 = pick_by_index(tq, cold_q)
    bio12 = np.nansum(prec, axis=0)
    bio13 = np.nanmax(prec, axis=0)
    bio14 = np.nanmin(prec, axis=0)
    pmean = np.nanmean(prec, axis=0)
    bio15 = np.where(pmean > 1e-8, 100.0 * np.nanstd(prec, axis=0, ddof=1) / pmean, np.nan)
    bio16 = pick_by_index(pq, wet_q)
    bio17 = pick_by_index(pq, dry_q)
    bio18 = pick_by_index(pq, warm_q)
    bio19 = pick_by_index(pq, cold_q)
    out = np.stack([bio1,bio2,bio3,bio4,bio5,bio6,bio7,bio8,bio9,bio10,bio11,
                    bio12,bio13,bio14,bio15,bio16,bio17,bio18,bio19]).astype("float32")
    out[:, ~CANADA_MASK] = np.nan
    return out
