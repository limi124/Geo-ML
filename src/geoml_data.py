"""Data access, raster alignment and split helpers for the Geo-ML tutorial."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject

PROJECT = Path(__file__).resolve().parents[1]
RAW = PROJECT / "data" / "raw"
PROCESSED = PROJECT / "data" / "processed"
TABLES = PROJECT / "outputs" / "tables"
FIGURES = PROJECT / "outputs" / "figures"
PREDICTIONS = PROJECT / "outputs" / "predictions"

TARGET = "GNLST.tif"
FEATURE_FILES = {
    "dem": "GNDEM.tif",
    "ndvi": "GNNDVI.tif",
    "tem": "GNTEM.tif",
    "precip": "precip_sum_2020.tif",
    "soil": "soil_moist_mean_2020.tif",
}
FEATURE_LABELS = {
    "dem": "高程",
    "ndvi": "NDVI",
    "tem": "气温",
    "precip": "降水",
    "soil": "土壤水分",
}
FEATURE_UNITS = {
    "dem": "原始像元值",
    "ndvi": "NDVI",
    "tem": "原始像元值",
    "precip": "原始像元值",
    "soil": "原始像元值",
}


def ensure_dirs() -> None:
    for p in (PROCESSED, TABLES, FIGURES, PREDICTIONS):
        p.mkdir(parents=True, exist_ok=True)


def read_band(path: Path | str, masked: bool = True) -> np.ndarray:
    with rasterio.open(path) as ds:
        return ds.read(1, masked=masked).astype("float32")


def target_grid():
    with rasterio.open(RAW / TARGET) as ds:
        return ds.transform, ds.crs, ds.height, ds.width, ds.nodata


def extent_of(transform: Affine, shape: tuple[int, int]) -> tuple[float, float, float, float]:
    h, w = shape
    left, top = transform * (0, 0)
    right, bottom = transform * (w, h)
    return (float(left), float(right), float(bottom), float(top))


def _fill_source_nan(arr: np.ndarray) -> np.ndarray:
    """Nearest-neighbour fill of a coarse source grid before resampling."""
    from scipy.ndimage import distance_transform_edt

    bad = ~np.isfinite(arr)
    if not bad.any():
        return arr
    idx = distance_transform_edt(bad, return_distances=False, return_indices=True)
    return arr[tuple(idx)]


def align_to_target(
    path: Path | str,
    *,
    resampling: Resampling = Resampling.bilinear,
    fill_coarse: bool = False,
    zero_as_nodata: bool = False,
) -> np.ndarray:
    """Warp one raster to the GNLST grid and return a float32 array with NaN NoData."""
    transform, crs, height, width, _ = target_grid()
    with rasterio.open(path) as ds:
        src = ds.read(1).astype("float64")
        if ds.nodata is not None:
            src[src == ds.nodata] = np.nan
        if zero_as_nodata:
            src[np.isclose(src, 0)] = np.nan
        src[~np.isfinite(src)] = np.nan
        if fill_coarse:
            src = _fill_source_nan(src)
        dst = np.full((height, width), np.nan, dtype="float32")
        reproject(
            source=src,
            destination=dst,
            src_transform=ds.transform,
            src_crs=ds.crs,
            dst_transform=transform,
            dst_crs=crs,
            src_nodata=np.nan,
            dst_nodata=np.nan,
            resampling=resampling,
        )
    return dst


def build_stack(save: bool = True) -> dict:
    """Align the five predictors and persist a compact stack bundle."""
    ensure_dirs()
    transform, crs, height, width, _ = target_grid()
    with rasterio.open(RAW / TARGET) as ds:
        target = ds.read(1).astype("float32")
        if ds.nodata is not None:
            target[target == ds.nodata] = np.nan
    arrays = {
        "dem": align_to_target(RAW / FEATURE_FILES["dem"], resampling=Resampling.bilinear),
        "ndvi": align_to_target(RAW / FEATURE_FILES["ndvi"], resampling=Resampling.bilinear),
        "tem": align_to_target(RAW / FEATURE_FILES["tem"], resampling=Resampling.bilinear),
        "precip": align_to_target(RAW / FEATURE_FILES["precip"], resampling=Resampling.nearest),
        "soil": align_to_target(RAW / FEATURE_FILES["soil"], resampling=Resampling.nearest),
    }
    valid = np.isfinite(target)
    bundle = {
        "target": target,
        "valid": valid,
        "transform": tuple(transform),
        "crs": crs.to_string(),
        "shape": (height, width),
        "extent": extent_of(transform, (height, width)),
        **arrays,
    }
    if save:
        np.savez_compressed(
            PROCESSED / "aligned_stack.npz",
            **{k: v for k, v in bundle.items() if k not in ("transform", "crs", "shape", "extent")},
            transform=np.array(bundle["transform"], dtype="float64"),
            crs=np.array(bundle["crs"]),
            shape=np.array(bundle["shape"]),
            extent=np.array(bundle["extent"], dtype="float64"),
        )
        write_stack_geotiffs(bundle)
    return bundle


def write_stack_geotiffs(bundle: dict) -> None:
    """Write aligned target and a five-band predictor GeoTIFF for GIS inspection."""
    ensure_dirs()
    transform = Affine(*bundle["transform"][:6])
    crs = bundle["crs"]
    h, w = bundle["shape"]
    profile = dict(
        driver="GTiff", height=h, width=w, count=1, dtype="float32",
        crs=crs, transform=transform, nodata=np.nan, compress="deflate",
        tiled=True, blockxsize=64, blockysize=64,
    )
    with rasterio.open(PROCESSED / "target_aligned.tif", "w", **profile) as ds:
        ds.write(bundle["target"].astype("float32"), 1)
        ds.set_band_description(1, "LST (raw pixel value)")
    profile.update(count=len(FEATURE_FILES))
    with rasterio.open(PROCESSED / "features_aligned.tif", "w", **profile) as ds:
        for i, (key, fname) in enumerate(FEATURE_FILES.items(), start=1):
            ds.write(bundle[key].astype("float32"), i)
            ds.set_band_description(i, key)
            ds.set_band_unit(i, FEATURE_UNITS[key])


def load_stack() -> dict:
    path = PROCESSED / "aligned_stack.npz"
    if not path.exists():
        return build_stack(save=True)
    with np.load(path, allow_pickle=False) as z:
        bundle = {k: z[k] for k in z.files}
    bundle["transform"] = tuple(float(v) for v in bundle["transform"])
    bundle["crs"] = str(bundle["crs"])
    bundle["shape"] = tuple(int(v) for v in bundle["shape"])
    bundle["extent"] = tuple(float(v) for v in bundle["extent"])
    bundle["valid"] = bundle["valid"].astype(bool)
    return bundle


def build_training_table(bundle: dict | None = None) -> pd.DataFrame:
    """One row per valid target pixel, keeping grid location and centre coordinates."""
    bundle = load_stack() if bundle is None else bundle
    transform = Affine(*bundle["transform"][:6])
    valid = bundle["valid"]
    rows, cols = np.nonzero(valid)
    xs, ys = rasterio.transform.xy(transform, rows, cols, offset="center")
    data = {
        "row": rows.astype(int),
        "col": cols.astype(int),
        "x": np.asarray(xs, dtype="float64"),
        "y": np.asarray(ys, dtype="float64"),
        "lst": bundle["target"][valid].astype("float64"),
    }
    for key in FEATURE_FILES:
        data[key] = bundle[key][valid].astype("float64")
    df = pd.DataFrame(data)
    ensure_dirs()
    df.to_csv(PROCESSED / "training_table.csv", index=False, encoding="utf-8-sig")
    return df


def add_splits(
    df: pd.DataFrame,
    n_folds: int = 5,
    block_rows: int = 6,
    block_cols: int = 6,
    seed: int = 42,
    test_block: tuple[int, int, int, int] = (55, 90, 50, 90),
) -> pd.DataFrame:
    """Hold out one contiguous south-central block and build spatial CV folds.

    ``test_block`` is ``(row_min, row_max, col_min, col_max)`` in the 176×106
    target grid. The block is excluded from every training/CV step. Remaining
    pixels are tiled into regular row/column blocks; whole blocks are assigned to
    folds by a seeded, count-balanced greedy rule.
    """
    out = df.copy()
    r0, r1, c0, c1 = test_block
    is_test = out["row"].between(r0, r1) & out["col"].between(c0, c1)
    out["split"] = np.where(is_test, "test", "train")
    out["fold"] = -1

    train = out["split"].eq("train")
    r = out.loc[train, "row"].to_numpy()
    c = out.loc[train, "col"].to_numpy()
    r_min, r_max = int(out["row"].min()), int(out["row"].max()) + 1
    c_min, c_max = int(out["col"].min()), int(out["col"].max()) + 1
    r_edges = np.linspace(r_min, r_max, block_rows + 1).astype(int)
    c_edges = np.linspace(c_min, c_max, block_cols + 1).astype(int)
    br = np.clip(np.searchsorted(r_edges, r, side="right") - 1, 0, block_rows - 1)
    bc = np.clip(np.searchsorted(c_edges, c, side="right") - 1, 0, block_cols - 1)
    block_id = br * block_cols + bc
    counts = pd.Series(block_id).value_counts().sort_index()
    rng = np.random.default_rng(seed)
    order = rng.permutation(counts.index.to_numpy())
    load = np.zeros(n_folds, dtype=int)
    assign = {}
    for b in order:
        f = int(np.argmin(load))
        assign[int(b)] = f
        load[f] += int(counts.loc[b])
    out.loc[train, "fold"] = [assign[int(b)] for b in block_id]
    out["block"] = -1
    out.loc[train, "block"] = block_id
    return out


def split_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, part in df.groupby("split", sort=False):
        rows.append({"划分": "独立测试区（中南连续区块）" if name == "test" else "训练区",
                     "像元数": len(part), "占比": len(part) / len(df)})
    for f, part in df[df["split"].eq("train")].groupby("fold"):
        rows.append({"划分": f"空间折 {f + 1}", "像元数": len(part), "占比": len(part) / len(df)})
    return pd.DataFrame(rows)


def load_splits() -> pd.DataFrame:
    path = PROCESSED / "training_table_splits.csv"
    if not path.exists():
        df = add_splits(build_training_table())
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return df
    return pd.read_csv(path)


def write_prediction(array: np.ndarray, out_name: str, mask_nodata: float = -9999.0) -> Path:
    """Write a full-grid prediction as a GeoTIFF with an explicit NoData value."""
    ensure_dirs()
    transform, crs, h, w, _ = target_grid()
    arr = np.asarray(array, dtype="float32").copy()
    arr[~np.isfinite(arr)] = mask_nodata
    profile = dict(
        driver="GTiff", height=h, width=w, count=1, dtype="float32",
        crs=crs, transform=transform, nodata=mask_nodata, compress="deflate",
    )
    path = PREDICTIONS / out_name
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(arr, 1)
        ds.set_band_description(1, "prediction")
    return path


def write_json(obj: dict, name: str) -> Path:
    ensure_dirs()
    path = TABLES / name
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def test_window(df: pd.DataFrame, bundle: dict | None = None) -> dict:
    """Return the tight rectangular window around the fixed test block.

    The returned ``extent`` is ready for ``imshow``; the arrays align with
    ``row - row0`` and ``col - col0``.
    """
    bundle = load_stack() if bundle is None else bundle
    test = df[df["split"].eq("test")]
    r0, r1 = int(test["row"].min()), int(test["row"].max())
    c0, c1 = int(test["col"].min()), int(test["col"].max())
    transform = Affine(*bundle["transform"][:6])
    left, top = transform * (c0, r0)
    right, bottom = transform * (c1 + 1, r1 + 1)
    obs = np.full((r1 - r0 + 1, c1 - c0 + 1), np.nan, dtype=float)
    obs[test["row"].to_numpy() - r0, test["col"].to_numpy() - c0] = test["lst"].to_numpy()
    return {
        "row0": r0, "row1": r1, "col0": c0, "col1": c1,
        "shape": obs.shape,
        "extent": (float(left), float(right), float(bottom), float(top)),
        "obs": obs,
    }


def grid_from_predictions(window: dict, pred) -> np.ndarray:
    """Place a vector of test-block predictions back on the cropped grid."""
    grid = np.full(window["shape"], np.nan, dtype=float)
    test_mask = np.isfinite(window["obs"])
    grid[test_mask] = np.asarray(pred, dtype=float)
    return grid
