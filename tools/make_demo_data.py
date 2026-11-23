"""Generate a self-contained synthetic demo dataset in data/raw/ for running Geo-ML.

The generated files mimic the spatial layout, CRS, resolutions, and NoData
conventions expected by the 8 tutorial notebooks, using purely synthetic
mathematical fields (not real observations).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from scipy.ndimage import gaussian_filter
from shapely.geometry import Polygon

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def _smooth_field(shape: tuple[int, int], sigma: float, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(0.0, 1.0, size=shape)
    field = gaussian_filter(noise, sigma=sigma, mode="reflect")
    std = field.std()
    return (field - field.mean()) / (std if std > 1e-8 else 1.0)


def _synthetic_boundary() -> gpd.GeoDataFrame:
    # Approximate GanNan extent (122.90-124.47E, 47.59-48.53N) with a smooth organic polygon
    cx, cy = 123.68, 48.06
    angles = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    rx = 0.68 + 0.08 * np.cos(2 * angles) + 0.04 * np.sin(3 * angles)
    ry = 0.41 + 0.05 * np.sin(2 * angles) - 0.03 * np.cos(3 * angles)
    coords = [(float(cx + rx[i] * np.cos(a)), float(cy + ry[i] * np.sin(a))) for i, a in enumerate(angles)]
    poly = Polygon(coords)
    return gpd.GeoDataFrame(
        {"name": ["甘南县（合成示例）"], "gb": [156230225]},
        geometry=[poly],
        crs="EPSG:4326",
    )


def _write_tif(
    path: Path,
    arr: np.ndarray,
    transform,
    crs: str,
    nodata: float | None,
    dtype: str = "float32",
) -> None:
    profile = {
        "driver": "GTiff",
        "height": arr.shape[0],
        "width": arr.shape[1],
        "count": 1,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "compress": "deflate",
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(arr.astype(dtype), 1)


def main(overwrite: bool = False) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    target_files = [
        "GNLST.tif", "GNDEM.tif", "GNNDVI.tif", "GNTEM.tif",
        "precip_sum_2020.tif", "soil_moist_mean_2020.tif", "GanNan.shp",
    ]
    if not overwrite and any((RAW / f).exists() for f in target_files):
        raise FileExistsError(
            "data/raw/ already contains dataset files. Pass --overwrite to replace them."
        )

    rng = np.random.default_rng(42)
    gdf = _synthetic_boundary()
    gdf.to_file(RAW / "GanNan.shp", encoding="utf-8")

    # 1. Reference 176x106 grid (~0.008983 deg)
    res_lst = 0.008983152841195215
    h_lst, w_lst = 106, 176
    west, north = 122.895, 48.535
    tf_lst = from_origin(west, north, res_lst, res_lst)

    inside_lst = ~geometry_mask(gdf.geometry, out_shape=(h_lst, w_lst), transform=tf_lst, invert=False)
    yy, xx = np.mgrid[0:h_lst, 0:w_lst]
    y_norm = (yy - h_lst / 2) / h_lst
    x_norm = (xx - w_lst / 2) / w_lst

    dem_base = 260.0 - 95.0 * x_norm - 45.0 * y_norm + 35.0 * _smooth_field((h_lst, w_lst), 8.0, rng)
    tem_base = 4.2 + 0.9 * x_norm + 0.5 * y_norm - 0.005 * (dem_base - 240.0) + 0.35 * _smooth_field((h_lst, w_lst), 10.0, rng)
    ndvi_base = np.clip(0.55 - 0.08 * x_norm + 0.12 * _smooth_field((h_lst, w_lst), 6.0, rng), 0.15, 0.88)
    precip_base = 460.0 - 35.0 * x_norm + 25.0 * _smooth_field((h_lst, w_lst), 14.0, rng)
    soil_base = 0.28 + 0.04 * _smooth_field((h_lst, w_lst), 16.0, rng)

    # Synthetic LST driven by smooth spatial factors + non-linear interactions + local noise
    lst_vals = (
        11.8
        + 1.35 * (tem_base - tem_base.mean())
        - 0.012 * (dem_base - dem_base.mean())
        - 2.2 * (ndvi_base - ndvi_base.mean())
        + 0.45 * np.tanh((tem_base - 4.0) * 1.5)
        + 0.55 * _smooth_field((h_lst, w_lst), 5.0, rng)
        + rng.normal(0.0, 0.35, size=(h_lst, w_lst))
    )
    nodata_f32 = float(np.finfo("float32").min)
    lst_out = np.where(inside_lst, lst_vals, nodata_f32).astype("float32")
    _write_tif(RAW / "GNLST.tif", lst_out, tf_lst, "EPSG:4326", nodata_f32)

    # 2. GNTEM.tif (same resolution, slightly smaller valid mask so ~1% edges are missing)
    inside_tem = inside_lst.copy()
    inside_tem[:, :2] = False
    inside_tem[-2:, :] = False
    tem_out = np.where(inside_tem, tem_base, nodata_f32).astype("float32")
    _write_tif(RAW / "GNTEM.tif", tem_out, tf_lst, "EPSG:4326", nodata_f32)

    # 3. GNNDVI.tif (~4x finer grid, no declared nodata, 0 outside boundary)
    scale_ndvi = 4
    h_ndvi, w_ndvi = h_lst * scale_ndvi, w_lst * scale_ndvi
    tf_ndvi = from_origin(west, north, res_lst / scale_ndvi, res_lst / scale_ndvi)
    inside_ndvi = ~geometry_mask(gdf.geometry, out_shape=(h_ndvi, w_ndvi), transform=tf_ndvi, invert=False)
    ndvi_hi = np.repeat(np.repeat(ndvi_base, scale_ndvi, axis=0), scale_ndvi, axis=1)
    ndvi_hi = np.clip(ndvi_hi + 0.03 * _smooth_field((h_ndvi, w_ndvi), 3.0, rng), 0.05, 0.95)
    ndvi_out = np.where(inside_ndvi, ndvi_hi, 0.0).astype("float32")
    _write_tif(RAW / "GNNDVI.tif", ndvi_out, tf_ndvi, "EPSG:4326", None)

    # 4. GNDEM.tif (UTM Zone 52N, ~250m demo resolution to keep generation fast)
    gdf_utm = gdf.to_crs(32652)
    minx, miny, maxx, maxy = gdf_utm.total_bounds
    res_dem = 250.0
    w_dem = int(np.ceil((maxx - minx) / res_dem))
    h_dem = int(np.ceil((maxy - miny) / res_dem))
    tf_dem = from_origin(minx, maxy, res_dem, res_dem)
    inside_dem = ~geometry_mask(gdf_utm.geometry, out_shape=(h_dem, w_dem), transform=tf_dem, invert=False)
    yy_d, xx_d = np.mgrid[0:h_dem, 0:w_dem]
    dem_hi = (
        260.0
        - 95.0 * ((xx_d - w_dem / 2) / w_dem)
        - 45.0 * ((yy_d - h_dem / 2) / h_dem)
        + 35.0 * _smooth_field((h_dem, w_dem), 12.0, rng)
    )
    dem_out = np.where(inside_dem, dem_hi, -9999.0).astype("float32")
    _write_tif(RAW / "GNDEM.tif", dem_out, tf_dem, "EPSG:32652", -9999.0)

    # 5. precip_sum_2020.tif (~5x coarser, NaN nodata)
    h_p, w_p = 22, 36
    tf_p = from_origin(west, north, res_lst * 5, res_lst * 5)
    inside_p = ~geometry_mask(gdf.geometry, out_shape=(h_p, w_p), transform=tf_p, invert=False)
    p_vals = 460.0 + 35.0 * _smooth_field((h_p, w_p), 2.5, rng)
    p_out = np.where(inside_p, p_vals, np.nan).astype("float32")
    _write_tif(RAW / "precip_sum_2020.tif", p_out, tf_p, "EPSG:4326", np.nan)

    # 6. soil_moist_mean_2020.tif (~10x coarser, NaN nodata)
    h_s, w_s = 12, 18
    tf_s = from_origin(west, north, res_lst * 10, res_lst * 10)
    inside_s = ~geometry_mask(gdf.geometry, out_shape=(h_s, w_s), transform=tf_s, invert=False)
    s_vals = 0.28 + 0.04 * _smooth_field((h_s, w_s), 1.8, rng)
    s_out = np.where(inside_s, s_vals, np.nan).astype("float32")
    _write_tif(RAW / "soil_moist_mean_2020.tif", s_out, tf_s, "EPSG:4326", np.nan)

    print(f"Generated synthetic demo dataset in {RAW}")
    print("Note: These files are purely synthetic for tutorial execution and testing.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate synthetic demo rasters for Geo-ML.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing files in data/raw/")
    args = ap.parse_args()
    main(overwrite=args.overwrite)
