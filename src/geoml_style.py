"""Shared Chinese cartographic style for the Geo-ML tutorial."""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT / "outputs" / "figures"
TABLE_DIR = PROJECT / "outputs" / "tables"

MISSING_COLOR = "#e4e4e4"
INK = "#2f3437"
GRID = "#d9d9d9"
PALETTE = ["#2a6f97", "#d97706", "#3a7d44", "#9b4f96", "#6b7280", "#b23a48"]


def setup() -> None:
    """Apply a print-friendly Chinese textbook style."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 110,
            "savefig.dpi": 220,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": INK,
            "axes.linewidth": 0.8,
            "axes.titlesize": 13,
            "axes.titleweight": "semibold",
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.2,
            "ytick.labelsize": 9.2,
            "legend.fontsize": 9.2,
            "legend.frameon": False,
            "axes.grid": False,
            "figure.constrained_layout.use": False,
        }
    )


def save(fig, name: str, dpi: int = 220, facecolor: str = "white") -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=facecolor)
    return path


def _format_coord_axes(ax, crs) -> None:
    """Degree labels for EPSG:4326, kilometre labels otherwise."""
    if crs is None:
        return
    crs_text = str(crs)
    if "4326" in crs_text:
        xspan = abs(ax.get_xlim()[1] - ax.get_xlim()[0])
        yspan = abs(ax.get_ylim()[1] - ax.get_ylim()[0])
        xdec = 3 if xspan < 0.1 else 2 if xspan < 0.5 else 1
        ydec = 3 if yspan < 0.1 else 2 if yspan < 0.5 else 1
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
        ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter(f"%.{xdec}f°E"))
        ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter(f"%.{ydec}f°N"))
    else:
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
        ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, p: f"{v / 1000:.0f}"))
        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, p: f"{v / 1000:.0f}"))
        ax.set_xlabel("东坐标 / km")
        ax.set_ylabel("北坐标 / km")


def add_north_arrow(ax, x: float = 0.92, y: float = 0.90, size: float = 0.055) -> None:
    ax.annotate(
        "N",
        xy=(x, y),
        xytext=(x, y - size * 1.8),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color=INK,
        arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.0, shrinkA=0, shrinkB=2),
        zorder=20,
    )


def add_scalebar(ax, extent, km: float = 20.0, color: str = INK) -> None:
    """Approximate scale bar using the latitude at the map centre."""
    left, right, bottom, top = extent
    lat = (bottom + top) / 2
    deg = km / (111.32 * np.cos(np.deg2rad(lat)))
    width = right - left
    height = top - bottom
    x0 = left + 0.06 * width
    y0 = bottom + 0.075 * height
    ax.plot([x0, x0 + deg], [y0, y0], color=color, lw=2.4, solid_capstyle="butt", zorder=20)
    ax.plot([x0, x0], [y0 - 0.008 * height, y0 + 0.008 * height], color=color, lw=1.0, zorder=20)
    ax.plot([x0 + deg, x0 + deg], [y0 - 0.008 * height, y0 + 0.008 * height], color=color, lw=1.0, zorder=20)
    ax.text(x0 + deg / 2, y0 + 0.022 * height, f"{km:g} km", ha="center", va="bottom",
            fontsize=8.5, color=color, zorder=20)


def plot_raster(
    ax,
    arr,
    extent,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
    title: str = "",
    cbar_label: str = "",
    boundary=None,
    crs=None,
    add_bar: bool = False,
    edge_ticks: bool = True,
):
    """Plot one north-up raster with light-grey NoData and optional map furniture."""
    cmap_obj = mpl.colormaps[cmap].copy()
    cmap_obj.set_bad(MISSING_COLOR)
    display = np.ma.masked_invalid(arr)
    if vmin is None:
        vmin = float(np.nanpercentile(arr, 2))
    if vmax is None:
        vmax = float(np.nanpercentile(arr, 98))
    im = ax.imshow(
        display,
        extent=extent,
        origin="upper",
        cmap=cmap_obj,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        aspect="auto",
        zorder=1,
    )
    ax.set_facecolor(MISSING_COLOR)
    if boundary is not None and crs is not None:
        boundary.to_crs(crs).boundary.plot(ax=ax, color="#3b3b3b", linewidth=0.7, zorder=5)
    ax.set_title(title, pad=7)
    _format_coord_axes(ax, crs)
    if add_bar:
        add_north_arrow(ax)
        add_scalebar(ax, extent)
    if not edge_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.036, pad=0.025)
    cb.set_label(cbar_label, fontsize=9)
    cb.ax.tick_params(labelsize=8)
    cb.outline.set_linewidth(0.5)
    return im


def map_extent(transform, shape) -> tuple[float, float, float, float]:
    h, w = shape
    left, top = transform * (0, 0)
    right, bottom = transform * (w, h)
    return (left, right, bottom, top)


def plot_triptych(
    obs,
    pred,
    resid,
    extent,
    title: str,
    out_name: str,
    cbar_label: str = "LST（原始像元值）",
    cmap: str = "magma",
    boundary=None,
    crs=None,
    add_bar: bool = True,
):
    """Observed / predicted / residual maps on the same held-out window."""
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.4), constrained_layout=True)
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    resid = np.asarray(resid, dtype=float)
    vmin = float(np.nanmin([np.nanmin(obs), np.nanmin(pred)]))
    vmax = float(np.nanmax([np.nanmax(obs), np.nanmax(pred)]))
    for ax, arr, panel, ttl in zip(
        axes[:2],
        [obs, pred],
        ["(a)", "(b)"],
        ["观测", "预测"],
    ):
        plot_raster(
            ax, arr, extent, cmap, vmin, vmax,
            title=f"{panel} {ttl}", cbar_label=cbar_label,
            boundary=boundary, crs=crs, add_bar=False,
        )
        if add_bar:
            add_north_arrow(ax)
            add_scalebar(ax, extent)
    lim = float(np.nanpercentile(np.abs(resid), 98))
    lim = max(lim, 1e-6)
    cmap_r = mpl.colormaps["RdBu_r"].copy()
    cmap_r.set_bad(MISSING_COLOR)
    im = axes[2].imshow(
        np.ma.masked_invalid(resid), extent=extent, origin="upper",
        cmap=cmap_r, norm=mpl.colors.TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim),
        interpolation="nearest", aspect="auto", zorder=1,
    )
    axes[2].set_facecolor(MISSING_COLOR)
    if boundary is not None and crs is not None:
        boundary.to_crs(crs).boundary.plot(ax=axes[2], color="#3b3b3b", linewidth=0.7, zorder=5)
    axes[2].set_title("(c) 残差（预测 − 观测）", pad=7)
    _format_coord_axes(axes[2], crs)
    if add_bar:
        add_north_arrow(axes[2])
        add_scalebar(axes[2], extent)
    cb = fig.colorbar(im, ax=axes[2], fraction=0.036, pad=0.025)
    cb.set_label("残差（预测 − 观测）", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    cb.outline.set_linewidth(0.5)
    for ax in axes:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        _format_coord_axes(ax, crs)
    fig.suptitle(title, fontsize=14, fontweight="semibold")
    path = save(fig, out_name)
    plt.close(fig)
    return path


def model_comparison(
    fold_scores: dict,
    test_scores: dict,
    out_name: str,
    title: str,
    ylabel: str = "RMSE（原始像元值）",
    lower_is_better: bool = True,
):
    """Dot-and-whisker comparison using one shared error scale."""
    names = list(fold_scores)
    fig, ax = plt.subplots(figsize=(8.4, 3.6), constrained_layout=True)
    rng = np.random.default_rng(42)
    for i, name in enumerate(names):
        vals = np.asarray(fold_scores[name], dtype=float)
        x = i + rng.uniform(-0.11, 0.11, size=vals.size)
        ax.scatter(x, vals, s=34, color=PALETTE[i % len(PALETTE)], alpha=0.75,
                   edgecolor="white", linewidth=0.5, zorder=3, label="空间折" if i == 0 else None)
        ax.hlines(vals.mean(), i - 0.24, i + 0.24, color=PALETTE[i % len(PALETTE)],
                  lw=2.0, zorder=4)
        ax.scatter([i], [test_scores[name]], marker="D", s=62, color="#1f2937",
                   edgecolor="white", linewidth=0.7, zorder=5, label="独立测试区" if i == 0 else None)
        ax.annotate(f"折均 {vals.mean():.2f}", (i, vals.mean()), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", color=GRID, lw=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="best", ncol=2)
    ax.margins(x=0.08, y=0.18)
    path = save(fig, out_name)
    plt.close(fig)
    return path


def plot_split_map(
    df,
    shape,
    extent,
    out_name: str,
    boundary=None,
    crs=None,
    title: str = "空间划分：5 个训练折与 1 个独立测试区",
):
    """Map fold labels on the full target grid, with the test block outlined."""
    arr = np.full(shape, np.nan, dtype=float)
    arr[df["row"].to_numpy(), df["col"].to_numpy()] = df["fold"].to_numpy()
    cmap = mpl.colors.ListedColormap(
        ["#bdbdbd", "#2a6f97", "#e9a23b", "#3a7d44", "#9b4f96", "#b23a48"]
    )
    fig, ax = plt.subplots(figsize=(7.6, 5.2), constrained_layout=True)
    ax.set_facecolor(MISSING_COLOR)
    im = ax.imshow(
        np.ma.masked_invalid(arr), extent=extent, origin="upper",
        cmap=cmap, interpolation="nearest", vmin=-1.5, vmax=4.5, aspect="auto",
    )
    if boundary is not None and crs is not None:
        boundary.to_crs(crs).boundary.plot(ax=ax, color="#303030", linewidth=0.8, zorder=5)
    test = df[df["split"].eq("test")]
    if len(test):
        x0 = test["x"].min()
        x1 = test["x"].max()
        y0 = test["y"].min()
        y1 = test["y"].max()
        ax.add_patch(
            mpl.patches.Rectangle(
                (x0, y0), x1 - x0, y1 - y0, fill=False,
                edgecolor="#111111", linewidth=1.4, linestyle=(0, (4, 2)), zorder=9,
            )
        )
        ax.annotate("独立测试区", xy=((x0 + x1) / 2, y0), xytext=(0, -18),
                    textcoords="offset points", ha="center", va="top",
                    fontsize=9, color="#111111", fontweight="semibold", zorder=10,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
    _format_coord_axes(ax, crs)
    ax.set_title(title)
    handles = [
        mpl.patches.Patch(color="#bdbdbd", label="独立测试区"),
        *[mpl.patches.Patch(color=cmap(i + 1), label=f"空间折 {i + 1}") for i in range(5)],
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8.2, ncol=2)
    add_north_arrow(ax)
    add_scalebar(ax, extent)
    path = save(fig, out_name)
    plt.close(fig)
    return path


def plot_scatter(
    y_true,
    y_pred,
    out_name: str,
    title: str,
    r2: float | None = None,
    mae: float | None = None,
):
    fig, ax = plt.subplots(figsize=(4.8, 4.6), constrained_layout=True)
    ax.scatter(y_true, y_pred, s=13, alpha=0.45, color="#2a6f97", edgecolor="none")
    lo = float(np.nanmin([np.nanmin(y_true), np.nanmin(y_pred)]))
    hi = float(np.nanmax([np.nanmax(y_true), np.nanmax(y_pred)]))
    ax.plot([lo, hi], [lo, hi], color="#b23a48", lw=1.1, ls="--", label="1:1")
    ax.set_xlabel("观测 LST（原始像元值）")
    ax.set_ylabel("预测 LST（原始像元值）")
    ax.set_title(title)
    ax.legend(loc="upper left")
    txt = []
    if r2 is not None:
        txt.append(f"R² = {r2:.3f}")
    if mae is not None:
        txt.append(f"MAE = {mae:.2f}")
    if txt:
        ax.text(0.97, 0.05, "\n".join(txt), transform=ax.transAxes, ha="right",
                va="bottom", fontsize=9, color=INK)
    ax.set_aspect("equal", adjustable="box")
    path = save(fig, out_name)
    plt.close(fig)
    return path
