#!/usr/bin/env python3
"""
make_network_montage_nilearn.py

Make a surface montage for network maps in GIFTI format.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import nibabel as nib
from nilearn import plotting, datasets, surface


def _read_gifti_scalar(gii_path: Path) -> np.ndarray:
    g = nib.load(str(gii_path))
    if not hasattr(g, "darrays") or len(g.darrays) == 0:
        raise ValueError("No darrays found")
    data = np.asarray(g.darrays[0].data).reshape(-1)
    return data.astype(np.float64)


def _infer_available_networks(maps_dir: Path) -> List[int]:
    pats = sorted(maps_dir.glob("net_r*_lh*.func.gii"))
    nets = []
    for p in pats:
        name = p.name
        if not name.startswith("net_r"):
            continue
        try:
            r = int(name.split("_")[1][1:])
            nets.append(r)
        except Exception:
            continue
    return sorted(set(nets))


def _pick_hemi_file(maps_dir: Path, r: int, hemi: str, prefer_pos: bool) -> Path:
    r2 = f"{r:02d}"
    if prefer_pos:
        cand = sorted(maps_dir.glob(f"net_r{r2}_{hemi}_pos.func.gii"))
        if cand:
            return cand[0]
    cand = sorted(maps_dir.glob(f"net_r{r2}_{hemi}.func.gii"))
    if cand:
        return cand[0]
    cand = sorted(maps_dir.glob(f"net_r{r2}_{hemi}*.func.gii"))
    if cand:
        return cand[0]
    raise FileNotFoundError("Missing hemisphere map")


def _load_maps(maps_dir: Path, networks: List[int], prefer_pos: bool) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    out = {}
    for r in networks:
        fL = _pick_hemi_file(maps_dir, r, "lh", prefer_pos)
        fR = _pick_hemi_file(maps_dir, r, "rh", prefer_pos)
        vL = _read_gifti_scalar(fL)
        vR = _read_gifti_scalar(fR)
        out[r] = (vL, vR)
    return out


def _load_and_rescale_bg(sulc_path: str, pct: float = 98.0) -> np.ndarray:
    bg_full = surface.load_surf_data(sulc_path).astype(np.float64).reshape(-1)
    bg = bg_full[np.isfinite(bg_full)]
    if bg.size == 0:
        return bg_full
    scale = np.percentile(np.abs(bg), pct)
    if not np.isfinite(scale) or scale <= 0:
        return bg_full
    bg_full = np.clip(bg_full / scale, -1.0, 1.0)
    return bg_full


def _compute_symmetric_vmax(maps: Dict[int, Tuple[np.ndarray, np.ndarray]], pct: float = 99.0) -> float:
    vals = []
    for _, (vL, vR) in maps.items():
        vals.append(np.abs(vL))
        vals.append(np.abs(vR))
    x = np.concatenate(vals)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 1.0
    vmax = float(np.percentile(x, pct))
    if vmax <= 0:
        vmax = float(np.max(x))
    return float(vmax if vmax > 0 else 1.0)


def _parse_margins(s: str) -> Tuple[float, float, float, float]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) != 4:
        raise ValueError("Bad margins format")
    return tuple(float(x) for x in parts)  # type: ignore


def _make_montage(
    maps: Dict[int, Tuple[np.ndarray, np.ndarray]],
    surf_L: str,
    surf_R: str,
    bg_L: np.ndarray,
    bg_R: np.ndarray,
    out_pdf: Path,
    title: str,
    views: List[str],
    cmap: str,
    bg_on_data: bool,
    darkness: float,
    dpi: int,
    out_png: Optional[Path],
    vmax: Optional[float],
    show_colorbar: bool,
    wspace: float,
    hspace: float,
    margins: Tuple[float, float, float, float],
):
    networks = sorted(maps.keys())
    n = len(networks)
    if n == 0:
        raise ValueError("No networks to plot")
    if len(views) != 2:
        raise ValueError("Need exactly 2 views")

    if vmax is None:
        vmax = _compute_symmetric_vmax(maps, pct=99.0)
    vmin = -vmax

    ncols = 4
    fig_h = max(2.4, 1.70 * n)
    fig_w = 8
    fig, axes = plt.subplots(
        n,
        ncols,
        figsize=(fig_w, fig_h),
        subplot_kw={"projection": "3d"},
        squeeze=False,
    )

    left, right, top, bottom = margins

    if title:
        fig.text(0.5, 0.992, title, ha="center", va="top", fontsize=15)
    col_titles = ["Lateral (LH)", "Medial (LH)", "Lateral (RH)", "Medial (RH)"]
    col_x = [0.17, 0.4, 0.65, 0.88]
    for j, lab in enumerate(col_titles):
        fig.text(col_x[j], 0.968, lab, ha="center", va="top", fontsize=11)

    for i, r in enumerate(networks):
        y = 0.94 - (i + 0.5) * (0.90 / n)
        fig.text(0.012, y, f"Net {r}", va="center", ha="left", fontsize=12)

    for i, r in enumerate(networks):
        vL, vR = maps[r]

        plotting.plot_surf_stat_map(
            surf_L, vL, hemi="left", view=views[0],
            bg_map=bg_L, bg_on_data=bg_on_data, darkness=darkness,
            cmap=cmap, colorbar=False,
            axes=axes[i, 0], vmin=vmin, vmax=vmax, title=""
        )
        plotting.plot_surf_stat_map(
            surf_L, vL, hemi="left", view=views[1],
            bg_map=bg_L, bg_on_data=bg_on_data, darkness=darkness,
            cmap=cmap, colorbar=False,
            axes=axes[i, 1], vmin=vmin, vmax=vmax, title=""
        )
        plotting.plot_surf_stat_map(
            surf_R, vR, hemi="right", view=views[0],
            bg_map=bg_R, bg_on_data=bg_on_data, darkness=darkness,
            cmap=cmap, colorbar=False,
            axes=axes[i, 2], vmin=vmin, vmax=vmax, title=""
        )
        plotting.plot_surf_stat_map(
            surf_R, vR, hemi="right", view=views[1],
            bg_map=bg_R, bg_on_data=bg_on_data, darkness=darkness,
            cmap=cmap, colorbar=False,
            axes=axes[i, 3], vmin=vmin, vmax=vmax, title=""
        )

    if show_colorbar:
        norm = Normalize(vmin=vmin, vmax=vmax)
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cax = fig.add_axes([0.25, 0.035, 0.50, 0.018])
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cb.ax.tick_params(labelsize=9)

    plt.subplots_adjust(
        left=left,
        right=right,
        top=top,
        bottom=bottom,
        hspace=hspace,
        wspace=wspace,
    )

    fig.savefig(str(out_pdf), dpi=dpi)
    if out_png is not None:
        fig.savefig(str(out_png), dpi=dpi)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser("Make a montage PDF for surface network maps using nilearn")
    ap.add_argument("--maps_dir", required=True)
    ap.add_argument("--out_pdf", required=True)
    ap.add_argument("--out_png", default="")
    ap.add_argument("--networks", default="all")
    ap.add_argument("--title", default="")

    ap.add_argument("--views", default="lateral,medial")
    ap.add_argument("--prefer_pos", action="store_true")

    ap.add_argument("--cmap", default="RdBu_r")
    ap.add_argument("--bg_on_data", action="store_true")
    ap.add_argument("--darkness", type=float, default=0.10)
    ap.add_argument("--bg_rescale_pct", type=float, default=99.0)

    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--vmax", type=float, default=np.nan)
    ap.add_argument("--no_colorbar", action="store_true")

    ap.add_argument("--wspace", type=float, default=0.005, help="subplot wspace")
    ap.add_argument("--hspace", type=float, default=0.08, help="subplot hspace")
    ap.add_argument(
        "--margins",
        default="0.04,0.985,0.93,0.06",
        help="left,right,top,bottom margins",
    )

    args = ap.parse_args()

    maps_dir = Path(args.maps_dir).expanduser().resolve()
    out_pdf = Path(args.out_pdf).expanduser().resolve()
    out_png = Path(args.out_png).expanduser().resolve() if args.out_png.strip() else None
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    if out_png is not None:
        out_png.parent.mkdir(parents=True, exist_ok=True)

    if args.networks.strip().lower() == "all":
        nets = _infer_available_networks(maps_dir)
        if not nets:
            raise FileNotFoundError("No network maps found")
    else:
        nets = [int(x.strip()) for x in args.networks.split(",") if x.strip()]

    views = [v.strip() for v in args.views.split(",") if v.strip()]
    if len(views) != 2:
        raise ValueError("Need exactly 2 views")

    maps = _load_maps(maps_dir, nets, prefer_pos=args.prefer_pos)

    fs = datasets.fetch_surf_fsaverage("fsaverage5")
    surf_L = fs["pial_left"]
    surf_R = fs["pial_right"]

    bgL_path = fs.get("sulc_left", None)
    bgR_path = fs.get("sulc_right", None)
    if bgL_path is None or bgR_path is None:
        raise RuntimeError("Missing fsaverage5 sulc maps")

    bg_L = _load_and_rescale_bg(bgL_path, pct=float(args.bg_rescale_pct))
    bg_R = _load_and_rescale_bg(bgR_path, pct=float(args.bg_rescale_pct))

    vmax = None if (not np.isfinite(args.vmax)) else float(args.vmax)
    margins = _parse_margins(args.margins)

    _make_montage(
        maps=maps,
        surf_L=surf_L,
        surf_R=surf_R,
        bg_L=bg_L,
        bg_R=bg_R,
        out_pdf=out_pdf,
        out_png=out_png,
        title=args.title,
        views=views,
        cmap=args.cmap,
        bg_on_data=bool(args.bg_on_data),
        darkness=float(args.darkness),
        dpi=int(args.dpi),
        vmax=vmax,
        show_colorbar=(not args.no_colorbar),
        wspace=float(args.wspace),
        hspace=float(args.hspace),
        margins=margins,
    )

    print("Wrote:", out_pdf)
    if out_png is not None:
        print("Wrote:", out_png)


if __name__ == "__main__":
    main()