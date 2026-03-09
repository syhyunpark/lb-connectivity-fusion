#!/usr/bin/env python3
# make_elbow_figure.py
#
# Reads *_summary.json under an anchor-refit folder and produces a single 3-panel elbow PDF:
#   (1) fMRI median relerr vs rank
#   (2) EEG median relerr vs rank
#   (3) Combined median relerr vs rank 
#
# Example:
# python3 make_elbow_figure.py \
#   --indir  /Users/hyung/lemon_integration_work/anchor_refit_EO_sep \
#   --outpdf /Users/hyung/lemon_integration_work/anchor_refit_EO_sep/Fig_elbow_EO.pdf \
#   --highlight_R 12

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _infer_R_from_path(p: Path) -> Optional[int]:
    s = p.as_posix()
    m = re.search(r"_R(\d+)\b", s)
    if m:
        return int(m.group(1))
    m = re.search(r"R(\d+)\b", s)
    if m:
        return int(m.group(1))
    return None


def _load_json(p: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(p, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _compute_total_relerr(j: Dict[str, Any], fmri: float, eeg: float) -> float:
    t = _safe_float(j.get("total_relerr_median"))
    if t is not None:
        return t

    w_fmri = _safe_float(j.get("w_fmri_used"))
    w_eeg = _safe_float(j.get("w_eeg_used"))

    if w_fmri is None:
        w_fmri = 1.0

    if w_eeg is None:
        F = _safe_float(j.get("F"))
        w_eeg = (1.0 / float(F)) if (F is not None and F > 0) else 0.05

    return w_fmri * fmri + w_eeg * eeg


def collect_points(indir: Path, use_offdiag: bool) -> pd.DataFrame:
    files = sorted(indir.rglob("*_summary.json"))
    if not files:
        raise FileNotFoundError("No summary files found")

    rows: List[Dict[str, Any]] = []
    for fp in files:
        j = _load_json(fp)
        if j is None:
            continue

        R = j.get("R", None)
        if R is None:
            R = _infer_R_from_path(fp)
        if R is None:
            continue
        R = int(R)

        if use_offdiag:
            fmri_key = "fmri_relerr_offdiag_median"
            eeg_key = "eeg_relerr_obs_offdiag_median"
        else:
            fmri_key = "fmri_relerr_median"
            eeg_key = "eeg_relerr_obs_median"

        fmri = _safe_float(j.get(fmri_key))
        eeg = _safe_float(j.get(eeg_key))
        if fmri is None or eeg is None:
            continue

        total = _compute_total_relerr(j, fmri, eeg)

        rows.append({
            "file": str(fp),
            "R": R,
            "fmri": fmri,
            "eeg": eeg,
            "total": total,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No usable summary rows found")

    df_best = (
        df.sort_values(["R", "total"], ascending=[True, True])
          .groupby("R", as_index=False)
          .first()
          .sort_values("R")
          .reset_index(drop=True)
    )
    return df_best


def _style_axes(ax):
    ax.grid(True, linewidth=0.6, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    ap = argparse.ArgumentParser("Create a 3-panel elbow figure from summary JSON files.")
    ap.add_argument("--indir", required=True, help="Folder searched recursively for summary JSON files")
    ap.add_argument("--outpdf", required=True, help="Output PDF path")
    ap.add_argument("--outcsv", default="", help="Optional CSV export")
    ap.add_argument("--highlight_R", type=int, default=12, help="Rank to mark with a vertical line")
    ap.add_argument("--offdiag", action="store_true", help="Use off-diagonal relative-error keys")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    indir = Path(args.indir).expanduser().resolve()
    outpdf = Path(args.outpdf).expanduser().resolve()
    outpdf.parent.mkdir(parents=True, exist_ok=True)

    df = collect_points(indir, use_offdiag=bool(args.offdiag))

    if args.outcsv:
        outcsv = Path(args.outcsv).expanduser().resolve()
        outcsv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(outcsv, index=False)

    Rvals = df["R"].to_numpy()
    y_fmri = df["fmri"].to_numpy()
    y_eeg = df["eeg"].to_numpy()
    y_total = df["total"].to_numpy()

    y_all = np.concatenate([y_fmri, y_eeg, y_total])
    y_min = float(np.min(y_all))
    y_max = float(np.max(y_all))
    pad = 0.05 * (y_max - y_min + 1e-12)
    ylim = (y_min - pad, y_max + pad)

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), sharex=True, sharey=True)

    panels = [
        ("fMRI", y_fmri),
        ("EEG", y_eeg),
        ("Total", y_total),
    ]

    for ax, (title, y) in zip(axes, panels):
        _style_axes(ax)
        ax.plot(Rvals, y, marker="o", linewidth=1.6)
        ax.set_title(title)
        ax.set_ylim(*ylim)
        ax.set_xlabel("Rank $R$")
        ax.set_xticks(Rvals)

        if args.highlight_R in set(Rvals.tolist()):
            ax.axvline(args.highlight_R, linestyle="--", linewidth=1.0, alpha=0.6)
            idx = np.where(Rvals == args.highlight_R)[0][0]
            ax.plot([Rvals[idx]], [y[idx]], marker="o", markersize=7)

    axes[0].set_ylabel("Median relative error")

    fig.tight_layout()
    fig.savefig(outpdf, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print("Wrote:", outpdf)
    if args.outcsv:
        print("Wrote:", outcsv)


if __name__ == "__main__":
    main()