#!/usr/bin/env python3
"""
roi_system_interpretability_from_BI.py

Summarize system composition from a bilateral Schaefer BI CSV.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


KNOWN_SYSTEMS = ["Default", "Cont", "Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic"]

BAD_TERMS = [
    "medial_wall",
    "medial wall",
    "background",
    "unknown",
    "???",
    "freesurfer_defined_medial_wall",
    "defined_medial_wall",
]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _guess_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols_lc = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lc:
            return cols_lc[cand.lower()]
    for cand in candidates:
        for c in df.columns:
            if cand.lower() in c.lower():
                return c
    return None


def _parse_network_id(x) -> Optional[int]:
    if pd.isna(x):
        return None
    s = str(x).strip()
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else None


def _clean_parcel_name(parcel: str) -> str:
    if not isinstance(parcel, str):
        return ""
    s = parcel.strip()
    s = s.replace("7Networks_LH_", "").replace("7Networks_RH_", "")
    s = s.replace("7Networks_", "")
    return s


def _parse_system(parcel_clean: str) -> str:
    if not isinstance(parcel_clean, str) or parcel_clean.strip() == "":
        return "Other"
    sys = parcel_clean.split("_")[0]
    return sys if sys in KNOWN_SYSTEMS else "Other"


def _parse_region_tag(parcel_clean: str) -> str:
    if not isinstance(parcel_clean, str) or parcel_clean.strip() == "":
        return "Other"
    parts = parcel_clean.split("_")
    if len(parts) < 2:
        return "Other"
    if len(parts) == 2:
        return parts[0]
    mid = parts[1:-1]
    return "_".join(mid) if mid else parts[0]


def load_bi_csv(bi_csv: str, mass_mode: str) -> pd.DataFrame:
    df = pd.read_csv(bi_csv)

    net_col = _guess_col(df, ["network", "net", "r", "component", "factor"])
    parcel_col = _guess_col(df, ["parcel_name_clean", "parcel_name", "parcel", "roi", "label", "name"])
    abs_col = _guess_col(df, ["abs_mean", "absmean"])
    nvert_col = _guess_col(df, ["n_vertices", "nverts", "n_vert", "n"])

    if net_col is None or parcel_col is None or abs_col is None:
        raise ValueError("Missing required columns")

    d = df.copy()
    d["r"] = d[net_col].apply(_parse_network_id)
    d["parcel_name_clean"] = d[parcel_col].astype(str).apply(_clean_parcel_name)

    name_lower = d["parcel_name_clean"].str.lower()
    bad = np.zeros(len(d), dtype=bool)
    for t in BAD_TERMS:
        bad |= name_lower.str.contains(t, regex=False, na=False)
    d = d[~bad].copy()

    d["abs_mean"] = pd.to_numeric(d[abs_col], errors="coerce")

    if mass_mode == "abs_mean_x_nverts":
        if nvert_col is None:
            raise ValueError("n_vertices column required for this mass_mode")
        d["n_vertices"] = pd.to_numeric(d[nvert_col], errors="coerce")
        d["mass"] = d["abs_mean"] * d["n_vertices"]
    else:
        d["mass"] = d["abs_mean"]

    d = d.dropna(subset=["r", "mass"])
    d["r"] = d["r"].astype(int)

    d["system"] = d["parcel_name_clean"].apply(_parse_system)
    d = d[d["system"].isin(KNOWN_SYSTEMS)].copy()

    return d[["r", "parcel_name_clean", "system", "abs_mean", "mass"]].copy()


def compute_system_pct(d_long: pd.DataFrame, networks: List[int]) -> pd.DataFrame:
    g = d_long[d_long["r"].isin(networks)].groupby(["r", "system"], as_index=False)["mass"].sum()
    wide = g.pivot_table(index="r", columns="system", values="mass", aggfunc="sum", fill_value=0.0)

    for s in KNOWN_SYSTEMS:
        if s not in wide.columns:
            wide[s] = 0.0
    wide = wide[KNOWN_SYSTEMS].sort_index()

    totals = wide.sum(axis=1).replace(0.0, np.nan)
    pct = (100.0 * wide.div(totals, axis=0)).fillna(0.0)
    return pct


def top_systems_str(pct_row: pd.Series, k: int = 3) -> str:
    s = pct_row.sort_values(ascending=False)
    out = []
    for sys, v in s.iloc[:k].items():
        out.append(f"{sys} ({v:.1f}\\%)")
    return ", ".join(out)


def dominant_anatomy_summary(
    d_long: pd.DataFrame,
    r: int,
    top_parcels: int = 15,
    top_systems: int = 3,
    top_regions_per_system: int = 3,
) -> str:
    dd = d_long[d_long["r"] == r].copy()
    if dd.empty:
        return "Unlabeled"

    dd = dd.sort_values("mass", ascending=False).head(top_parcels).copy()
    sys_mass = dd.groupby("system")["mass"].sum().sort_values(ascending=False)
    sys_keep = list(sys_mass.index[:top_systems])

    chunks = []
    for sys in sys_keep:
        dsys = dd[dd["system"] == sys].copy()
        if dsys.empty:
            continue
        dsys["region"] = dsys["parcel_name_clean"].apply(_parse_region_tag)
        reg_mass = dsys.groupby("region")["mass"].sum().sort_values(ascending=False)
        regs = [x for x in reg_mass.index[:top_regions_per_system] if isinstance(x, str) and x]
        reg_str = ", ".join(regs) if regs else sys
        chunks.append(rf"\textbf{{{sys}}} ({reg_str})")

    return ", ".join(chunks) if chunks else "Unlabeled"


def plot_heatmap(pct: pd.DataFrame, out_pdf: Path, title: str, also_png: bool) -> None:
    mat = pct[KNOWN_SYSTEMS].to_numpy(float)
    nets = [int(r) for r in pct.index.tolist()]

    fig = plt.figure(figsize=(7.6, 4.3))
    ax = plt.gca()
    im = ax.imshow(mat, aspect="auto", interpolation="nearest")

    ax.set_yticks(np.arange(len(nets)))
    ax.set_yticklabels([f"Net {r}" for r in nets], fontsize=9)

    ax.set_xticks(np.arange(len(KNOWN_SYSTEMS)))
    ax.set_xticklabels(KNOWN_SYSTEMS, rotation=30, ha="right", fontsize=9)

    ax.set_title(title, fontsize=12)
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Percent mass (%)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    plt.tight_layout()
    fig.savefig(out_pdf, dpi=300)
    if also_png:
        fig.savefig(out_pdf.with_suffix(".png"), dpi=300)
    plt.close(fig)


def write_combined_table_tex(
    pct: pd.DataFrame,
    d_long: pd.DataFrame,
    networks: List[int],
    out_tex: Path,
    topk_show: int,
    top_parcels: int,
    top_systems: int,
    top_regions_per_system: int,
) -> None:
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Schaefer-7 system composition and dominant anatomy summaries for each fused network (bilateral pooling; magnitude-based).}")
    lines.append(r"\label{tab:schaefer7_comp_and_anatomy}")
    lines.append(r"\setlength{\tabcolsep}{5pt}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{c p{0.42\linewidth} p{0.44\linewidth}}")
    lines.append(r"\toprule")
    lines.append(r"Net $r$ & Top systems (percent mass) & Dominant systems / anatomy (bilateral) \\")
    lines.append(r"\midrule")

    for r in networks:
        if r not in pct.index:
            top_sys = "Unlabeled"
        else:
            top_sys = top_systems_str(pct.loc[r], k=topk_show)
        anat = dominant_anatomy_summary(
            d_long=d_long,
            r=r,
            top_parcels=top_parcels,
            top_systems=top_systems,
            top_regions_per_system=top_regions_per_system,
        )
        lines.append(f"{r} & {top_sys} & {anat} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out_tex.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser("System composition heatmap and table from bilateral Schaefer BI CSV")
    ap.add_argument("--bi_csv", required=True)
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--mass_mode", choices=["abs_mean_x_nverts", "abs_mean"], default="abs_mean_x_nverts")
    ap.add_argument("--networks", default="all", help="Comma list like 1,2,3 or 'all'")
    ap.add_argument("--topk_show", type=int, default=3)

    ap.add_argument("--top_parcels_for_anatomy", type=int, default=15)
    ap.add_argument("--top_systems_for_anatomy", type=int, default=3)
    ap.add_argument("--top_regions_per_system", type=int, default=3)

    ap.add_argument("--heatmap_title", default="MPI-LEMON (EO): system composition of fused networks (R=12)")
    ap.add_argument("--also_write_png", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir).expanduser().resolve()
    ensure_dir(outdir)

    d_long = load_bi_csv(args.bi_csv, mass_mode=args.mass_mode)

    if args.networks.strip().lower() == "all":
        networks = sorted(int(x) for x in d_long["r"].unique())
    else:
        networks = [int(x.strip()) for x in args.networks.split(",") if x.strip()]

    pct = compute_system_pct(d_long, networks=networks)

    pct.reset_index().rename(columns={"r": "network"}).to_csv(
        outdir / "system_composition_wide.csv",
        index=False,
    )

    plot_heatmap(
        pct=pct,
        out_pdf=outdir / "system_composition_heatmap.pdf",
        title=args.heatmap_title,
        also_png=bool(args.also_write_png),
    )

    write_combined_table_tex(
        pct=pct,
        d_long=d_long,
        networks=networks,
        out_tex=outdir / "system_composition_and_anatomy_table.tex",
        topk_show=int(args.topk_show),
        top_parcels=int(args.top_parcels_for_anatomy),
        top_systems=int(args.top_systems_for_anatomy),
        top_regions_per_system=int(args.top_regions_per_system),
    )

    print("Wrote outputs to:", outdir)
    print(" -", outdir / "system_composition_heatmap.pdf")
    print(" -", outdir / "system_composition_wide.csv")
    print(" -", outdir / "system_composition_and_anatomy_table.tex")


if __name__ == "__main__":
    main()