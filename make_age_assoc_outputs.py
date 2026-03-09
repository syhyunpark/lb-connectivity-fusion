#!/usr/bin/env python3
"""
make_age_assoc_outputs.py

Create tables and figures for age-association screening results.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RE_R = re.compile(r"_r(\d+)$", re.IGNORECASE)


def strip_prefix(feature: str) -> str:
    f = str(feature)
    if f.startswith("Fused__"):
        return f[len("Fused__"):]
    if f.startswith("FMRIonly__"):
        return f[len("FMRIonly__"):]
    if f.startswith("EEGonly__"):
        return f[len("EEGonly__"):]
    if "__" in f:
        return f.split("__", 1)[1]
    return f


def parse_r(feature: str) -> Optional[int]:
    base = strip_prefix(feature)
    m = RE_R.search(base)
    return int(m.group(1)) if m else None


def feature_set_name_from_feature(feature: str) -> str:
    f = str(feature)
    if f.startswith("FMRIonly__"):
        return "FMRI-only"
    if f.startswith("EEGonly__"):
        return "EEG-only"
    if f.startswith("Fused__"):
        return "Fused"
    return "Fused"


_RE_FMRI = re.compile(r"^(Lambda_fmri|fMRI_strength|FMRI_score)_r\d+$", re.IGNORECASE)
_RE_COM = re.compile(r"^EEG_(muHz|specCOM|specMean|specMeanHz|spectralMean|spectral_mean)_r\d+$", re.IGNORECASE)
_RE_BAND = re.compile(r"^EEG_(theta|alpha|beta|gamma)(?:_AUC)?_r\d+$", re.IGNORECASE)

_BAND_TO_TEX = {
    "theta": r"$\theta$",
    "alpha": r"$\alpha$",
    "beta": r"$\beta$",
    "gamma": r"$\gamma$",
}


def joint_feature_type(feature: str) -> str:
    base = strip_prefix(feature)

    if _RE_FMRI.match(base):
        return "Network strength"

    if _RE_COM.match(base):
        return "Spectral centroid (Hz)"

    m = _RE_BAND.match(base)
    if m:
        band = m.group(1).lower()
        return f"Band mass: {_BAND_TO_TEX.get(band, band)}"

    return "Other"


def type_short_label(t: str) -> str:
    if t == "Network strength":
        return "Network strength"
    if t == "Spectral centroid (Hz)":
        return "Spectral centroid"
    if t.startswith("Band mass:"):
        return t.replace("Band mass: ", "") + " mass"
    return "Other"


def network_label(r: int) -> str:
    return f"Network {r}"


def short_network_label(r: int) -> str:
    return f"Net {r}"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def latex_escape(s: str) -> str:
    return (
        str(s)
        .replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
    )


def resolve_results_csv(path_like: str) -> Path:
    p = Path(path_like).expanduser().resolve()
    if p.exists() and p.is_dir():
        cand = p / "assoc_feature_on_exposure_results.csv"
        if cand.exists():
            return cand
        hits = sorted(p.glob("*assoc*results*.csv")) + sorted(p.glob("*results*.csv"))
        if hits:
            return hits[0]
        raise FileNotFoundError("No results CSV found")
    if p.exists() and p.is_file():
        return p
    parent = p.parent
    if parent.exists() and parent.is_dir():
        cand = parent / "assoc_feature_on_exposure_results.csv"
        if cand.exists():
            return cand
        hits = sorted(parent.glob("*assoc*results*.csv")) + sorted(parent.glob("*results*.csv"))
        if hits:
            return hits[0]
        raise FileNotFoundError("Missing results CSV")
    raise FileNotFoundError("Missing results CSV")


def topk_networks_from_summary(summary_json: Path, K: int) -> List[int]:
    s = read_json(summary_json)
    if "energy_total_median" not in s:
        raise ValueError("Missing energy_total_median")
    E = np.asarray(s["energy_total_median"], dtype=float)
    order = np.argsort(-E)
    top = (order[: min(K, E.size)] + 1).tolist()
    return sorted(int(x) for x in top)


def write_counts_table(df_exp: pd.DataFrame, outdir: Path, alpha: float) -> None:
    qcol = "q_exposure" if "q_exposure" in df_exp.columns else ("q_global" if "q_global" in df_exp.columns else None)
    if qcol is None:
        raise ValueError("No q-values found")

    d = df_exp.copy()
    d["q"] = pd.to_numeric(d[qcol], errors="coerce")
    d["is_sig"] = d["q"] <= alpha

    summ = (
        d.groupby("feature_set", dropna=False)
        .agg(n_tests=("feature", "count"), n_sig=("is_sig", "sum"))
        .reset_index()
        .sort_values("feature_set")
    )
    summ.to_csv(outdir / "age_assoc_counts_by_set.csv", index=False)

    lines = []
    lines += [r"\begin{table}[t]", r"\centering"]
    lines += [
        r"\caption{Counts of BH-FDR significant associations by feature set.}",
        r"\label{tab:age_assoc_counts}",
        r"\setlength{\tabcolsep}{6pt}",
        r"\small",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Feature set & \#tests & \#sig (BH-FDR) \\",
        r"\midrule",
    ]
    for _, r in summ.iterrows():
        lines.append(f"{latex_escape(r['feature_set'])} & {int(r['n_tests'])} & {int(r['n_sig'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (outdir / "age_assoc_counts_by_set.tex").write_text("\n".join(lines))


def write_topk_fused_all_table(df_top: pd.DataFrame, outdir: Path, K: int, exposure: str) -> None:
    df_top.to_csv(outdir / "age_assoc_topK_fused_all.csv", index=False)

    type_order = {
        "Network strength": 0,
        "Spectral centroid (Hz)": 1,
        f"Band mass: {_BAND_TO_TEX['theta']}": 2,
        f"Band mass: {_BAND_TO_TEX['alpha']}": 3,
        f"Band mass: {_BAND_TO_TEX['beta']}": 4,
        f"Band mass: {_BAND_TO_TEX['gamma']}": 5,
        "Other": 9,
    }

    df2 = df_top.copy()
    df2["type_rank"] = df2["feature_type"].map(type_order).fillna(9).astype(int)
    df2 = df2.sort_values(["r", "type_rank", "feature_type"]).reset_index(drop=True)

    lines = []
    lines += [r"\begin{table}[t]", r"\centering"]
    lines.append(
        rf"\caption{{Age associations for all fused features derived from the Top-$K$ energy-screened networks "
        rf"($K={K}$), adjusting for sex. Each row corresponds to a univariate regression of $z(\mathrm{{feature}})$ "
        rf"on {latex_escape(exposure)} (years) plus covariates, with robust standard errors and BH-FDR correction. "
        rf"Effects are reported per decade (10-year change) in SD units of the feature. Rows are ordered by network "
        rf"ID (energy order) and feature type.}}"
    )
    lines += [
        r"\label{tab:age_assoc_topk_all}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\scriptsize",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Network & Feature summary & $\beta$ (per decade) & SE & $q$ \\",
        r"\midrule",
    ]

    for _, r in df2.iterrows():
        net = latex_escape(r["network"])
        feat = str(r["feature_human"])
        beta = float(r["beta_decade"])
        se = float(r["se_decade"])
        q = float(r["q"])
        lines.append(f"{net} & {feat} & {beta:.3f} & {se:.3f} & {q:.2e} \\\\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (outdir / "age_assoc_topK_fused_all.tex").write_text("\n".join(lines))


def write_baseline_longtable(df_base: pd.DataFrame, out_tex: Path, caption: str, label: str) -> None:
    lines = []
    lines.append(r"\begin{longtable}{llrrrr}")
    lines.append(rf"\caption{{{caption}}}\label{{{label}}}\\")
    lines.append(r"\toprule")
    lines.append(r"Network & Feature summary & $\beta$ (per decade) & SE & $q$ & Set \\")
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(r"\toprule")
    lines.append(r"Network & Feature summary & $\beta$ (per decade) & SE & $q$ & Set \\")
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{6}{r}{\footnotesize Continued on next page}\\")
    lines.append(r"\endfoot")
    lines.append(r"\bottomrule")
    lines.append(r"\endlastfoot")

    for _, r in df_base.iterrows():
        lines.append(
            f"{latex_escape(r['network'])} & {str(r['feature_human'])} & "
            f"{float(r['beta_decade']):.3f} & {float(r['se_decade']):.3f} & {float(r['q']):.2e} & "
            f"{latex_escape(r['feature_set'])} \\\\"
        )

    lines.append(r"\end{longtable}")
    out_tex.write_text("\n".join(lines))


def barplot_topN(
    df_topk_fused: pd.DataFrame,
    outdir: Path,
    title: str,
    N: int,
    formats: Tuple[str, ...],
    alpha: float,
) -> None:
    d = df_topk_fused.copy().reset_index(drop=True)

    d_sig = d[d["q"] <= alpha].copy()
    d_use = d_sig if len(d_sig) > 0 else d

    d_use["abs_beta"] = d_use["beta_decade"].abs()
    d_use = d_use.sort_values("abs_beta", ascending=False).head(N).reset_index(drop=True)

    labels = [f"{row['network_short']}\n{row['type_short']}" for _, row in d_use.iterrows()]
    y = d_use["abs_beta"].to_numpy(float)
    yerr = d_use["se_decade"].to_numpy(float)

    type_order = [
        "Network strength",
        "Spectral centroid (Hz)",
        f"Band mass: {_BAND_TO_TEX['theta']}",
        f"Band mass: {_BAND_TO_TEX['alpha']}",
        f"Band mass: {_BAND_TO_TEX['beta']}",
        f"Band mass: {_BAND_TO_TEX['gamma']}",
        "Other",
    ]
    cmap = plt.get_cmap("tab10")
    type_to_color = {t: cmap(i % 10) for i, t in enumerate(type_order)}

    fig = plt.figure(figsize=(10.5, 4.6))
    ax = plt.gca()
    x = np.arange(len(d_use))

    for i in range(len(d_use)):
        t = d_use.loc[i, "feature_type"]
        ax.bar(
            x[i],
            y[i],
            yerr=yerr[i],
            capsize=3,
            color=type_to_color.get(t, cmap(0)),
            edgecolor="black",
            linewidth=0.4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=7)
    ax.set_ylabel(r"$|\beta|$ (SD per decade)")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)

    handles = []
    labels_leg = []
    seen = set()
    for t in d_use["feature_type"].tolist():
        if t not in seen:
            seen.add(t)
            handles.append(plt.Rectangle((0, 0), 1, 1, color=type_to_color.get(t, cmap(0)), ec="black", lw=0.4))
            labels_leg.append(t)
    ax.legend(handles, labels_leg, frameon=False, ncol=3, fontsize=9, loc="upper right")

    plt.tight_layout()
    for ext in formats:
        plt.savefig(outdir / f"age_assoc_top{len(d_use)}_bar.{ext}", dpi=300)
    plt.close(fig)


def effect_significance_plot(
    df: pd.DataFrame,
    outbase: Path,
    title: str,
    alpha: float,
    formats: Tuple[str, ...],
    sets: List[str],
) -> None:
    d = df[df["feature_set"].isin(sets)].copy()
    d["p"] = pd.to_numeric(d["p"], errors="coerce")
    d["q"] = pd.to_numeric(d["q"], errors="coerce")
    d["beta_decade"] = pd.to_numeric(d["beta_decade"], errors="coerce")
    d = d[np.isfinite(d["beta_decade"]) & np.isfinite(d["p"]) & np.isfinite(d["q"])].copy()
    if d.empty:
        return

    d["mlog10p"] = -np.log10(np.clip(d["p"].to_numpy(float), 1e-300, 1.0))
    d["is_sig"] = d["q"].to_numpy(float) <= float(alpha)

    fig = plt.figure(figsize=(8.2, 4.6))
    ax = plt.gca()

    ax.scatter(d["beta_decade"], d["mlog10p"], s=18, alpha=0.25, edgecolors="none")
    ds = d[d["is_sig"]]
    ax.scatter(ds["beta_decade"], ds["mlog10p"], s=34, alpha=0.9, edgecolors="black", linewidths=0.9)

    ax.axvline(0.0, linewidth=1, color="0.3")
    ax.axhline(-math.log10(0.05), linestyle="--", linewidth=1, color="0.5")

    ax.set_xlabel(r"$\beta$ (SD per decade)")
    ax.set_ylabel(r"$-\log_{10}(p)$")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)

    ax.text(
        0.98,
        0.18,
        "gray dashed: p = 0.05\nblack outline: BH-FDR significant",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
    )

    plt.tight_layout()
    for ext in formats:
        plt.savefig(outbase.with_suffix(f".{ext}"), dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results_csv",
        required=True,
        help="Path to assoc_feature_on_exposure_results.csv or a directory containing it",
    )
    ap.add_argument("--model_summary_json", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--exposure", default="age_mid")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--topk_networks", type=int, default=5)
    ap.add_argument("--top_plot_n", type=int, default=10)
    ap.add_argument("--per_decade", type=float, default=10.0)
    ap.add_argument("--formats", default="pdf,png")
    ap.add_argument("--title_prefix", default="MPI-LEMON (EO)")
    ap.add_argument("--restrict_baselines_to_topk", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir).expanduser().resolve()
    ensure_dir(outdir)
    formats = tuple([x.strip() for x in args.formats.split(",") if x.strip()])

    results_csv = resolve_results_csv(args.results_csv)
    summary_json = Path(args.model_summary_json).expanduser().resolve()
    if not summary_json.exists():
        raise FileNotFoundError("Missing model summary JSON")

    df = pd.read_csv(results_csv)
    if "exposure" not in df.columns:
        raise ValueError("Expected column 'exposure'")

    d = df[df["exposure"] == args.exposure].copy()
    if d.empty:
        raise ValueError("No rows for requested exposure")

    if "feature_set" not in d.columns:
        d["feature_set"] = d["feature"].apply(feature_set_name_from_feature)

    for col in ["beta", "se", "p"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")

    qcol = "q_exposure" if "q_exposure" in d.columns else ("q_global" if "q_global" in d.columns else None)
    if qcol is None:
        raise ValueError("No q-values found")
    d["q"] = pd.to_numeric(d[qcol], errors="coerce")

    d["beta_decade"] = d["beta"] * float(args.per_decade)
    d["se_decade"] = d["se"] * float(args.per_decade)

    d["r"] = d["feature"].apply(parse_r)
    d["feature_type"] = d["feature"].apply(joint_feature_type)
    d["type_short"] = d["feature_type"].apply(type_short_label)

    top_r = topk_networks_from_summary(summary_json, args.topk_networks)

    write_counts_table(d, outdir, alpha=args.alpha)

    df_topk = d[(d["feature_set"] == "Fused") & (d["r"].isin(top_r))].copy().reset_index(drop=True)
    df_topk["network"] = df_topk["r"].apply(lambda r: network_label(int(r)) if pd.notna(r) else "Network ?")
    df_topk["network_short"] = df_topk["r"].apply(
        lambda r: short_network_label(int(r)) if pd.notna(r) else "Net ?"
    )
    df_topk["feature_human"] = df_topk.apply(
        lambda row: f"{network_label(int(row['r']))}: {row['feature_type']}"
        if pd.notna(row["r"]) else "Network ?: feature",
        axis=1,
    )

    write_topk_fused_all_table(df_topk, outdir, K=args.topk_networks, exposure=args.exposure)

    barplot_topN(
        df_topk_fused=df_topk,
        outdir=outdir,
        title=f"{args.title_prefix} Top {args.top_plot_n} age associations (fused features)",
        N=int(args.top_plot_n),
        formats=formats,
        alpha=args.alpha,
    )

    effect_significance_plot(
        d,
        outdir / "age_assoc_effect_significance_fused",
        title=f"{args.title_prefix}: Effect vs significance (fused features)",
        alpha=args.alpha,
        formats=formats,
        sets=["Fused"],
    )

    d_base = d[d["feature_set"].isin(["FMRI-only", "EEG-only"])].copy()
    if args.restrict_baselines_to_topk:
        d_base = d_base[d_base["r"].isin(top_r)].copy()

    effect_significance_plot(
        d_base,
        outdir / "age_assoc_effect_significance_baselines",
        title=f"{args.title_prefix}: Effect vs significance (single-modality baselines)",
        alpha=args.alpha,
        formats=formats,
        sets=["FMRI-only", "EEG-only"],
    )

    df_fmri_only = d[d["feature_set"] == "FMRI-only"].copy()
    df_eeg_only = d[d["feature_set"] == "EEG-only"].copy()
    if args.restrict_baselines_to_topk:
        df_fmri_only = df_fmri_only[df_fmri_only["r"].isin(top_r)].copy()
        df_eeg_only = df_eeg_only[df_eeg_only["r"].isin(top_r)].copy()

    for df_base, name in [(df_fmri_only, "fmri_only"), (df_eeg_only, "eeg_only")]:
        df_base["network"] = df_base["r"].apply(lambda r: network_label(int(r)) if pd.notna(r) else "Network ?")
        df_base["feature_human"] = df_base.apply(
            lambda row: f"{network_label(int(row['r']))}: {joint_feature_type(row['feature'])}"
            if pd.notna(row["r"]) else "Network ?: feature",
            axis=1,
        )
        type_rank_map = {
            "Network strength": 0,
            "Spectral centroid (Hz)": 1,
            f"Band mass: {_BAND_TO_TEX['theta']}": 2,
            f"Band mass: {_BAND_TO_TEX['alpha']}": 3,
            f"Band mass: {_BAND_TO_TEX['beta']}": 4,
            f"Band mass: {_BAND_TO_TEX['gamma']}": 5,
            "Other": 9,
        }
        df_base["feature_type"] = df_base["feature"].apply(joint_feature_type)
        df_base["type_rank"] = df_base["feature_type"].map(type_rank_map).fillna(9).astype(int)
        df_base = df_base.sort_values(["r", "type_rank", "feature_type"]).reset_index(drop=True)

        out_csv = outdir / f"supp_age_assoc_{name}_all.csv"
        df_base.to_csv(out_csv, index=False)

        out_tex = outdir / f"supp_age_assoc_{name}_all.tex"
        caption = (
            f"All age associations for {('fMRI-only' if name == 'fmri_only' else 'EEG-only')} baseline features "
            f"(effects per decade)."
        )
        if args.restrict_baselines_to_topk:
            caption += f" Restricted to Top-{args.topk_networks} networks by energy."
        label = f"tab:supp_age_assoc_{name}"
        write_baseline_longtable(df_base, out_tex, caption=caption, label=label)

    print("Using results CSV:", results_csv)
    print("Wrote outputs to:", outdir)


if __name__ == "__main__":
    main()