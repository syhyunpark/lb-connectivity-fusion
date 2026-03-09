#!/usr/bin/env python3
# analyze_oracle_results.py

import argparse
import re
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def parse_from_path(p: str) -> Optional[Tuple[int, int, float, int]]:
    base = Path(p).name
    m = re.search(r"trueR(?P<trueR>\d+)_KMAX(?P<KMAX>\d+)_SNR(?P<SNR>[0-9pm\.]+)_seed(?P<seed>\d+)", base)
    if m is None:
        return None
    d = m.groupdict()
    snr = d["SNR"].replace("p", ".").replace("m", "-")
    return int(d["trueR"]), int(d["KMAX"]), float(snr), int(d["seed"])


def add_metadata(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if all(c in df.columns for c in ["trueR", "KMAX", "SNR", "seed"]):
        return df
    if "file" not in df.columns:
        raise ValueError("Need a file column to parse metadata")
    parsed = df["file"].apply(parse_from_path)
    if parsed.isnull().any():
        bad = df.loc[parsed.isnull(), "file"].head(5).tolist()
        raise ValueError("Could not parse metadata for some rows")
    vals = parsed.tolist()
    df["trueR"] = [t[0] for t in vals]
    df["KMAX"] = [t[1] for t in vals]
    df["SNR"] = [t[2] for t in vals]
    df["seed"] = [t[3] for t in vals]
    return df


def scenario_order(df: pd.DataFrame) -> List[Tuple[int, float]]:
    scen = df[["KMAX", "SNR"]].drop_duplicates().sort_values(["KMAX", "SNR"])
    return [(int(r.KMAX), float(r.SNR)) for r in scen.itertuples(index=False)]


def scenario_label(kmax: int, snr: float) -> str:
    return f"SNR={snr:g}\n$k^{{\\mathrm{{EEG}}}}_{{\\max}}={kmax}$"


def add_scenario_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["scenario_pair"] = list(zip(df["KMAX"].astype(int), df["SNR"].astype(float)))
    df["scenario_label"] = df.apply(lambda r: scenario_label(int(r.KMAX), float(r.SNR)), axis=1)
    return df


def _apply_common_axes(ax, y_label: str, ylim: Optional[Tuple[float, float]] = None, title: Optional[str] = None):
    ax.set_ylabel(y_label)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if title is not None:
        ax.set_title(title)


def plot_box_with_jitter(
    ax,
    data_by_group: List[np.ndarray],
    tick_labels: List[str],
    y_label: str,
    ylim: Optional[Tuple[float, float]],
    title: str,
    jitter_seed: int = 0,
):
    ax.boxplot(data_by_group, tick_labels=tick_labels, showfliers=False)

    rng = np.random.default_rng(jitter_seed)
    for j, vals in enumerate(data_by_group):
        if vals.size == 0:
            continue
        x = (j + 1) + (rng.random(vals.size) - 0.5) * 0.18
        ax.scatter(x, vals, s=8)

    ax.tick_params(axis="x", labelrotation=0)
    _apply_common_axes(ax, y_label=y_label, ylim=ylim, title=title)


def scenario_summary_mean_sd(df: pd.DataFrame, metric_cols: List[str]) -> pd.DataFrame:
    g = df.groupby(["trueR", "KMAX", "SNR"])
    out = g[metric_cols].agg(["count", "mean", "std"]).reset_index()
    return out


def main():
    ap = argparse.ArgumentParser("Analyze oracle-fit summary_oracle.csv")
    ap.add_argument("--summary_csv", type=str, required=True, help="Path to summary_oracle.csv")
    ap.add_argument("--outdir", type=str, required=True, help="Output directory")
    ap.add_argument("--formats", type=str, default="png,pdf", help="Comma-separated output formats")
    ap.add_argument("--jitter_seed", type=int, default=0)
    ap.add_argument("--ylim_relerr", type=str, default="", help="e.g. 0,0.6")
    ap.add_argument("--ylim_dsub", type=str, default="", help="e.g. 0,1.5")
    ap.add_argument("--ylim_angle", type=str, default="", help="e.g. 0,30")
    args = ap.parse_args()

    summary_csv = Path(args.summary_csv).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    ensure_dir(outdir)

    fmts = [x.strip() for x in args.formats.split(",") if x.strip()]

    def parse_ylim(s: str) -> Optional[Tuple[float, float]]:
        if not s.strip():
            return None
        a, b = s.split(",")
        return (float(a), float(b))

    ylim_relerr = parse_ylim(args.ylim_relerr)
    ylim_dsub = parse_ylim(args.ylim_dsub)
    ylim_angle = parse_ylim(args.ylim_angle)

    df = pd.read_csv(summary_csv)
    df = add_metadata(df)
    df = add_scenario_cols(df)

    rename_map = {}
    if "dsub_full" not in df.columns and "dsub" in df.columns:
        rename_map["dsub"] = "dsub_full"
    if "angle_mean_deg_full" not in df.columns and "angle_mean_deg" in df.columns:
        rename_map["angle_mean_deg"] = "angle_mean_deg_full"
    if "angle_max_deg_full" not in df.columns and "angle_max_deg" in df.columns:
        rename_map["angle_max_deg"] = "angle_max_deg_full"
    if rename_map:
        df = df.rename(columns=rename_map)

    df.to_csv(outdir / "oracle_rows_enriched.csv", index=False)

    metrics = [
        "RelErr_obs_mean",
        "dsub_full",
        "angle_mean_deg_full",
        "rmse_lambda",
        "procrustes_offdiag_mean",
        "RelErr_eeg_miss_mean",
        "RelErr_fmri_miss_mean",
    ]
    metrics = [m for m in metrics if m in df.columns]
    if not metrics:
        raise ValueError("No recognizable oracle metrics found")

    summ = scenario_summary_mean_sd(df, metrics)
    summ.to_csv(outdir / "oracle_scenario_summary_mean_sd.csv", index=False)

    plot_list_main = ["RelErr_obs_mean"]
    plot_list_supp = ["dsub_full", "angle_mean_deg_full", "RelErr_eeg_miss_mean", "RelErr_fmri_miss_mean"]
    plot_list_main = [m for m in plot_list_main if m in df.columns]
    plot_list_supp = [m for m in plot_list_supp if m in df.columns]

    scen_pairs = scenario_order(df)
    tick_labels = [scenario_label(kmax, snr) for (kmax, snr) in scen_pairs]

    trueRs = sorted(df["trueR"].unique())

    def make_panel_plot(metric: str, ylim: Optional[Tuple[float, float]], ylab: str, fname_prefix: str):
        fig, axes = plt.subplots(len(trueRs), 1, figsize=(10, 3.6 * len(trueRs)), constrained_layout=True)
        if len(trueRs) == 1:
            axes = [axes]

        for ax, tr in zip(axes, trueRs):
            dsub = df[df["trueR"] == tr].copy()
            data_by_group = []
            for (kmax, snr) in scen_pairs:
                vals = dsub.loc[
                    (dsub["KMAX"] == kmax) & (np.isclose(dsub["SNR"], snr)),
                    metric
                ].dropna().values
                data_by_group.append(vals)

            plot_box_with_jitter(
                ax=ax,
                data_by_group=data_by_group,
                tick_labels=tick_labels,
                y_label=ylab,
                ylim=ylim,
                title=f"True rank = {int(tr)}",
                jitter_seed=args.jitter_seed,
            )

        for ext in fmts:
            outpath = outdir / f"{fname_prefix}.{ext}"
            fig.savefig(outpath, dpi=250)
        plt.close(fig)

    if "RelErr_obs_mean" in plot_list_main:
        make_panel_plot(
            metric="RelErr_obs_mean",
            ylim=ylim_relerr,
            ylab=r"Observed-support error $\mathrm{RelErr}(\mathcal{R}_{\mathrm{obs}})$",
            fname_prefix="Fig_oracle_RelErr_obs_box_jitter",
        )

    if "dsub_full" in plot_list_supp:
        make_panel_plot(
            metric="dsub_full",
            ylim=ylim_dsub,
            ylab=r"Subspace error $d_{\mathrm{sub}}$",
            fname_prefix="Fig_oracle_dsub_box_jitter",
        )
    if "angle_mean_deg_full" in plot_list_supp:
        make_panel_plot(
            metric="angle_mean_deg_full",
            ylim=ylim_angle,
            ylab=r"Mean principal angle (degrees)",
            fname_prefix="Fig_oracle_anglemean_box_jitter",
        )
    if "RelErr_eeg_miss_mean" in plot_list_supp:
        make_panel_plot(
            metric="RelErr_eeg_miss_mean",
            ylim=ylim_relerr,
            ylab=r"EEG-missing error $\mathrm{RelErr}(\mathcal{R}_{\mathrm{EEG\text{-}miss}})$",
            fname_prefix="Fig_oracle_RelErr_eegmiss_box_jitter",
        )
    if "RelErr_fmri_miss_mean" in plot_list_supp:
        make_panel_plot(
            metric="RelErr_fmri_miss_mean",
            ylim=ylim_relerr,
            ylab=r"fMRI-missing error $\mathrm{RelErr}(\mathcal{R}_{\mathrm{fMRI\text{-}miss}})$",
            fname_prefix="Fig_oracle_RelErr_fmrimiss_box_jitter",
        )

    print("Done. Outputs written to:", outdir)
    print(" -", outdir / "oracle_rows_enriched.csv")
    print(" -", outdir / "oracle_scenario_summary_mean_sd.csv")


if __name__ == "__main__":
    main()