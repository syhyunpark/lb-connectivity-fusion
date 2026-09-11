#!/usr/bin/env python3
"""
observation_model_summary.py

Summarize the redesigned observation-model validation simulations and create  diagnostic
figures for recovery, modality ablations, reconstruction comparisons,  and the cross-modal divergence "negative control".
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = [
    "max_principal_angle_deg",
    "mean_principal_angle_deg",
    "projection_error_rel",
    "eeg_observed_max_principal_angle_deg",
    "eeg_observed_mean_principal_angle_deg",
    "eeg_observed_projection_error_rel",
    "mean_abs_phi_inner_product",
    "eeg_fit_clean_relerr",
    "fmri_fit_clean_relerr",
    "eeg_common_phi_oracle_relerr",
    "eeg_band_truth_oracle_relerr",
    "fmri_common_phi_oracle_relerr",
    "eeg_independent_rankR_psd_relerr",
    "fmri_independent_rankR_psd_relerr",
    "summary_snr_eeg_recomputed",
    "summary_snr_fmri_recomputed",
    "noisy_to_clean_max_principal_angle_deg",
    "eeg_observed_noisy_to_clean_max_principal_angle_deg",
    "fit_to_eeg_bands_mean_angle_deg",
    "eeg_observed_fit_to_bands_mean_angle_deg",
]


def parse_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def save_formats(fig: plt.Figure, outbase: Path, formats: List[str]) -> None:
    for fmt in formats:
        fig.savefig(outbase.with_suffix(f".{fmt}"), bbox_inches="tight", dpi=250)


def long_summary(df: pd.DataFrame, group_cols: List[str], metrics: List[str]) -> pd.DataFrame:
    rows = []
    for key, group in df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        base = dict(zip(group_cols, key))
        for metric in metrics:
            vals = pd.to_numeric(group[metric], errors="coerce").dropna()
            row = dict(base)
            row.update(
                {
                    "metric": metric,
                    "count": int(vals.size),
                    "mean": float(vals.mean()) if vals.size else np.nan,
                    "std": float(vals.std(ddof=1)) if vals.size > 1 else np.nan,
                    "median": float(vals.median()) if vals.size else np.nan,
                    "q25": float(vals.quantile(0.25)) if vals.size else np.nan,
                    "q75": float(vals.quantile(0.75)) if vals.size else np.nan,
                    "min": float(vals.min()) if vals.size else np.nan,
                    "max": float(vals.max()) if vals.size else np.nan,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def boxplot_by_scenario(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    outbase: Path,
    formats: List[str],
    variants: List[str] | None = None,
) -> None:
    d = df.copy()
    if variants is not None:
        d = d[d["fit_variant"].isin(variants)]
    labels = []
    values = []
    for (scenario, variant), group in d.groupby(["scenario_label", "fit_variant"], sort=False):
        vals = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy()
        if vals.size:
            labels.append(f"{scenario}\n{variant}")
            values.append(vals)
    if not values:
        return
    fig, ax = plt.subplots(figsize=(max(10, 0.72 * len(labels) + 3), 5.2))
    ax.boxplot(values, tick_labels=labels, showfliers=False)
    rng = np.random.default_rng(0)
    for j, vals in enumerate(values, start=1):
        ax.scatter(j + rng.uniform(-0.08, 0.08, size=vals.size), vals, s=8, alpha=0.5)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    save_formats(fig, outbase, formats)
    plt.close(fig)


def modality_ablation_plot(df: pd.DataFrame, outdir: Path, formats: List[str]) -> None:
    d = df[df["fit_variant"].isin(["fused", "eeg_only", "fmri_only"])].copy()
    d = d[d["scenario_label"].isin(["raw_aecov_snr4", "raw_aecov_broadtruth_snr4"])]
    if d.empty:
        return

    grouped = d.groupby(["scenario_label", "fit_variant"]).agg(
        full_angle=("max_principal_angle_deg", "mean"),
        full_sd=("max_principal_angle_deg", "std"),
        observed_angle=("eeg_observed_max_principal_angle_deg", "mean"),
        observed_sd=("eeg_observed_max_principal_angle_deg", "std"),
    ).reset_index()

    scenarios = list(dict.fromkeys(grouped["scenario_label"].tolist()))
    variants = ["fused", "eeg_only", "fmri_only"]
    x = np.arange(len(scenarios), dtype=float)
    width = 0.24

    fig, ax = plt.subplots(figsize=(max(8, 2.1 * len(scenarios) + 3), 5.0))
    for j, variant in enumerate(variants):
        means, errs = [], []
        for scenario in scenarios:
            row = grouped[(grouped["scenario_label"] == scenario) & (grouped["fit_variant"] == variant)]
            means.append(float(row["full_angle"].iloc[0]) if not row.empty else np.nan)
            errs.append(float(row["full_sd"].iloc[0]) if not row.empty else np.nan)
        ax.bar(x + (j - 1) * width, means, width, yerr=errs, capsize=2, label=variant)
    ax.set_xticks(x, scenarios, rotation=25, ha="right")
    ax.set_ylabel("Full-space maximum principal angle (degrees)")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    save_formats(fig, outdir / "fig_observation_model_modality_ablation_full", formats)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(8, 2.1 * len(scenarios) + 3), 5.0))
    for j, variant in enumerate(variants):
        means, errs = [], []
        for scenario in scenarios:
            row = grouped[(grouped["scenario_label"] == scenario) & (grouped["fit_variant"] == variant)]
            means.append(float(row["observed_angle"].iloc[0]) if not row.empty else np.nan)
            errs.append(float(row["observed_sd"].iloc[0]) if not row.empty else np.nan)
        ax.bar(x + (j - 1) * width, means, width, yerr=errs, capsize=2, label=variant)
    ax.set_xticks(x, scenarios, rotation=25, ha="right")
    ax.set_ylabel("EEG-observed maximum principal angle (degrees)")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    save_formats(fig, outdir / "fig_observation_model_modality_ablation_observed", formats)
    plt.close(fig)


def divergence_plot(df: pd.DataFrame, outdir: Path, formats: List[str]) -> None:
    d = df[(df["scenario"] == "crossmodal_divergence") & (df["fit_variant"] == "fused")].copy()
    if d.empty:
        return
    grouped = d.groupby("topology_angle_deg").agg(
        to_fmri=("mean_principal_angle_deg", "mean"),
        to_fmri_sd=("mean_principal_angle_deg", "std"),
        to_eeg=("fit_to_eeg_bands_mean_angle_deg", "mean"),
        to_eeg_sd=("fit_to_eeg_bands_mean_angle_deg", "std"),
        eeg_error=("eeg_fit_clean_relerr", "mean"),
        fmri_error=("fmri_fit_clean_relerr", "mean"),
    ).reset_index().sort_values("topology_angle_deg")

    fig, ax = plt.subplots(figsize=(6.3, 4.8))
    ax.errorbar(grouped["topology_angle_deg"], grouped["to_fmri"], yerr=grouped["to_fmri_sd"], marker="o", capsize=3, label="Fitted to fMRI truth")
    ax.errorbar(grouped["topology_angle_deg"], grouped["to_eeg"], yerr=grouped["to_eeg_sd"], marker="s", linestyle="--", capsize=3, label="Fitted to EEG truth")
    ax.set_xlabel("True EEG–fMRI subspace separation (degrees)")
    ax.set_ylabel("Mean principal angle (degrees)")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    save_formats(fig, outdir / "fig_observation_model_negative_control_angles", formats)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.3, 4.8))
    ax.plot(grouped["topology_angle_deg"], grouped["eeg_error"], marker="o", label="EEG reconstruction error")
    ax.plot(grouped["topology_angle_deg"], grouped["fmri_error"], marker="s", label="fMRI reconstruction error")
    ax.set_xlabel("True EEG–fMRI subspace separation (degrees)")
    ax.set_ylabel("Clean-summary relative reconstruction error")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    save_formats(fig, outdir / "fig_observation_model_negative_control_reconstruction", formats)
    plt.close(fig)


def benchmark_plot(df: pd.DataFrame, outdir: Path, formats: List[str]) -> None:
    d = df[(df["fit_variant"] == "fused") & df["scenario_label"].str.contains("raw_aecov", na=False)].copy()
    if d.empty:
        return
    grouped = d.groupby("scenario_label").agg(
        fitted=("eeg_fit_clean_relerr", "mean"),
        common_oracle=("eeg_common_phi_oracle_relerr", "mean"),
        band_oracle=("eeg_band_truth_oracle_relerr", "mean"),
        independent=("eeg_independent_rankR_psd_relerr", "mean"),
    ).reset_index()
    x = np.arange(len(grouped))
    width = 0.2
    fig, ax = plt.subplots(figsize=(max(8, 1.45 * len(grouped) + 3), 4.8))
    ax.bar(x - 1.5 * width, grouped["fitted"], width, label="Fitted shared model")
    ax.bar(x - 0.5 * width, grouped["common_oracle"], width, label="Common-Φ oracle")
    ax.bar(x + 0.5 * width, grouped["band_oracle"], width, label="Band-truth oracle")
    ax.bar(x + 1.5 * width, grouped["independent"], width, label="Independent rank-R PSD")
    ax.set_xticks(x, grouped["scenario_label"], rotation=30, ha="right")
    ax.set_ylabel("EEG clean-summary relative reconstruction error")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    save_formats(fig, outdir / "fig_observation_model_reconstruction_benchmarks", formats)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze redesigned observation-model validation simulations")
    ap.add_argument("--summary_csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--manifest", default="")
    ap.add_argument("--formats", default="png,pdf")
    args = ap.parse_args()

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    formats = parse_list(args.formats)
    df = pd.read_csv(args.summary_csv)
    if args.manifest:
        manifest = pd.read_csv(args.manifest)
        keep = [c for c in ["scenario_id", "role"] if c in manifest.columns]
        if keep:
            df = df.merge(manifest[keep], left_on="scenario_label", right_on="scenario_id", how="left")

    df.to_csv(outdir / "observation_model_all_results.csv", index=False)
    metrics = [m for m in METRICS if m in df.columns]
    grouped = long_summary(df, ["scenario_label", "fit_variant"], metrics)
    grouped.to_csv(outdir / "observation_model_grouped_summary.csv", index=False)

    diagnostic_cols = [
        c
        for c in [
            "scenario_label",
            "seed",
            "scenario",
            "summary_snr_target",
            "summary_snr_eeg_realized",
            "summary_snr_fmri_realized",
            "summary_snr_eeg_recomputed",
            "summary_snr_fmri_recomputed",
            "k0_phi",
            "truth_lowk_energy_fraction",
            "mask",
            "eeg_candidate_modes",
            "topology_angle_deg",
            "true_band_to_common_mean_angle_deg",
            "true_band_to_common_max_angle_deg",
        ]
        if c in df.columns
    ]
    df[diagnostic_cols].drop_duplicates(["scenario_label", "seed"]).to_csv(
        outdir / "observation_model_diagnostics.csv", index=False
    )

    boxplot_by_scenario(
        df,
        "max_principal_angle_deg",
        "Full-space maximum principal angle (degrees)",
        outdir / "fig_observation_model_full_recovery",
        formats,
        variants=["fused", "clean_fused"],
    )
    boxplot_by_scenario(
        df,
        "eeg_observed_max_principal_angle_deg",
        "EEG-observed maximum principal angle (degrees)",
        outdir / "fig_observation_model_observed_recovery",
        formats,
    )
    modality_ablation_plot(df, outdir, formats)
    divergence_plot(df, outdir, formats)
    benchmark_plot(df, outdir, formats)

    print("Wrote outputs to:", outdir)


if __name__ == "__main__":
    main()
