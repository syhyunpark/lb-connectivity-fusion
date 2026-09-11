#!/usr/bin/env python3
"""
rank_screening_summary.py

Summarize compact rank-screening robustness simulations and recompute effective rank at
multiple energy thresholds without refitting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def R_eff(energy: Any, threshold: float) -> int:
    E = np.asarray(energy, dtype=float)
    if E.size == 0 or np.max(E) <= 0:
        return 0
    return int(np.sum(E / (np.max(E) + 1e-12) >= threshold))


def load_rows(eval_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    files = sorted(eval_dir.glob("*_eval.json"))
    if not files:
        raise FileNotFoundError(f"No evaluation JSONs in {eval_dir}")
    scalar_keys = [
        "trueR", "Rmax", "n", "K", "F", "KMAX", "SNR", "seed",
        "noise_model", "t_df", "hetero_gamma",
        "topRtrue_projection_error_relative",
        "topRtrue_angle_mean_deg", "topRtrue_angle_max_deg",
        "topRtrue_matched_abs_inner_mean",
        "RelErr_eeg_observed", "RelErr_eeg_missing",
        "RelErr_fmri", "RelErr_combined",
    ]
    for fp in files:
        with open(fp) as f:
            j = json.load(f)
        row = {"file": str(fp)}
        for key in scalar_keys:
            row[key] = j.get(key, np.nan)
        row["energy_r"] = j["energy_r"]
        rows.append(row)
    return pd.DataFrame(rows)


def grouped_numeric(df: pd.DataFrame, by: List[str], cols: List[str]) -> pd.DataFrame:
    pieces = []
    g = df.groupby(by, dropna=False)
    for col in cols:
        s = g[col].agg(["count", "median", "mean", "std"]).reset_index()
        s.columns = by + [
            f"{col}_count", f"{col}_median",
            f"{col}_mean", f"{col}_std"
        ]
        pieces.append(s)
    out = pieces[0]
    for p in pieces[1:]:
        out = out.merge(p, on=by, how="outer")
    return out


def save_fig(fig: plt.Figure, base: Path, formats: List[str]) -> None:
    for fmt in formats:
        fig.savefig(base.with_suffix("." + fmt), dpi=300, bbox_inches="tight")
    plt.close(fig)


def boxplot_metric(
    df: pd.DataFrame,
    y: str,
    ylabel: str,
    outbase: Path,
    formats: List[str],
    horizontal_truth: bool = False,
) -> None:
    trueRs = sorted(df["trueR"].unique())
    ns = sorted(df["n"].unique())
    noises = list(dict.fromkeys(df["noise_model"].tolist()))
    fig, axes = plt.subplots(
        len(trueRs), len(ns),
        figsize=(4.4 * len(ns), 3.4 * len(trueRs)),
        squeeze=False,
    )
    rng = np.random.default_rng(0)
    for ir, tr in enumerate(trueRs):
        for jn, n in enumerate(ns):
            ax = axes[ir, jn]
            d = df[(df.trueR == tr) & (df.n == n)]
            data = [d.loc[d.noise_model == nm, y].dropna().values for nm in noises]
            pos = np.arange(1, len(noises) + 1)
            ax.boxplot(data, positions=pos, widths=0.55, showfliers=False)
            for j, nm in enumerate(noises):
                vals = d.loc[d.noise_model == nm, y].dropna().to_numpy()
                if vals.size:
                    x = pos[j] + (rng.random(vals.size) - 0.5) * 0.22
                    ax.scatter(x, vals, s=11, alpha=0.65)
            if horizontal_truth:
                ax.axhline(tr, linestyle="--", linewidth=1)
            ax.set_xticks(pos)
            ax.set_xticklabels(noises, rotation=20, ha="right")
            ax.set_title(f"True rank={tr}, n={n}")
            ax.set_ylabel(ylabel)
            ax.grid(True, linewidth=0.5, alpha=0.3)
            for side in ["top", "right", "bottom", "left"]:
                ax.spines[side].set_visible(True)
    fig.tight_layout()
    save_fig(fig, outbase, formats)


def threshold_bias_plot(summary: pd.DataFrame, outbase: Path, formats: List[str]) -> None:
    trueRs = sorted(summary.trueR.unique())
    ns = sorted(summary.n.unique())
    noises = list(dict.fromkeys(summary.noise_model.tolist()))
    fig, axes = plt.subplots(
        len(trueRs), len(ns),
        figsize=(4.5 * len(ns), 3.5 * len(trueRs)),
        squeeze=False,
    )
    for ir, tr in enumerate(trueRs):
        for jn, n in enumerate(ns):
            ax = axes[ir, jn]
            d = summary[(summary.trueR == tr) & (summary.n == n)]
            for nm in noises:
                z = d[d.noise_model == nm].sort_values("threshold")
                ax.plot(z.threshold, z.rank_bias_mean, marker="o", label=nm)
            ax.axhline(0, linestyle="--", linewidth=1)
            ax.set_xscale("log")
            ax.set_xlabel("Energy threshold")
            ax.set_ylabel("Mean rank bias")
            ax.set_title(f"True rank={tr}, n={n}")
            ax.grid(True, linewidth=0.5, alpha=0.3)
            if ir == 0 and jn == len(ns) - 1:
                ax.legend(frameon=False)
    fig.tight_layout()
    save_fig(fig, outbase, formats)


def main() -> None:
    ap = argparse.ArgumentParser("Analyze compact rank-screening robustness simulations.")
    ap.add_argument("--eval_dir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--thresholds", default="0.005,0.01,0.02,0.05")
    ap.add_argument("--formats", default="png,pdf")
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    ensure_dir(outdir)
    formats = [x.strip() for x in args.formats.split(",") if x.strip()]
    thresholds = parse_float_list(args.thresholds)
    df = load_rows(eval_dir)

    long_rows = []
    for _, r in df.iterrows():
        for th in thresholds:
            reff = R_eff(r.energy_r, th)
            long_rows.append(
                {
                    **{k: r[k] for k in [
                        "file", "trueR", "Rmax", "n", "K", "F", "KMAX",
                        "SNR", "seed", "noise_model", "t_df", "hetero_gamma"
                    ]},
                    "threshold": th,
                    "R_eff": reff,
                    "rank_error": reff - int(r.trueR),
                    "abs_rank_error": abs(reff - int(r.trueR)),
                    "exact": int(reff == int(r.trueR)),
                    "within1": int(abs(reff - int(r.trueR)) <= 1),
                }
            )

    rank_long = pd.DataFrame(long_rows)
    df.to_csv(outdir / "ard_robustness_rows.csv", index=False)
    rank_long.to_csv(outdir / "rank_threshold_rows.csv", index=False)

    by_rank = ["trueR", "n", "noise_model", "threshold"]
    g = rank_long.groupby(by_rank, dropna=False)
    rank_summary = g.agg(
        N=("R_eff", "count"),
        R_eff_median=("R_eff", "median"),
        R_eff_mean=("R_eff", "mean"),
        R_eff_std=("R_eff", "std"),
        rank_bias_mean=("rank_error", "mean"),
        rank_bias_median=("rank_error", "median"),
        MAE=("abs_rank_error", "mean"),
        exact_rate=("exact", "mean"),
        within1_rate=("within1", "mean"),
    ).reset_index()
    rank_summary.to_csv(outdir / "rank_screening_summary.csv", index=False)

    leading_cols = [
        "topRtrue_projection_error_relative",
        "topRtrue_angle_mean_deg",
        "topRtrue_angle_max_deg",
        "topRtrue_matched_abs_inner_mean",
    ]
    leading = grouped_numeric(df, ["trueR", "n", "noise_model"], leading_cols)
    leading.to_csv(outdir / "leading_subspace_summary.csv", index=False)

    rec_cols = [
        "RelErr_eeg_observed", "RelErr_eeg_missing",
        "RelErr_fmri", "RelErr_combined",
    ]
    rec = grouped_numeric(df, ["trueR", "n", "noise_model"], rec_cols)
    rec.to_csv(outdir / "reconstruction_summary.csv", index=False)

    primary = rank_long[np.isclose(rank_long.threshold, 0.01)].copy()
    boxplot_metric(
        primary,
        "R_eff",
        "Estimated effective rank",
        outdir / "Fig_effective_rank_threshold_0p01",
        formats,
        horizontal_truth=True,
    )
    boxplot_metric(
        df,
        "topRtrue_projection_error_relative",
        "Leading-subspace projection error",
        outdir / "Fig_leading_subspace_error",
        formats,
    )
    boxplot_metric(
        df,
        "RelErr_combined",
        "Combined latent reconstruction error",
        outdir / "Fig_combined_reconstruction_error",
        formats,
    )
    threshold_bias_plot(
        rank_summary,
        outdir / "Fig_rank_bias_by_threshold",
        formats,
    )

    print("Wrote outputs to:", outdir)
    print("\nPrimary threshold (0.01) compact summary:")
    print(
        rank_summary[np.isclose(rank_summary.threshold, 0.01)]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
