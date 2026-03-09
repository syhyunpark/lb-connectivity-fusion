#!/usr/bin/env python3
# analyze_ard_results.py

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_json(path: Path) -> Optional[dict]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def require_key(d: dict, key: str, path: Path):
    if key not in d:
        raise KeyError(f"Missing key '{key}'")


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def th_tag(x: float) -> str:
    s = f"{x:g}"
    return s.replace(".", "p").replace("-", "m")


def recompute_R_eff_from_energy(energy_r: List[float], rel_thresh: float) -> int:
    E = np.asarray(energy_r, dtype=float)
    if E.size == 0:
        return 0
    Emax = float(np.max(E))
    if Emax <= 0:
        return 0
    return int(np.sum((E / (Emax + 1e-12)) >= rel_thresh))


def add_scenario_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["scenario_key"] = df.apply(lambda r: (int(r.KMAX), float(r.SNR)), axis=1)
    return df


def scenario_order(df: pd.DataFrame) -> List[Tuple[int, float]]:
    scen = df[["KMAX", "SNR"]].drop_duplicates().sort_values(["KMAX", "SNR"])
    return [(int(r.KMAX), float(r.SNR)) for r in scen.itertuples(index=False)]


def scenario_ticklabels(scen_list: List[Tuple[int, float]]) -> List[str]:
    labs = []
    for kmax, snr in scen_list:
        labs.append(f"SNR={snr:g}\n$k_{{\\max}}^{{\\mathrm{{EEG}}}}={kmax}$")
    return labs


def save_formats(fig: plt.Figure, out_base: Path, formats: List[str]) -> None:
    for fmt in formats:
        fig.savefig(out_base.with_suffix(f".{fmt}"), dpi=300, bbox_inches="tight")


def parse_int_list(s: str) -> List[int]:
    s = s.strip()
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _style_axes(ax):
    ax.grid(True, linewidth=0.6, alpha=0.3)
    for side in ["top", "right", "bottom", "left"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.8)


def _apply_common_axes(
    ax: plt.Axes,
    *,
    y_label: str,
    ylim,
    trueR: int,
    xticklabels: List[str],
    xtick_positions,
) -> None:
    ax.set_ylabel(y_label)
    ax.axhline(trueR, linestyle="--", linewidth=1)
    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xticklabels, rotation=0, ha="center")
    ax.tick_params(axis="x", labelbottom=True)


def load_eval_ard(eval_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    files = sorted(eval_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError("No JSON files found")

    for fp in files:
        j = load_json(fp)
        if j is None:
            continue

        for k in ["trueR", "KMAX", "SNR", "seed"]:
            require_key(j, k, fp)

        ard = j.get("ard", {})
        if not isinstance(ard, dict):
            ard = {}

        row: Dict[str, Any] = {
            "file": str(fp),
            "trueR": int(j["trueR"]),
            "KMAX": int(j["KMAX"]),
            "SNR": float(j["SNR"]),
            "seed": int(j["seed"]),
            "is_ard": bool(j.get("is_ard", True)),
        }

        energy_r = ard.get("energy_r", None)
        if energy_r is None:
            raise KeyError("Missing ard.energy_r")
        row["energy_r"] = energy_r

        row["Rmax"] = int(ard.get("Rmax", j.get("Rmax", np.nan))) if (
            ard.get("Rmax", None) is not None or j.get("Rmax", None) is not None
        ) else np.nan
        row["tau_min"] = ard.get("tau_min", np.nan)
        row["tau_med"] = ard.get("tau_med", np.nan)
        row["tau_max"] = ard.get("tau_max", np.nan)

        rows.append(row)

    df = pd.DataFrame(rows)
    df = add_scenario_label(df)
    return df


def scenario_summary(df: pd.DataFrame, Rcol: str) -> pd.DataFrame:
    g = df.groupby(["trueR", "KMAX", "SNR"])
    out = g[Rcol].agg(N="count", median="median", mean="mean", std="std").reset_index()
    out["exact_rate"] = g.apply(lambda x: (x[Rcol] == x["trueR"]).mean()).values
    out["within1_rate"] = g.apply(lambda x: (abs(x[Rcol] - x["trueR"]) <= 1).mean()).values
    out["MAE"] = g.apply(lambda x: (abs(x[Rcol] - x["trueR"])).mean()).values
    out["RMSE"] = g.apply(lambda x: np.sqrt(((x[Rcol] - x["trueR"]) ** 2).mean())).values
    return out


def plot_box_with_jitter(
    df: pd.DataFrame,
    out_base: Path,
    ycol: str,
    *,
    ylim,
    formats: List[str],
    jitter_seed: int,
):
    rng = np.random.default_rng(jitter_seed)

    trueRs = sorted(df["trueR"].unique())
    scen_list = scenario_order(df)
    xtlabs = scenario_ticklabels(scen_list)

    fig, axes = plt.subplots(len(trueRs), 1, figsize=(11.0, 3.6 * len(trueRs)), sharex=True)
    if len(trueRs) == 1:
        axes = [axes]

    scatter_color = "C0"

    for ax, tr in zip(axes, trueRs):
        _style_axes(ax)
        d = df[df["trueR"] == tr].copy()

        data = [d.loc[d["scenario_key"] == s, ycol].values for s in scen_list]
        positions = np.arange(1, len(scen_list) + 1)

        ax.boxplot(
            data,
            positions=positions,
            widths=0.50,
            showfliers=False,
            patch_artist=False,
        )

        for j, s in enumerate(scen_list):
            y = d.loc[d["scenario_key"] == s, ycol].to_numpy()
            if y.size == 0:
                continue
            x0 = positions[j]
            x = x0 + (rng.random(y.size) - 0.5) * 0.25
            ax.scatter(x, y, s=10, alpha=0.65, color=scatter_color)

        ax.set_title(f"True rank = {int(tr)}")

        _apply_common_axes(
            ax,
            y_label="Estimated effective rank",
            ylim=ylim,
            trueR=int(tr),
            xticklabels=xtlabs,
            xtick_positions=positions,
        )

    fig.tight_layout()
    save_formats(fig, out_base, formats)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser("Analyze ARD eval JSONs and make box+jitter plots")
    ap.add_argument("--eval_dir", type=str, required=True)
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument(
        "--thresholds",
        type=str,
        default="0.01,0.02,0.05,0.005",
        help="Comma-separated thresholds for R_eff from energy_r/Emax",
    )
    ap.add_argument(
        "--trueR_keep",
        type=str,
        default="5,10",
        help="Comma-separated true ranks to keep",
    )
    ap.add_argument(
        "--ylim",
        type=str,
        default="",
        help="Optional y-limits as 'low,high'",
    )
    ap.add_argument(
        "--formats",
        type=str,
        default="png",
        help="Comma-separated output formats",
    )
    ap.add_argument("--jitter_seed", type=int, default=0)
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    ensure_dir(outdir)

    formats = [x.strip() for x in args.formats.split(",") if x.strip()]
    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]

    ylim = None
    if args.ylim.strip():
        lo, hi = [float(x.strip()) for x in args.ylim.split(",")]
        ylim = (lo, hi)

    df_all = load_eval_ard(eval_dir)
    df_all.to_csv(outdir / "ard_rows_raw.csv", index=False)

    keep_trueR = set(parse_int_list(args.trueR_keep))
    df = df_all[df_all["trueR"].isin(keep_trueR)].copy() if keep_trueR else df_all.copy()
    if df.empty:
        raise RuntimeError("No rows left after filtering")

    df.to_csv(outdir / "ard_rows_filtered.csv", index=False)

    tag_trueR = args.trueR_keep.replace(",", "_").replace(" ", "")

    for th in thresholds:
        tag = th_tag(th)
        col = f"R_eff_th{tag}"
        df[col] = df["energy_r"].apply(lambda er: recompute_R_eff_from_energy(er, th)).astype(int)

        summ = scenario_summary(df, Rcol=col)
        summ.to_csv(outdir / f"summary_{col}.csv", index=False)

        plot_box_with_jitter(
            df,
            outdir / f"Fig_{col}_boxscatter_trueR{tag_trueR}",
            ycol=col,
            ylim=ylim,
            formats=formats,
            jitter_seed=args.jitter_seed,
        )

    print("Done. Wrote outputs to:", outdir)
    print("Key files:")
    print(" -", outdir / "ard_rows_raw.csv")
    print(" -", outdir / "ard_rows_filtered.csv")
    for th in thresholds:
        print(" -", outdir / f"summary_R_eff_th{th_tag(th)}.csv")


if __name__ == "__main__":
    main()