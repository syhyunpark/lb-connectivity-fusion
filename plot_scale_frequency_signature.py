#!/usr/bin/env python3
"""
plot_scale_frequency_signature.py

LB effective spatial frequency (x) vs oscillatory signature (y),
one point per learned network r.

Inputs:
  --fit_npz     : fitted model npz (expects Phi_hat, lambda_hat)
  --data_npz    : data npz (expects omega)
  --basis_npz   : fsaverage5_phi_K200_pial_A.npz (expects evals_lh, evals_rh)

Plot one point per network: spatial scale vs oscillatory signature.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def load_npz(path: str) -> dict:
    return dict(np.load(path, allow_pickle=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit_npz", required=True)
    ap.add_argument("--data_npz", required=True)
    ap.add_argument("--basis_npz", required=True)
    ap.add_argument("--out_pdf", required=True)
    ap.add_argument("--out_png", default="")
    ap.add_argument("--K", type=int, default=50, help="K used in fit")
    ap.add_argument(
        "--y_metric",
        choices=["peak", "mu_median"],
        default="mu_median",
        help="peak frequency of median spectrum or median spectral centroid",
    )
    ap.add_argument("--title", default="Oscillatory signature vs spatial scale")
    ap.add_argument("--label_points", action="store_true", help="Annotate points with Net r")
    ap.add_argument("--use_log_x", action="store_true", help="Use log scale on x-axis")
    ap.add_argument("--use_log_y", action="store_true", help="Use log scale on y-axis")
    args = ap.parse_args()

    out_pdf = Path(args.out_pdf).expanduser().resolve()
    ensure_dir(out_pdf.parent)
    out_png = Path(args.out_png).expanduser().resolve() if args.out_png.strip() else None
    if out_png is not None:
        ensure_dir(out_png.parent)

    fit = load_npz(args.fit_npz)
    dat = load_npz(args.data_npz)
    bas = load_npz(args.basis_npz)

    Phi_hat = np.asarray(fit["Phi_hat"], float)
    lam_eeg = np.asarray(fit["lambda_hat"], float)
    n, F, R = lam_eeg.shape

    K = int(args.K)
    if Phi_hat.shape[0] != K:
        raise ValueError("Phi_hat K does not match --K")

    omega = np.asarray(dat.get("omega", np.arange(F, dtype=float)), float).reshape(-1)
    if omega.size != F:
        raise ValueError("omega does not match lambda_hat")

    evals_lh = np.asarray(bas["evals_lh"], float).reshape(-1)
    evals_rh = np.asarray(bas["evals_rh"], float).reshape(-1)
    if evals_lh.size < K or evals_rh.size < K:
        raise ValueError("basis eigenvalues shorter than K")

    lam_lb = 0.5 * (evals_lh[:K] + evals_rh[:K])

    phi = Phi_hat
    num = (lam_lb[:, None] * (phi ** 2)).sum(axis=0)
    den = (phi ** 2).sum(axis=0) + 1e-12
    s_r = num / den

    med_curve = np.median(lam_eeg, axis=0)
    if args.y_metric == "peak":
        peak_idx = np.argmax(med_curve, axis=0)
        y_r = omega[peak_idx]
        y_name = r"Peak frequency of median spectrum (Hz)"
    else:
        denom_mu = lam_eeg.sum(axis=1) + 1e-12
        mu_ir = (lam_eeg * omega[None, :, None]).sum(axis=1) / denom_mu
        y_r = np.median(mu_ir, axis=0)
        y_name = r"Median spectral centroid (Hz)"

    out_csv = out_pdf.with_suffix(".csv")
    pd.DataFrame({
        "r": np.arange(1, R + 1),
        "s_r": s_r,
        "y_r": y_r,
    }).to_csv(out_csv, index=False)
    print("Wrote:", out_csv)

    fig = plt.figure(figsize=(7.0, 5.0))
    ax = plt.gca()

    ax.scatter(s_r, y_r, s=45, edgecolors="black", linewidths=0.5, alpha=0.9, zorder=3)

    if args.label_points:
        for r in range(R):
            ax.annotate(
                f"Net {r+1}",
                (s_r[r], y_r[r]),
                textcoords="offset points",
                xytext=(4, 3),
                fontsize=8,
                zorder=4,
            )

    if args.use_log_x:
        ax.set_xscale("log")
    if args.use_log_y:
        ax.set_yscale("log")

    ax.set_xlabel(r"Effective spatial frequency $s_r$ (LB-eigenvalue weighted)")
    ax.set_ylabel(y_name)
    ax.set_title(args.title)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    fig.savefig(out_pdf, dpi=300)
    print("Wrote:", out_pdf)
    if out_png is not None:
        fig.savefig(out_png, dpi=300)
        print("Wrote:", out_png)
    plt.close(fig)


if __name__ == "__main__":
    main()