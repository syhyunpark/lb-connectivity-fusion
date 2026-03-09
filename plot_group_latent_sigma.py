#!/usr/bin/env python3
"""
plot_group_latent_sigma.py

Plot group-typical latent connectivity summaries from a fitted model.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_npz(path: str) -> dict:
    return dict(np.load(path, allow_pickle=True))


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def nearest_freq_indices(omega: np.ndarray, freqs: List[float]) -> List[int]:
    idx = []
    for f in freqs:
        idx.append(int(np.argmin(np.abs(omega - f))))
    return idx


def compute_group_lambda(
    lambda_hat: np.ndarray,
    summary: str = "median",
) -> np.ndarray:
    if lambda_hat.ndim != 3:
        raise ValueError("lambda_hat must have shape (n,F,R)")
    if summary == "median":
        return np.median(lambda_hat, axis=0)
    if summary == "mean":
        return np.mean(lambda_hat, axis=0)
    raise ValueError("Unknown summary")


def sigma_from_phi_lambda(Phi: np.ndarray, lam_r: np.ndarray) -> np.ndarray:
    return (Phi * lam_r[None, :]) @ Phi.T


def compute_group_sigma(Phi: np.ndarray, tilde_lambda: np.ndarray) -> np.ndarray:
    F = tilde_lambda.shape[0]
    K = Phi.shape[0]
    out = np.zeros((F, K, K), dtype=float)
    for f in range(F):
        out[f] = sigma_from_phi_lambda(Phi, tilde_lambda[f])
    return out


def plot_diag_heatmap(
    sigma_diag: np.ndarray,
    omega: np.ndarray,
    out_pdf: Path,
    also_png: bool,
    title: str,
):
    K, F = sigma_diag.shape

    fig = plt.figure(figsize=(6.8, 4.2))
    ax = plt.gca()

    im = ax.imshow(
        sigma_diag,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[omega[0], omega[-1], 1, K],
    )
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("LB mode index $k$")
    ax.set_title(title)

    cb = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label(r"$\widetilde{\Sigma}(k,k;\omega)$")

    plt.tight_layout()
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    if also_png:
        fig.savefig(out_pdf.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_sigma_slice(
    Sigma: np.ndarray,
    freq_hz: float,
    out_pdf: Path,
    also_png: bool,
    title: str,
    vmax: Optional[float] = None,
):
    K = Sigma.shape[0]
    if vmax is None:
        vmax = float(np.percentile(np.abs(Sigma[np.isfinite(Sigma)]), 99.0))
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = float(np.max(np.abs(Sigma))) if np.isfinite(Sigma).any() else 1.0

    fig = plt.figure(figsize=(4.2, 3.8))
    ax = plt.gca()
    im = ax.imshow(
        Sigma,
        origin="lower",
        interpolation="nearest",
        aspect="equal",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        extent=[1, K, 1, K],
    )
    ax.set_xlabel(r"$k'$")
    ax.set_ylabel(r"$k$")
    ax.set_title(f"{title}\n{freq_hz:.2f} Hz")

    cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label(r"$\widetilde{\Sigma}(k,k';\omega)$")

    plt.tight_layout()
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    if also_png:
        fig.savefig(out_pdf.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_combined_figure(
    sigma_diag: np.ndarray,
    sigma_all: np.ndarray,
    omega: np.ndarray,
    slice_idx: List[int],
    out_pdf: Path,
    also_png: bool,
    main_title: str,
    combined_wspace: float = 0.42,
):
    K, F = sigma_diag.shape
    n_slices = len(slice_idx)
    if n_slices not in (2, 3):
        raise ValueError("Need 2 or 3 slice frequencies")

    slabs = np.stack([sigma_all[idx] for idx in slice_idx], axis=0)
    vmax_slice = float(np.percentile(np.abs(slabs[np.isfinite(slabs)]), 99.0))
    if not np.isfinite(vmax_slice) or vmax_slice <= 0:
        vmax_slice = float(np.max(np.abs(slabs))) if np.isfinite(slabs).any() else 1.0

    fig_w = 11.5 if n_slices == 2 else 14.0
    fig = plt.figure(figsize=(fig_w, 4.2))
    width_ratios = [1.45] + [1.0] * n_slices
    gs = fig.add_gridspec(1, 1 + n_slices, width_ratios=width_ratios, wspace=combined_wspace)

    ax0 = fig.add_subplot(gs[0, 0])
    im0 = ax0.imshow(
        sigma_diag,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[omega[0], omega[-1], 1, K],
    )
    ax0.set_xlabel("Frequency (Hz)")
    ax0.set_ylabel("LB mode index $k$")
    ax0.set_title(r"Diagonal summary: $\widetilde{\Sigma}(k,k;\omega)$")
    cb0 = fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.02)
    cb0.set_label("Value")

    for j, idx in enumerate(slice_idx):
        ax = fig.add_subplot(gs[0, j + 1])
        im = ax.imshow(
            sigma_all[idx],
            origin="lower",
            interpolation="nearest",
            aspect="equal",
            cmap="RdBu_r",
            vmin=-vmax_slice,
            vmax=vmax_slice,
            extent=[1, K, 1, K],
        )
        ax.set_xlabel(r"$k'$")
        if j == 0:
            ax.set_ylabel(r"$k$", labelpad=1)
        else:
            ax.set_yticklabels([])
        ax.set_title(f"{omega[idx]:.2f} Hz")

    if n_slices == 2:
        cax = fig.add_axes([0.915, 0.18, 0.012, 0.64])
    else:
        cax = fig.add_axes([0.92, 0.18, 0.012, 0.64])

    sm = plt.cm.ScalarMappable(cmap="RdBu_r")
    sm.set_clim(-vmax_slice, vmax_slice)
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(r"$\widetilde{\Sigma}(k,k';\omega)$")

    if main_title:
        fig.suptitle(main_title, y=0.995, fontsize=13)

    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    if also_png:
        fig.savefig(out_pdf.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit_npz", required=True, help="fit NPZ containing Phi_hat and lambda_hat")
    ap.add_argument("--data_npz", default="", help="optional data NPZ containing omega")
    ap.add_argument("--outdir", required=True)
    ap.add_argument(
        "--summary",
        choices=["median", "mean"],
        default="median",
        help="how to summarize lambda_hat across subjects",
    )
    ap.add_argument(
        "--freqs",
        default="4,10,30",
        help="comma-separated frequencies (Hz) for KxK slices",
    )
    ap.add_argument("--also_write_png", action="store_true")
    ap.add_argument("--title", default="Group-typical latent spatio-spectral connectivity field")
    ap.add_argument(
        "--combined_wspace",
        type=float,
        default=0.42,
        help="horizontal spacing in the combined figure",
    )

    args = ap.parse_args()

    outdir = Path(args.outdir).expanduser().resolve()
    ensure_dir(outdir)

    fit = load_npz(args.fit_npz)
    if "Phi_hat" not in fit or "lambda_hat" not in fit:
        raise KeyError("fit NPZ must contain Phi_hat and lambda_hat")

    Phi = np.asarray(fit["Phi_hat"], dtype=float)
    lambda_hat = np.asarray(fit["lambda_hat"], dtype=float)

    n, F, R = lambda_hat.shape
    K = Phi.shape[0]
    if Phi.shape[1] != R:
        raise ValueError("Phi_hat and lambda_hat shapes do not match")

    if args.data_npz.strip():
        data = load_npz(args.data_npz)
        if "omega" in data:
            omega = np.asarray(data["omega"], dtype=float).reshape(-1)
        else:
            raise KeyError("No omega found in data NPZ")
    else:
        omega = np.arange(1, F + 1, dtype=float)

    if omega.size != F:
        raise ValueError("omega length does not match lambda_hat")

    tilde_lambda = compute_group_lambda(lambda_hat, summary=args.summary)
    sigma_all = compute_group_sigma(Phi, tilde_lambda)
    sigma_diag = np.stack([np.diag(sigma_all[f]) for f in range(F)], axis=1)

    freq_targets = parse_float_list(args.freqs)
    if len(freq_targets) not in (2, 3):
        raise ValueError("--freqs must contain 2 or 3 values")
    idx_sel = nearest_freq_indices(omega, freq_targets)
    freq_sel = [float(omega[i]) for i in idx_sel]

    np.savez_compressed(
        outdir / "sigma_group_typical_arrays.npz",
        Phi_hat=Phi,
        lambda_tilde=tilde_lambda,
        sigma_all=sigma_all,
        sigma_diag=sigma_diag,
        omega=omega,
        freq_idx_selected=np.asarray(idx_sel, dtype=int),
        freq_selected=np.asarray(freq_sel, dtype=float),
        summary=np.array(args.summary, dtype=object),
    )

    plot_diag_heatmap(
        sigma_diag=sigma_diag,
        omega=omega,
        out_pdf=outdir / "sigma_diag_heatmap.pdf",
        also_png=bool(args.also_write_png),
        title=rf"Diagonal latent field summary ({args.summary} across subjects)",
    )

    for idx, f_hz in zip(idx_sel, freq_sel):
        tag = f"{f_hz:.2f}".replace(".", "p")
        plot_sigma_slice(
            Sigma=sigma_all[idx],
            freq_hz=f_hz,
            out_pdf=outdir / f"sigma_slice_{tag}Hz.pdf",
            also_png=bool(args.also_write_png),
            title=rf"Latent field slice: $\widetilde{{\Sigma}}(k,k';\omega)$",
        )

    plot_combined_figure(
        sigma_diag=sigma_diag,
        sigma_all=sigma_all,
        omega=omega,
        slice_idx=idx_sel,
        out_pdf=outdir / "sigma_summary_figure.pdf",
        also_png=bool(args.also_write_png),
        main_title=args.title,
        combined_wspace=float(args.combined_wspace),
    )

    print("Wrote outputs to:", outdir)
    print(" -", outdir / "sigma_group_typical_arrays.npz")
    print(" -", outdir / "sigma_diag_heatmap.pdf")
    for f_hz in freq_sel:
        tag = f"{f_hz:.2f}".replace(".", "p")
        print(" -", outdir / f"sigma_slice_{tag}Hz.pdf")
    print(" -", outdir / "sigma_summary_figure.pdf")


if __name__ == "__main__":
    main()