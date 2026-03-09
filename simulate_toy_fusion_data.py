#!/usr/bin/env python3
"""
simulate_toy_fusion_data.py

Generate a small synthetic LB-domain EEG-fMRI connectivity dataset compatible with
fit_map_fixedR_cov_fast.py.

Writes an NPZ with the fields expected by the fitter:
  - C_eeg         (n, F, K, K)
  - C_fmri        (n, K, K)
  - W_eeg         (K,  K)
  - W_fmri       (K, K)
  - m_fmri       (indices of low-frequency bins used by aggregate summaries)
  - w_m          (frequency weights; fitter normalizes over m_fmri)
  - sigma_eeg
  - sigma_fmri
  - include_diag
  - omega
  - subject_ids

Also stores latent truth:
  - Phi_true
  - lambda_eeg_true
  - lambda_fmri_true
  - trueR, KMAX, SNR, seed
  - generate_fmri_mode


True factors Phi_true are low-k concentrated (bandwidth-aware); and EEG spectral profiles vary smoothly across subjects 
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.linalg as la


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def sym(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.T)


def frob(A: np.ndarray) -> float:
    return float(np.sqrt(np.sum(A * A)))


def orthonormal_columns(A: np.ndarray) -> np.ndarray:
    Q, _ = la.qr(A, mode="economic")
    return Q


def sigma_from_phi_lambda(Phi: np.ndarray, lam: np.ndarray) -> np.ndarray:
    return (Phi * lam[None, :]) @ Phi.T


def add_scaled_symmetric_noise(S: np.ndarray, snr: float, rng: np.random.Generator) -> np.ndarray:
    """
    Add symmetric Gaussian noise N so that ||S||_F / ||N||_F ~= snr.
    """
    N = rng.normal(size=S.shape)
    N = sym(N)
    sS = frob(S)
    sN = frob(N)
    if sN <= 0:
        return S.copy()
    scale = sS / max(snr, 1e-8) / sN
    return S + scale * N


def make_lowk_factors(K: int, R_true: int, rng: np.random.Generator, phi_decay: float = 0.12) -> np.ndarray:
    """
    Construct low-k concentrated factor loadings over LB mode index k.

    We combine:
      - a coarse low-k Gaussian bump for each factor
      - exponentially decaying random perturbations over k
    and then orthonormalize the columns.

    Returns Phi_true: (K, R_true)
    """
    kk = np.arange(1, K + 1, dtype=float)
    Phi0 = np.zeros((K, R_true), dtype=float)

    # place factor centers in the low-k regime
    max_center = min(max(6, 3 * R_true), max(8, K // 3))
    centers = np.linspace(2.0, float(max_center), R_true)
    widths = np.linspace(2.0, 4.5, R_true)

    decay_profile = np.exp(-phi_decay * (kk - 1.0))  # low-k emphasis

    for r in range(R_true):
        bump = np.exp(-0.5 * ((kk - centers[r]) / widths[r]) ** 2)
        noise = rng.normal(size=K) * decay_profile
        col = bump + 0.20 * noise
        Phi0[:, r] = col

    # orthonormalize; this may introduce sign changes but preserves low-k concentration
    Phi_true = orthonormal_columns(Phi0)
    return Phi_true


def make_base_spectral_templates(omega: np.ndarray, R_true: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Base nonnegative spectral templates for each factor.
    Returns:
      H_base: (F, R_true)
      centers: (R_true,)
      widths:  (R_true,)
    """
    centers = np.linspace(4.0, 28.0, R_true)
    widths = np.linspace(3.0, 6.0, R_true)

    F = omega.size
    H = np.zeros((F, R_true), dtype=float)
    for r in range(R_true):
        g = np.exp(-0.5 * ((omega - centers[r]) / widths[r]) ** 2)
        bg = 0.18 * np.exp(-omega / 25.0)
        H[:, r] = g + bg + 0.05

    H /= np.maximum(H.max(axis=0, keepdims=True), 1e-12)
    return H, centers, widths


def make_subject_specific_lambda(
    omega: np.ndarray,
    n: int,
    R_true: int,
    subj_amp: np.ndarray,
    rng: np.random.Generator,
    shift_sd: float = 1.5,
    width_log_sd: float = 0.15,
    smooth_perturb_scale: float = 0.18,
) -> np.ndarray:
    """
    Construct subject-specific smooth nonnegative spectral weights lambda_eeg_true (n,F,R).

    Subject variation enters through:
      - factor-specific center shifts
      - factor-specific width jitter
      - smooth multiplicative perturbations over frequency
      - subject-specific amplitudes
    """
    F = omega.size
    H_base, centers, widths = make_base_spectral_templates(omega, R_true)
    lam_all = np.zeros((n, F, R_true), dtype=float)

    # frequency-standardized coordinate for smooth perturbations
    x = (omega - omega.mean()) / max(omega.std(), 1e-8)
    bump_mid = np.exp(-0.5 * ((omega - 12.0) / 8.0) ** 2)
    bump_hi = np.exp(-0.5 * ((omega - 28.0) / 10.0) ** 2)

    for i in range(n):
        for r in range(R_true):
            c = centers[r] + shift_sd * rng.normal()
            w = widths[r] * np.exp(width_log_sd * rng.normal())

            g = np.exp(-0.5 * ((omega - c) / w) ** 2)
            bg = 0.18 * np.exp(-omega / 25.0)
            base = g + bg + 0.05

            # smooth subject-specific perturbation
            a1 = rng.normal()
            a2 = rng.normal()
            a3 = rng.normal()
            perturb = a1 * x + a2 * bump_mid + a3 * bump_hi
            multiplier = np.exp(smooth_perturb_scale * perturb)

            lam = subj_amp[i, r] * base * multiplier
            lam = np.clip(lam, 1e-6, None)

            # normalize per-factor template roughly to preserve scale
            lam /= np.maximum(np.max(lam), 1e-12)
            lam *= subj_amp[i, r]

            lam_all[i, :, r] = lam

    return lam_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output NPZ path")
    ap.add_argument("--n", type=int, default=40, help="Number of subjects")
    ap.add_argument("--K", type=int, default=30, help="LB dimension")
    ap.add_argument("--F", type=int, default=20, help="Number of EEG frequency bins")
    ap.add_argument("--R_true", type=int, default=4, help="True latent rank")
    ap.add_argument("--snr", type=float, default=4.0, help="Signal-to-noise ratio")
    ap.add_argument("--seed", type=int, default=0, help="Random seed")

    ap.add_argument("--k_eeg_max", type=int, default=12,
                    help="Maximum LB mode index reliably observed by EEG (defines W_eeg)")
    ap.add_argument("--omega_min", type=float, default=1.0)
    ap.add_argument("--omega_max", type=float, default=45.0)
    ap.add_argument("--fmri_cut_hz", type=float, default=8.0,
                    help="Low-frequency cutoff (Hz) defining m_fmri for aggregate summaries")

    #  Separate mode is the practical default.....
    ap.add_argument("--generate_fmri_mode", choices=["aggregate", "separate"], default="separate",
                    help="How to generate latent fMRI strengths (default: separate)")

    # Option A: low-k-concentrated factors..
    ap.add_argument("--phi_decay", type=float, default=0.12,
                    help="Exponential decay controlling low-k concentration of true factors")

    # Option B:  subject-specific EEG spectral heterogeneity..
    ap.add_argument("--spectral_shift_sd", type=float, default=1.5,
                    help="SD of subject-specific factor center shifts (Hz)")
    ap.add_argument("--spectral_width_log_sd", type=float, default=0.15,
                    help="Log-SD of subject-specific factor width jitter")
    ap.add_argument("--spectral_perturb_scale", type=float, default=0.18,
                    help="Scale of smooth subject-specific multiplicative perturbation over frequency")

    args = ap.parse_args()

    out = Path(args.out).expanduser().resolve()
    ensure_parent(out)

    rng = np.random.default_rng(args.seed)

    n = int(args.n)
    K = int(args.K)
    F = int(args.F)
    R_true = int(args.R_true)
    k_eeg_max = int(args.k_eeg_max)
    snr = float(args.snr)

    if R_true > K:
        raise ValueError(f"R_true={R_true} must be <= K={K}")
    if not (1 <= k_eeg_max <= K):
        raise ValueError(f"k_eeg_max={k_eeg_max} must be in [1, K={K}]")

    omega = np.linspace(float(args.omega_min), float(args.omega_max), F)

    # (A) low-k concentrated true factors
    Phi_true = make_lowk_factors(K, R_true, rng=rng, phi_decay=float(args.phi_decay))

    # subject-specific amplitudes
    subj_amp = rng.lognormal(mean=0.0, sigma=0.35, size=(n, R_true))

    # (B) subject-specific smooth spectral weights
    lambda_eeg_true = make_subject_specific_lambda(
        omega=omega,
        n=n,
        R_true=R_true,
        subj_amp=subj_amp,
        rng=rng,
        shift_sd=float(args.spectral_shift_sd),
        width_log_sd=float(args.spectral_width_log_sd),
        smooth_perturb_scale=float(args.spectral_perturb_scale),
    )

    # low-frequency bins used by aggregate summary / bookkeeping
    m_fmri = np.where(omega <= float(args.fmri_cut_hz))[0]
    if m_fmri.size == 0:
        m_fmri = np.array([0], dtype=int)

    w_m = np.zeros(F, dtype=float)
    w_m[m_fmri] = 1.0

    # subject-specific fMRI strengths
    ww = w_m[m_fmri] / np.sum(w_m[m_fmri])
    lam_agg = np.sum(lambda_eeg_true[:, m_fmri, :] * ww[None, :, None], axis=1)

    if args.generate_fmri_mode == "aggregate":
        lambda_fmri_true = lam_agg.copy()
    else:
        # separate mode: correlated with low-frequency EEG aggregate, but not identical
        extra = rng.lognormal(mean=-0.15, sigma=0.25, size=(n, R_true))
        lambda_fmri_true = 0.75 * lam_agg + 0.25 * extra
        lambda_fmri_true = np.clip(lambda_fmri_true, 1e-6, None)

    # masks
    W_fmri = np.ones((K, K), dtype=float)
    eeg_obs = (np.arange(K) < k_eeg_max).astype(float)
    W_eeg = np.outer(eeg_obs, eeg_obs)

    # clean signal covariances
    C_eeg_clean = np.zeros((n, F, K, K), dtype=float)
    C_fmri_clean = np.zeros((n, K, K), dtype=float)
    for i in range(n):
        for f in range(F):
            C_eeg_clean[i, f] = sigma_from_phi_lambda(Phi_true, lambda_eeg_true[i, f])
        C_fmri_clean[i] = sigma_from_phi_lambda(Phi_true, lambda_fmri_true[i])

    # add symmetric noise
    C_eeg = np.zeros_like(C_eeg_clean)
    C_fmri = np.zeros_like(C_fmri_clean)
    for i in range(n):
        for f in range(F):
            C_eeg[i, f] = add_scaled_symmetric_noise(C_eeg_clean[i, f], snr=snr, rng=rng)
        C_fmri[i] = add_scaled_symmetric_noise(C_fmri_clean[i], snr=snr, rng=rng)

    subject_ids = np.array([f"toy_{i:03d}" for i in range(n)], dtype=object)

    np.savez_compressed(
        out,
        C_eeg=C_eeg,
        C_fmri=C_fmri,
        W_eeg=W_eeg,
        W_fmri=W_fmri,
        m_fmri=m_fmri.astype(int),
        w_m=w_m.astype(float),
        sigma_eeg=np.array(1.0, dtype=float),
        sigma_fmri=np.array(1.0, dtype=float),
        include_diag=np.array(1, dtype=int),
        omega=omega.astype(float),
        subject_ids=subject_ids,

        # latent truth, metadata
        Phi_true=Phi_true,
        lambda_eeg_true=lambda_eeg_true,
        lambda_fmri_true=lambda_fmri_true,
        trueR=np.array(R_true, dtype=int),
        KMAX=np.array(k_eeg_max, dtype=int),
        SNR=np.array(snr, dtype=float),
        seed=np.array(int(args.seed), dtype=int),
        generate_fmri_mode=np.array(args.generate_fmri_mode, dtype=object),
        phi_decay=np.array(float(args.phi_decay), dtype=float),
        spectral_shift_sd=np.array(float(args.spectral_shift_sd), dtype=float),
        spectral_width_log_sd=np.array(float(args.spectral_width_log_sd), dtype=float),
        spectral_perturb_scale=np.array(float(args.spectral_perturb_scale), dtype=float),
    )

    print("Wrote:", out)
    print("Shapes:")
    print(" - C_eeg       :", C_eeg.shape)
    print(" - C_fmri      :", C_fmri.shape)
    print(" - W_eeg       :", W_eeg.shape)
    print(" - W_fmri      :", W_fmri.shape)
    print(" - Phi_true    :", Phi_true.shape)
    print(" - lambda_eeg  :", lambda_eeg_true.shape)
    print(" - lambda_fmri :", lambda_fmri_true.shape)
    print("Low-frequency fMRI bins (m_fmri):", m_fmri.tolist())
    print("Generation mode:", args.generate_fmri_mode)
    print("Done.")


if __name__ == "__main__":
    main()