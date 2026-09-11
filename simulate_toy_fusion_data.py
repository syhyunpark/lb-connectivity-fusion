#!/usr/bin/env python3
"""
simulate_toy_fusion_data.py

Generate a small model-concordant synthetic dataset for the revised
shared-spatial-factor EEG--fMRI fusion model.

Scientific purpose
------------------
This is a pedagogical example, not the manuscript's observation-model
validation simulation. EEG and fMRI connectivity summaries are generated as
distinct modality-specific matrices that share an orthonormal spatial basis
Phi_true but have separate nonnegative subject-level strengths. EEG strengths
are additionally frequency resolved, and only low-order LB covariance entries
are used by the EEG fitting mask.

The output is directly compatible with shared_spatial_model.py using
--fmri_mode separate.

SNR convention
--------------
For each modality, noise is scaled to the requested squared-Frobenius SNR on
the entries used by that modality:

    SNR = ||W * signal||_F^2 / ||W * noise||_F^2.

This matches the manuscript-facing simulation convention.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.linalg as la


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def orthonormal_columns(A: np.ndarray) -> np.ndarray:
    Q, _ = la.qr(A, mode="economic")
    return Q


def sigma_from_phi_lambda(Phi: np.ndarray, lam: np.ndarray) -> np.ndarray:
    return (Phi * lam[None, :]) @ Phi.T


def make_lowk_factors(
    K: int,
    R_true: int,
    rng: np.random.Generator,
    phi_decay: float = 0.12,
) -> np.ndarray:
    """Construct smooth-ish orthonormal factors concentrated toward low LB modes."""
    k = np.arange(1, K + 1, dtype=float)
    Phi0 = np.zeros((K, R_true), dtype=float)

    max_center = min(max(6, 3 * R_true), max(8, K // 3))
    centers = np.linspace(2.0, float(max_center), R_true)
    widths = np.linspace(2.0, 4.5, R_true)
    decay = np.exp(-phi_decay * (k - 1.0))

    for r in range(R_true):
        bump = np.exp(-0.5 * ((k - centers[r]) / widths[r]) ** 2)
        perturb = 0.20 * rng.normal(size=K) * decay
        Phi0[:, r] = bump + perturb

    return orthonormal_columns(Phi0)


def make_eeg_strengths(
    omega: np.ndarray,
    shared_amp: np.ndarray,
    rng: np.random.Generator,
    shift_sd: float = 1.5,
    width_log_sd: float = 0.15,
    smooth_perturb_scale: float = 0.18,
) -> np.ndarray:
    """Generate smooth subject- and frequency-specific nonnegative EEG strengths."""
    n, R_true = shared_amp.shape
    F = omega.size
    centers = np.linspace(4.0, 28.0, R_true)
    widths = np.linspace(3.0, 6.0, R_true)

    x = (omega - omega.mean()) / max(float(omega.std()), 1e-8)
    bump_mid = np.exp(-0.5 * ((omega - 12.0) / 8.0) ** 2)
    bump_hi = np.exp(-0.5 * ((omega - 28.0) / 10.0) ** 2)

    lam = np.zeros((n, F, R_true), dtype=float)
    for i in range(n):
        for r in range(R_true):
            center = centers[r] + shift_sd * rng.normal()
            width = widths[r] * np.exp(width_log_sd * rng.normal())

            profile = (
                np.exp(-0.5 * ((omega - center) / width) ** 2)
                + 0.18 * np.exp(-omega / 25.0)
                + 0.05
            )
            smooth_noise = (
                rng.normal() * x
                + rng.normal() * bump_mid
                + rng.normal() * bump_hi
            )
            profile *= np.exp(smooth_perturb_scale * smooth_noise)
            profile /= max(float(profile.max()), 1e-12)
            lam[i, :, r] = shared_amp[i, r] * np.clip(profile, 1e-6, None)

    return lam


def make_fmri_strengths(
    shared_amp: np.ndarray,
    rng: np.random.Generator,
    jitter_sd: float = 0.25,
) -> np.ndarray:
    """
    Generate modality-specific fMRI strengths.

    fMRI and EEG share subject-by-factor amplitude tendencies through shared_amp,
    but fMRI strengths are not defined as EEG frequency averages and are not
    assumed to equal EEG strengths.
    """
    jitter = np.exp(jitter_sd * rng.normal(size=shared_amp.shape))
    return np.clip(shared_amp * jitter, 1e-6, None)


def add_masked_symmetric_noise(
    signal: np.ndarray,
    W: np.ndarray,
    snr: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add symmetric Gaussian noise at a target squared-Frobenius masked SNR."""
    noise = rng.normal(size=signal.shape)
    noise = 0.5 * (noise + noise.T)

    signal_norm = float(np.linalg.norm((W * signal).ravel()))
    noise_norm = float(np.linalg.norm((W * noise).ravel()))
    if signal_norm <= 1e-12 or noise_norm <= 1e-12 or snr <= 0:
        return signal.copy()

    scale = signal_norm / (np.sqrt(float(snr)) * noise_norm)
    out = signal + scale * noise
    return 0.5 * (out + out.T)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a toy EEG--fMRI fusion dataset.")
    ap.add_argument("--out", default="toy_fusion_data.npz")
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--K", type=int, default=50)
    ap.add_argument("--F", type=int, default=20)
    ap.add_argument("--R_true", type=int, default=5)
    ap.add_argument(
        "--snr",
        type=float,
        default=4.0,
        help="Squared-Frobenius signal/noise ratio on each modality's fitting support.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k_eeg_max", type=int, default=20)
    ap.add_argument("--omega_min", type=float, default=1.0)
    ap.add_argument("--omega_max", type=float, default=45.0)
    ap.add_argument("--phi_decay", type=float, default=0.12)
    ap.add_argument("--spectral_shift_sd", type=float, default=1.5)
    ap.add_argument("--spectral_width_log_sd", type=float, default=0.15)
    ap.add_argument("--spectral_perturb_scale", type=float, default=0.18)
    ap.add_argument("--fmri_jitter_sd", type=float, default=0.25)
    args = ap.parse_args()

    out = Path(args.out).expanduser().resolve()
    ensure_parent(out)

    if args.R_true > args.K:
        raise ValueError("R_true must be <= K")
    if not (1 <= args.k_eeg_max <= args.K):
        raise ValueError("k_eeg_max must lie in [1, K]")
    if args.F < 3:
        raise ValueError("F must be >= 3")

    rng = np.random.default_rng(args.seed)
    omega = np.linspace(args.omega_min, args.omega_max, args.F)

    Phi_true = make_lowk_factors(
        args.K, args.R_true, rng, phi_decay=args.phi_decay
    )

    # Shared subject-by-factor tendencies induce cross-modal correspondence while
    # preserving distinct modality-specific strengths.
    shared_amp = rng.lognormal(mean=0.0, sigma=0.35, size=(args.n, args.R_true))
    lambda_eeg_true = make_eeg_strengths(
        omega,
        shared_amp,
        rng,
        shift_sd=args.spectral_shift_sd,
        width_log_sd=args.spectral_width_log_sd,
        smooth_perturb_scale=args.spectral_perturb_scale,
    )
    lambda_fmri_true = make_fmri_strengths(
        shared_amp, rng, jitter_sd=args.fmri_jitter_sd
    )

    W_fmri = np.ones((args.K, args.K), dtype=float)
    q_eeg = (np.arange(args.K) < args.k_eeg_max).astype(float)
    W_eeg = np.outer(q_eeg, q_eeg)

    C_eeg_clean = np.zeros((args.n, args.F, args.K, args.K), dtype=float)
    C_fmri_clean = np.zeros((args.n, args.K, args.K), dtype=float)
    C_eeg = np.zeros_like(C_eeg_clean)
    C_fmri = np.zeros_like(C_fmri_clean)

    for i in range(args.n):
        for f in range(args.F):
            S = sigma_from_phi_lambda(Phi_true, lambda_eeg_true[i, f])
            C_eeg_clean[i, f] = S
            C_eeg[i, f] = add_masked_symmetric_noise(
                S, W_eeg, args.snr, rng
            )

        S_f = sigma_from_phi_lambda(Phi_true, lambda_fmri_true[i])
        C_fmri_clean[i] = S_f
        C_fmri[i] = add_masked_symmetric_noise(
            S_f, W_fmri, args.snr, rng
        )

    # shared_spatial_model.py reads these fields even in separate-fMRI mode.
    # They are compatibility metadata only in this toy example.
    m_fmri = np.array([0], dtype=int)
    w_m = np.zeros(args.F, dtype=float)
    w_m[0] = 1.0

    subject_ids = np.asarray(
        [f"toy_{i:03d}" for i in range(args.n)],
        dtype=f"<U{max(7, len(str(args.n)) + 4)}",
    )

    metadata = {
        "description": "Model-concordant pedagogical toy example",
        "n": args.n,
        "K": args.K,
        "F": args.F,
        "R_true": args.R_true,
        "summary_snr": args.snr,
        "k_eeg_max": args.k_eeg_max,
        "seed": args.seed,
        "fmri_strengths": "separate modality-specific strengths sharing subject-factor tendencies with EEG",
    }

    np.savez_compressed(
        out,
        C_eeg=C_eeg.astype(np.float32),
        C_fmri=C_fmri.astype(np.float32),
        C_eeg_clean=C_eeg_clean.astype(np.float32),
        C_fmri_clean=C_fmri_clean.astype(np.float32),
        W_eeg=W_eeg.astype(np.float32),
        W_fmri=W_fmri.astype(np.float32),
        m_fmri=m_fmri,
        w_m=w_m,
        sigma_eeg=np.array(1.0),
        sigma_fmri=np.array(1.0),
        include_diag=np.array(1, dtype=np.int32),
        omega=omega,
        subject_ids=subject_ids,
        Phi_true=Phi_true,
        lambda_eeg_true=lambda_eeg_true,
        lambda_fmri_true=lambda_fmri_true,
        trueR=np.array(args.R_true, dtype=np.int32),
        KMAX=np.array(args.k_eeg_max, dtype=np.int32),
        SNR=np.array(args.snr),
        seed=np.array(args.seed, dtype=np.int32),
        metadata_json=np.array(json.dumps(metadata)),
    )

    print(f"Wrote {out}")
    print("C_eeg:", C_eeg.shape)
    print("C_fmri:", C_fmri.shape)
    print("Phi_true:", Phi_true.shape)
    print("lambda_eeg_true:", lambda_eeg_true.shape)
    print("lambda_fmri_true:", lambda_fmri_true.shape)
    print(f"EEG fitting support: first {args.k_eeg_max} of {args.K} LB modes")
    print(f"Masked squared-Frobenius SNR: {args.snr:g}")


if __name__ == "__main__":
    main()
