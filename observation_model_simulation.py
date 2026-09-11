#!/usr/bin/env python3
"""
observation_model_simulation.py

Generate simulations for validating shared spatial factors  under distinct EEG and fMRI observation pipelines.

Primary design:
- latent oscillations are generated in factor space and projected through Phi;
- EEG uses windowed raw amplitude-envelope covariance after symmetric
  orthogonalization on the candidate low-mode block; and 
- fMRI uses HRF-convolved slow factor drives and covariance;
- observation noise is added after both summaries are constructed and is
  calibrated to an exact masked connectivity-level SNR.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import scipy.linalg as la
import scipy.signal as sig

try:
    from joblib import Parallel, delayed

    _HAS_JOBLIB = True
except Exception:
    _HAS_JOBLIB = False


Array = np.ndarray


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_str_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def ensure_parent(path: str | Path) -> None:
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def sign_convention(Q: Array) -> Array:
    Q = np.asarray(Q, dtype=float).copy()
    for r in range(Q.shape[1]):
        j = int(np.argmax(np.abs(Q[:, r])))
        if Q[j, r] < 0:
            Q[:, r] *= -1.0
    return Q


def make_phi_true(K: int, R: int, k0: float, rng: np.random.Generator) -> Array:
    """Generate an orthonormal LB dictionary with decaying mode variance."""
    mode_var = np.exp(-np.arange(K, dtype=float) / float(k0))
    raw = rng.normal(size=(K, R)) * np.sqrt(mode_var[:, None])
    Q, _ = la.qr(raw, mode="economic")
    return sign_convention(Q[:, :R])


def make_orthogonal_complement(Phi: Array, rng: np.random.Generator) -> Array:
    K, R = Phi.shape
    raw = rng.normal(size=(K, R))
    raw -= Phi @ (Phi.T @ raw)
    Q, _ = la.qr(raw, mode="economic")
    Q = Q[:, :R]
    Q -= Phi @ (Phi.T @ Q)
    Q, _ = la.qr(Q, mode="economic")
    return sign_convention(Q[:, :R])


def rotate_subspace(Phi: Array, Psi: Array, angle_deg: float) -> Array:
    theta = np.deg2rad(float(angle_deg))
    M = np.cos(theta) * Phi + np.sin(theta) * Psi
    Q, _ = la.qr(M, mode="economic")
    return sign_convention(Q[:, : Phi.shape[1]])


def make_phi_bands(
    Phi: Array,
    F: int,
    scenario: str,
    topology_angle_deg: float,
    rng: np.random.Generator,
) -> Tuple[Array | None, Array]:
    """Create EEG band dictionaries and realized angles to the fMRI/common truth."""
    if scenario in {"direct_covariance", "observation"}:
        return None, np.zeros(F, dtype=float)

    Psi = make_orthogonal_complement(Phi, rng)
    if scenario == "freq_rotation":
        nominal_angles = np.linspace(0.0, float(topology_angle_deg), F)
    elif scenario == "crossmodal_divergence":
        nominal_angles = np.full(F, float(topology_angle_deg), dtype=float)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    Phi_bands = np.stack([rotate_subspace(Phi, Psi, a) for a in nominal_angles], axis=0)
    realized = []
    for Pb in Phi_bands:
        s = np.clip(la.svdvals(Phi.T @ Pb), 0.0, 1.0)
        realized.append(float(np.mean(np.degrees(np.arccos(s)))))
    return Phi_bands, np.asarray(realized, dtype=float)


def make_band_profiles(R: int, centers: Array, bandwidth: float) -> Array:
    if R == 5 and centers.size == 4:
        preferred = np.asarray([6.0, 10.0, 18.0, 28.0, 38.0])
    else:
        preferred = np.exp(
            np.linspace(np.log(float(np.min(centers))), np.log(float(np.max(centers))), R)
        )
    logc = np.log(centers)[None, :]
    logp = np.log(preferred)[:, None]
    profiles = np.exp(-((logc - logp) ** 2) / (2.0 * float(bandwidth) ** 2))
    return profiles / np.maximum(profiles.sum(axis=1, keepdims=True), 1e-12)


def make_lambdas(
    n: int,
    F: int,
    R: int,
    profiles: Array,
    rng: np.random.Generator,
    lambda0: float,
    sigma_subject: float,
    sigma_band: float,
    rho_band: float,
) -> Array:
    shared = rng.normal(0.0, sigma_subject, size=(n, R))
    idx = np.arange(F)
    cov = sigma_band**2 * rho_band ** np.abs(idx[:, None] - idx[None, :])
    chol = la.cholesky(cov + 1e-12 * np.eye(F), lower=True)

    out = np.zeros((n, F, R), dtype=float)
    for i in range(n):
        for r in range(R):
            band_dev = chol @ rng.normal(size=F)
            out[i, :, r] = lambda0 * profiles[r] * np.exp(shared[i, r] + band_dev)
    return out


def canonical_hrf(fs: float, duration: float = 32.0) -> Array:
    t = np.arange(0.0, duration + 1.0 / fs, 1.0 / fs)
    h1 = t**5 * np.exp(-t) / float(math.gamma(6))
    h2 = t**15 * np.exp(-t) / float(math.gamma(16))
    h = h1 - h2 / 6.0
    return h / max(float(np.max(np.abs(h))), 1e-12)


def make_eeg_reliability(
    K: int,
    kmax: int,
    mask: str,
    taper_s: float,
    candidate_modes: int,
) -> Tuple[Array, Array]:
    """Return mode reliabilities q and the edge reliability mask q q^T."""
    if not (1 <= kmax <= candidate_modes <= K):
        raise ValueError("Require 1 <= kmax <= candidate_modes <= K")
    k = np.arange(1, K + 1, dtype=float)
    if mask == "hard":
        q = (k <= float(kmax)).astype(float)
    elif mask == "smooth":
        q = 1.0 / (1.0 + np.exp((k - float(kmax)) / float(taper_s)))
        q[k > candidate_modes] = 0.0
    else:
        raise ValueError("mask must be 'hard' or 'smooth'")
    return q, np.outer(q, q)


def center_time(X: Array) -> Array:
    return X - X.mean(axis=1, keepdims=True)


def zscore_time(X: Array, eps: float = 1e-8) -> Array:
    Xc = center_time(X)
    return Xc / np.maximum(Xc.std(axis=1, keepdims=True), eps)


def cov_time(X: Array) -> Array:
    return (X @ X.T) / max(X.shape[1] - 1, 1)


def ar1_process(n_series: int, T: int, rho: float, rng: np.random.Generator) -> Array:
    innovations = rng.normal(size=(n_series, T))
    x = np.empty_like(innovations)
    x[:, 0] = innovations[:, 0]
    scale = np.sqrt(max(0.0, 1.0 - rho**2))
    for t in range(1, T):
        x[:, t] = rho * x[:, t - 1] + scale * innovations[:, t]
    return zscore_time(x)


def butter_bandpass_sos(low: float, high: float, fs: float, order: int = 4) -> Array:
    nyq = fs / 2.0
    low_n = max(0.001, float(low)) / nyq
    high_n = min(float(high), 0.98 * nyq) / nyq
    if not (0 < low_n < high_n < 1):
        raise ValueError(f"Invalid bandpass [{low}, {high}] for fs={fs}")
    return sig.butter(order, [low_n, high_n], btype="bandpass", output="sos")


def sym_orthogonalize(X: Array, ridge: float) -> Array:
    C = (X @ X.T) / max(X.shape[1] - 1, 1)
    vals, vecs = la.eigh(C + float(ridge) * np.eye(X.shape[0]))
    vals = np.maximum(vals, float(ridge))
    invsqrt = (vecs / np.sqrt(vals)[None, :]) @ vecs.T
    return invsqrt @ X


def windowed_raw_aecov(
    X_band: Array,
    K: int,
    candidate_modes: int,
    fs: float,
    window_sec: float,
    orth_ridge: float,
) -> Array:
    """Windowed raw amplitude-envelope covariance on the candidate low-mode block."""
    win = int(round(float(window_sec) * float(fs)))
    if win < 8 or X_band.shape[1] < win:
        raise ValueError("EEG window is too short or exceeds the simulated duration")
    nwin = X_band.shape[1] // win
    mats = []
    for w in range(nwin):
        Xw = X_band[:candidate_modes, w * win : (w + 1) * win]
        Xo = sym_orthogonalize(Xw, orth_ridge)
        amp = np.abs(sig.hilbert(Xo, axis=1))
        Csmall = cov_time(center_time(amp))
        C = np.zeros((K, K), dtype=float)
        C[:candidate_modes, :candidate_modes] = Csmall
        mats.append(C)
    return np.mean(mats, axis=0)


def masked_norm_sq(X: Array, W: Array) -> float:
    return float(np.sum((X * W) ** 2))


def add_masked_symmetric_noise(
    C_clean: Array,
    W: Array,
    target_snr: float,
    rng: np.random.Generator,
) -> Tuple[Array, float, float]:
    """Add symmetric Gaussian noise at an exact masked connectivity-level SNR."""
    if target_snr <= 0:
        raise ValueError("target_snr must be positive")
    noise = rng.normal(size=C_clean.shape)
    noise = 0.5 * (noise + np.swapaxes(noise, -1, -2))
    signal_energy = masked_norm_sq(C_clean, W)
    noise_energy = masked_norm_sq(noise, W)
    if signal_energy <= 0 or noise_energy <= 0:
        raise ValueError("Signal or noise has zero masked energy")
    scale = np.sqrt(signal_energy / (float(target_snr) * noise_energy))
    C_noisy = C_clean + scale * noise
    C_noisy = 0.5 * (C_noisy + np.swapaxes(C_noisy, -1, -2))
    realized = signal_energy / masked_norm_sq(C_noisy - C_clean, W)
    return C_noisy, float(scale), float(realized)


def masked_frob_norms(C: Array, W: Array) -> Array:
    lead = int(np.prod(C.shape[:-2]))
    X = (C * W).reshape(lead, -1)
    return np.sqrt(np.sum(X * X, axis=1))


def robust_modality_rescale(
    C_eeg: Array,
    C_eeg_clean: Array,
    C_fmri: Array,
    C_fmri_clean: Array,
    W_eeg: Array,
    W_fmri: Array,
) -> Tuple[Array, Array, Array, Array, float, float]:
    n, F = C_eeg.shape[:2]
    eeg_norm = masked_frob_norms(C_eeg, W_eeg).reshape(n, F).mean(axis=1)
    fmri_norm = masked_frob_norms(C_fmri, W_fmri)
    s_eeg = max(float(np.median(eeg_norm)), 1e-12)
    s_fmri = max(float(np.median(fmri_norm)), 1e-12)
    return (
        C_eeg / s_eeg,
        C_eeg_clean / s_eeg,
        C_fmri / s_fmri,
        C_fmri_clean / s_fmri,
        s_eeg,
        s_fmri,
    )


def direct_covariance_clean(Phi: Array, lambda_eeg: Array, band_weights: Array) -> Dict[str, Array]:
    n, F, _ = lambda_eeg.shape
    C_eeg = np.stack(
        [
            np.stack(
                [(Phi * lambda_eeg[i, f][None, :]) @ Phi.T for f in range(F)],
                axis=0,
            )
            for i in range(n)
        ],
        axis=0,
    )
    lambda_fmri = np.einsum("f,nfr->nr", band_weights, lambda_eeg)
    C_fmri = np.stack([(Phi * lambda_fmri[i][None, :]) @ Phi.T for i in range(n)], axis=0)
    return {"C_eeg_clean": C_eeg, "C_fmri_clean": C_fmri, "lambda_fmri_true": lambda_fmri}


def simulate_clean_subject(
    i: int,
    seed: int,
    Phi_fmri: Array,
    Phi_bands: Array | None,
    lambda_i: Array,
    band_intervals: Sequence[Tuple[float, float]],
    band_centers: Array,
    band_weights: Array,
    fs: float,
    duration: float,
    tr: float,
    tau_env: float,
    env_log_scale: float,
    hrf: Array,
    eeg_candidate_modes: int,
    eeg_window_sec: float,
    orth_ridge: float,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    K, R = Phi_fmri.shape
    F = len(band_intervals)
    T = int(round(float(duration) * float(fs)))
    times = np.arange(T, dtype=float) / float(fs)
    down = max(1, int(round(float(tr) * float(fs))))
    V = T // down
    rho_env = float(np.exp(-(1.0 / float(fs)) / float(tau_env)))
    filters = [butter_bandpass_sos(lo, hi, fs) for lo, hi in band_intervals]

    m = ar1_process(R * F, T, rho_env, rng).reshape(R, F, T)
    sqrt_lam = np.sqrt(np.maximum(lambda_i, 0.0))

    # fMRI: slow shared factor drives projected through the fMRI/common dictionary.
    drive = np.zeros((R, T), dtype=float)
    for f in range(F):
        drive += band_weights[f] * sqrt_lam[f, :, None] * m[:, f, :]
    lambda_fmri_true = np.sum((band_weights[:, None] ** 2) * lambda_i, axis=0)
    neural_fmri = Phi_fmri @ drive
    bold_full = sig.fftconvolve(neural_fmri, hrf[None, :], mode="full", axes=1)[:, :T]
    bold = center_time(bold_full[:, : V * down : down] / float(fs))
    C_fmri_clean = cov_time(bold)

    # EEG: amplitude-modulated oscillations are generated in factor space, then
    # projected through the band-specific spatial dictionaries.
    eeg = np.zeros((K, T), dtype=float)
    for f, center in enumerate(band_centers):
        Phi_eeg = Phi_fmri if Phi_bands is None else Phi_bands[f]
        phases = rng.uniform(0.0, 2.0 * np.pi, size=(R, 1))
        amplitude = sqrt_lam[f, :, None] * np.exp(float(env_log_scale) * m[:, f, :])
        factors = amplitude * np.cos(2.0 * np.pi * float(center) * times[None, :] + phases)
        eeg += Phi_eeg @ factors

    C_eeg_clean = np.zeros((F, K, K), dtype=float)
    for f, sos in enumerate(filters):
        band_signal = sig.sosfiltfilt(sos, eeg, axis=1)
        C_eeg_clean[f] = windowed_raw_aecov(
            band_signal,
            K=K,
            candidate_modes=eeg_candidate_modes,
            fs=fs,
            window_sec=eeg_window_sec,
            orth_ridge=orth_ridge,
        )

    return {
        "i": i,
        "C_eeg_clean": C_eeg_clean,
        "C_fmri_clean": C_fmri_clean,
        "lambda_fmri_true": lambda_fmri_true,
    }


def observation_clean(
    Phi: Array,
    Phi_bands: Array | None,
    lambda_eeg: Array,
    band_intervals: Sequence[Tuple[float, float]],
    band_centers: Array,
    band_weights: Array,
    fs: float,
    duration: float,
    tr: float,
    tau_env: float,
    env_log_scale: float,
    eeg_candidate_modes: int,
    eeg_window_sec: float,
    orth_ridge: float,
    subject_seeds: Sequence[int],
    n_jobs: int,
) -> Dict[str, Array]:
    hrf = canonical_hrf(fs)
    jobs = [
        (
            i,
            int(subject_seeds[i]),
            Phi,
            Phi_bands,
            lambda_eeg[i],
            band_intervals,
            band_centers,
            band_weights,
            fs,
            duration,
            tr,
            tau_env,
            env_log_scale,
            hrf,
            eeg_candidate_modes,
            eeg_window_sec,
            orth_ridge,
        )
        for i in range(lambda_eeg.shape[0])
    ]
    if n_jobs > 1 and _HAS_JOBLIB:
        rows = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(simulate_clean_subject)(*job) for job in jobs
        )
    else:
        rows = [simulate_clean_subject(*job) for job in jobs]
    rows.sort(key=lambda x: x["i"])
    return {
        "C_eeg_clean": np.stack([x["C_eeg_clean"] for x in rows], axis=0),
        "C_fmri_clean": np.stack([x["C_fmri_clean"] for x in rows], axis=0),
        "lambda_fmri_true": np.stack([x["lambda_fmri_true"] for x in rows], axis=0),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate redesigned observation-model validation simulation data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenario_label", default="")
    ap.add_argument(
        "--scenario",
        choices=["direct_covariance", "observation", "freq_rotation", "crossmodal_divergence"],
        default="observation",
    )
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--K", type=int, default=50)
    ap.add_argument("--R_true", type=int, default=5)
    ap.add_argument("--duration", type=float, default=300.0)
    ap.add_argument("--fs", type=float, default=200.0)
    ap.add_argument("--tr", type=float, default=1.4)
    ap.add_argument("--summary_snr", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=1001)
    ap.add_argument("--n_jobs", type=int, default=4)

    ap.add_argument("--k0_phi", type=float, default=12.0)
    ap.add_argument("--kmax_eeg", type=int, default=20)
    ap.add_argument("--mask", choices=["hard", "smooth"], default="hard")
    ap.add_argument("--taper_s", type=float, default=3.0)
    ap.add_argument("--eeg_candidate_modes", type=int, default=20)
    ap.add_argument("--topology_angle_deg", type=float, default=0.0)

    ap.add_argument("--tau_env", type=float, default=5.0)
    ap.add_argument("--env_log_scale", type=float, default=0.35)
    ap.add_argument("--eeg_window_sec", type=float, default=10.0)
    ap.add_argument("--orth_ridge", type=float, default=1e-6)

    ap.add_argument("--h_f", type=float, default=0.45)
    ap.add_argument("--lambda0", type=float, default=1.0)
    ap.add_argument("--sigma_subject", type=float, default=0.35)
    ap.add_argument("--sigma_band", type=float, default=0.15)
    ap.add_argument("--rho_band", type=float, default=0.7)

    ap.add_argument("--band_names", default="theta,alpha,beta,gamma")
    ap.add_argument("--band_lows", default="4,8,13,30")
    ap.add_argument("--band_highs", default="7,12,30,45")
    ap.add_argument("--band_centers", default="6,10,20,38")
    ap.add_argument("--band_weights", default="equal")
    ap.add_argument("--modality_scaling", choices=["none", "robust"], default="robust")
    args = ap.parse_args()

    if args.n <= 0 or args.K <= 0 or args.R_true <= 0:
        raise ValueError("n, K, and R_true must be positive")
    if args.R_true > args.K:
        raise ValueError("R_true cannot exceed K")

    names = parse_str_list(args.band_names)
    lows = np.asarray(parse_float_list(args.band_lows), dtype=float)
    highs = np.asarray(parse_float_list(args.band_highs), dtype=float)
    centers = np.asarray(parse_float_list(args.band_centers), dtype=float)
    if not (len(names) == lows.size == highs.size == centers.size):
        raise ValueError("Band definitions must have equal length")
    intervals = list(zip(lows.tolist(), highs.tolist()))
    F = centers.size

    if args.band_weights == "equal":
        band_weights = np.full(F, 1.0 / F, dtype=float)
    else:
        band_weights = np.asarray(parse_float_list(args.band_weights), dtype=float)
        if band_weights.size != F:
            raise ValueError("band_weights must have one value per band")
        band_weights /= max(float(np.sum(band_weights)), 1e-12)

    seed_seq = np.random.SeedSequence(args.seed)
    ss_phi, ss_lam, ss_topology, ss_subjects, ss_eeg_noise, ss_fmri_noise = seed_seq.spawn(6)
    rng_phi = np.random.default_rng(ss_phi)
    rng_lam = np.random.default_rng(ss_lam)
    rng_topology = np.random.default_rng(ss_topology)
    rng_eeg_noise = np.random.default_rng(ss_eeg_noise)
    rng_fmri_noise = np.random.default_rng(ss_fmri_noise)

    Phi_true = make_phi_true(args.K, args.R_true, args.k0_phi, rng_phi)
    Phi_bands, realized_band_angles = make_phi_bands(
        Phi_true,
        F,
        args.scenario,
        args.topology_angle_deg,
        rng_topology,
    )
    profiles = make_band_profiles(args.R_true, centers, args.h_f)
    lambda_eeg_true = make_lambdas(
        args.n,
        F,
        args.R_true,
        profiles,
        rng_lam,
        lambda0=args.lambda0,
        sigma_subject=args.sigma_subject,
        sigma_band=args.sigma_band,
        rho_band=args.rho_band,
    )

    q_eeg, W_eeg = make_eeg_reliability(
        args.K,
        args.kmax_eeg,
        args.mask,
        args.taper_s,
        args.eeg_candidate_modes,
    )
    W_fmri = np.ones((args.K, args.K), dtype=float)

    if args.scenario == "direct_covariance":
        clean = direct_covariance_clean(Phi_true, lambda_eeg_true, band_weights)
    else:
        subject_seeds = [int(x.generate_state(1)[0]) for x in ss_subjects.spawn(args.n)]
        clean = observation_clean(
            Phi_true,
            Phi_bands,
            lambda_eeg_true,
            intervals,
            centers,
            band_weights,
            args.fs,
            args.duration,
            args.tr,
            args.tau_env,
            args.env_log_scale,
            args.eeg_candidate_modes,
            args.eeg_window_sec,
            args.orth_ridge,
            subject_seeds,
            args.n_jobs,
        )

    C_eeg_clean = np.asarray(clean["C_eeg_clean"], dtype=float)
    C_fmri_clean = np.asarray(clean["C_fmri_clean"], dtype=float)
    C_eeg, eeg_noise_scale, eeg_snr_realized = add_masked_symmetric_noise(
        C_eeg_clean, W_eeg, args.summary_snr, rng_eeg_noise
    )
    C_fmri, fmri_noise_scale, fmri_snr_realized = add_masked_symmetric_noise(
        C_fmri_clean, W_fmri, args.summary_snr, rng_fmri_noise
    )

    scale_eeg = scale_fmri = 1.0
    if args.modality_scaling == "robust":
        C_eeg, C_eeg_clean, C_fmri, C_fmri_clean, scale_eeg, scale_fmri = robust_modality_rescale(
            C_eeg,
            C_eeg_clean,
            C_fmri,
            C_fmri_clean,
            W_eeg,
            W_fmri,
        )

    lowk_energy = float(np.sum(Phi_true[: args.kmax_eeg] ** 2) / np.sum(Phi_true**2))
    metadata: Dict[str, Any] = {
        "scenario_label": args.scenario_label or args.scenario,
        "scenario": args.scenario,
        "n": args.n,
        "K": args.K,
        "R_true": args.R_true,
        "F": int(F),
        "seed": args.seed,
        "duration": args.duration,
        "fs": args.fs,
        "tr": args.tr,
        "summary_snr_target": args.summary_snr,
        "summary_snr_eeg_realized": eeg_snr_realized,
        "summary_snr_fmri_realized": fmri_snr_realized,
        "k0_phi": args.k0_phi,
        "kmax_eeg": args.kmax_eeg,
        "eeg_candidate_modes": args.eeg_candidate_modes,
        "truth_lowk_energy_fraction": lowk_energy,
        "mask": args.mask,
        "taper_s": args.taper_s,
        "topology_angle_deg": args.topology_angle_deg,
        "realized_band_angles_to_common_deg": realized_band_angles.tolist(),
        "eeg_summary": "rawenv_cov",
        "eeg_window_sec": args.eeg_window_sec,
        "orthogonalization": "symmetric_within_window_candidate_block",
        "orth_ridge": args.orth_ridge,
        "noise_stage": "post_connectivity_summary",
        "modality_scaling": args.modality_scaling,
        "modality_scale_eeg": scale_eeg,
        "modality_scale_fmri": scale_fmri,
        "band_names": names,
        "band_lows": lows.tolist(),
        "band_highs": highs.tolist(),
        "band_centers": centers.tolist(),
        "band_weights": band_weights.tolist(),
        "tau_env": args.tau_env,
        "env_log_scale": args.env_log_scale,
        "eeg_noise_scale": eeg_noise_scale,
        "fmri_noise_scale": fmri_noise_scale,
    }

    save: Dict[str, Any] = {
        "C_eeg": C_eeg.astype(np.float32),
        "C_fmri": C_fmri.astype(np.float32),
        "C_eeg_clean": C_eeg_clean.astype(np.float32),
        "C_fmri_clean": C_fmri_clean.astype(np.float32),
        "W_eeg": W_eeg.astype(np.float32),
        "q_eeg": q_eeg.astype(np.float32),
        "W_fmri": W_fmri.astype(np.float32),
        "omega": centers.astype(float),
        "include_diag": np.array(True),
        "m_fmri": np.arange(F, dtype=np.int32),
        "w_m": band_weights.astype(float),
        "sigma_eeg": np.array(1.0),
        "sigma_fmri": np.array(1.0),
        "subject_ids": np.asarray([f"sim-{i + 1:04d}" for i in range(args.n)]),
        "Phi_true": Phi_true.astype(float),
        "lambda_eeg_true": lambda_eeg_true.astype(float),
        "lambda_fmri_true": np.asarray(clean["lambda_fmri_true"], dtype=float),
        "scenario": np.array(args.scenario),
        "scenario_label": np.array(metadata["scenario_label"]),
        "metadata_json": np.array(json.dumps(metadata, indent=2)),
    }
    if Phi_bands is not None:
        save["Phi_band_true"] = Phi_bands.astype(float)
        save["band_angles_to_common_deg"] = realized_band_angles.astype(float)

    ensure_parent(args.out)
    np.savez_compressed(args.out, **save)
    print(f"Wrote {args.out}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
