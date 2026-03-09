#!/usr/bin/env python3
# sim_generate_cov.py
#
# Generate simulated EEG (frequency-resolved) and fMRI (low-frequency aggregate)
# covariance/connectivity matrices under the latent low-rank model:
#   Sigma_i(omega_f) = Phi_true diag(lambda_true[i,f]) Phi_true^T
#
# Generate simulated EEG and fMRI covariance matrices under a low-rank latent model.


import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import scipy.linalg as la


@dataclass
class SimParams:
    n: int = 200
    K: int = 50
    F: int = 20
    omega_min: float = 1.0
    omega_max: float = 40.0

    m_fmri: Tuple[int, ...] = (0, 1, 2, 3)

    L_spline: int = 4
    tau: float = 0.4
    peak_hz: Optional[List[float]] = None
    peak_width: float = 0.8

    common_sigma: bool = True
    include_diag: bool = True
    seed: int = 0


def _make_simple_spline_basis(omega: np.ndarray, L: int) -> np.ndarray:
    """Simple smooth basis on [0, 1]. Returns (F, L)."""
    x = (omega - omega.min()) / (omega.max() - omega.min() + 1e-12)
    basis = [np.ones_like(x)]
    if L >= 2:
        basis.append(x)
    if L >= 3:
        basis.append(x * (1.0 - x))
    if L >= 4:
        basis.append(np.sin(2.0 * np.pi * x))
    for j in range(4, L):
        basis.append(np.sin((j - 2) * np.pi * x))
    B = np.vstack(basis[:L]).T
    B = B / (np.sqrt(np.sum(B**2, axis=0, keepdims=True)) + 1e-12)
    return B


def _make_mu_templates(
    omega: np.ndarray,
    R: int,
    peak_hz: Optional[List[float]],
    peak_width: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Return log-Gaussian templates on log-frequency. Shape (R, F).

    These are not mean-centered.
    """
    logw = np.log(omega)
    if peak_hz is None:
        base = np.array([6.0, 10.0, 20.0, 30.0])
        if R <= len(base):
            peaks = base[:R]
        else:
            extra = rng.uniform(5.0, 35.0, size=R - len(base))
            peaks = np.concatenate([base, extra])
        peak_hz = peaks.tolist()
    else:
        if len(peak_hz) < R:
            raise ValueError("peak_hz length is too short")

    mu = np.zeros((R, omega.size), dtype=float)
    for r in range(R):
        c = np.log(peak_hz[r])
        mu[r, :] = -0.5 * ((logw - c) / peak_width) ** 2
    return mu


def _make_eeg_mask(K: int, kmax_eeg: int, include_diag: bool) -> np.ndarray:
    if not (1 <= kmax_eeg <= K):
        raise ValueError("kmax_eeg out of range")
    idx = np.arange(K)
    ok = idx < kmax_eeg
    W = np.outer(ok.astype(float), ok.astype(float))
    if not include_diag:
        np.fill_diagonal(W, 0.0)
    return W


def _make_full_mask(K: int, include_diag: bool) -> np.ndarray:
    W = np.ones((K, K), dtype=float)
    if not include_diag:
        np.fill_diagonal(W, 0.0)
    return W


def _sym_gaussian_noise(K: int, sigma: float, rng: np.random.Generator, include_diag: bool) -> np.ndarray:
    E = np.zeros((K, K), dtype=float)
    if include_diag:
        iu = np.triu_indices(K, k=0)
        E[iu] = rng.normal(0.0, sigma, size=iu[0].shape[0])
        E = E + E.T - np.diag(np.diag(E))
    else:
        iu = np.triu_indices(K, k=1)
        E[iu] = rng.normal(0.0, sigma, size=iu[0].shape[0])
        E = E + E.T
    return E


def _expected_noise_fro2_unit(K: int, mask: np.ndarray, include_diag: bool) -> float:
    if include_diag:
        iu = np.triu_indices(K, k=0)
        w = mask[iu]
        diag = np.diag(mask)
        off = w[(iu[0] != iu[1])]
        return float(np.sum(diag**2) + 2.0 * np.sum(off**2))
    iu = np.triu_indices(K, k=1)
    w = mask[iu]
    return float(2.0 * np.sum(w**2))


def _sigma_for_target_snr(
    signal_fro2_mean: float,
    K: int,
    mask: np.ndarray,
    snr: float,
    include_diag: bool,
) -> float:
    denom_unit = _expected_noise_fro2_unit(K, mask, include_diag)
    if denom_unit <= 0:
        raise ValueError("Mask has no observable entries")
    sigma2 = signal_fro2_mean / (snr * denom_unit)
    return float(np.sqrt(max(sigma2, 1e-16)))


def generate_simulated_data(
    R: int,
    kmax_eeg: int,
    snr: float,
    *,
    params: SimParams,
    save_full: bool = False,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(params.seed)

    n, K, F = params.n, params.K, params.F
    omega = np.linspace(params.omega_min, params.omega_max, F)

    Z = rng.normal(size=(K, R))
    Phi_true, _ = la.qr(Z, mode="economic")

    B = _make_simple_spline_basis(omega, params.L_spline)
    mu = _make_mu_templates(omega, R, params.peak_hz, params.peak_width, rng)
    beta = rng.normal(0.0, params.tau, size=(n, R, params.L_spline))

    loglam = mu[None, :, :] + beta @ B.T
    lam_true = np.exp(loglam)
    lam_true = np.transpose(lam_true, (0, 2, 1))

    W_eeg = _make_eeg_mask(K, kmax_eeg, params.include_diag)
    W_fmri = _make_full_mask(K, params.include_diag)

    m_fmri = np.array(params.m_fmri, dtype=int)
    if np.any(m_fmri < 0) or np.any(m_fmri >= F):
        raise ValueError("m_fmri indices out of range")
    w_m = np.zeros(F, dtype=float)
    w_m[m_fmri] = 1.0 / len(m_fmri)

    eeg_signal_vals = []
    fmri_signal_vals = []
    for i in range(n):
        Sig_f = np.zeros((K, K), dtype=float)
        for f in range(F):
            Sig = (Phi_true * lam_true[i, f][None, :]) @ Phi_true.T
            eeg_signal_vals.append(np.sum((W_eeg * Sig) ** 2))
            if f in m_fmri:
                Sig_f += w_m[f] * Sig
        fmri_signal_vals.append(np.sum((W_fmri * Sig_f) ** 2))

    eeg_signal_fro2 = float(np.mean(eeg_signal_vals))
    fmri_signal_fro2 = float(np.mean(fmri_signal_vals))

    sigma_eeg = _sigma_for_target_snr(eeg_signal_fro2, K, W_eeg, snr, params.include_diag)
    sigma_fmri = _sigma_for_target_snr(fmri_signal_fro2, K, W_fmri, snr, params.include_diag)

    if params.common_sigma:
        sigma_common = float(np.sqrt(sigma_eeg * sigma_fmri))
        sigma_eeg = sigma_common
        sigma_fmri = sigma_common

    C_eeg = np.zeros((n, F, K, K), dtype=np.float32)
    for i in range(n):
        for f in range(F):
            Sig = (Phi_true * lam_true[i, f][None, :]) @ Phi_true.T
            E = _sym_gaussian_noise(K, sigma_eeg, rng, params.include_diag)
            C_eeg[i, f] = (W_eeg * Sig + E).astype(np.float32)

    C_fmri = np.zeros((n, K, K), dtype=np.float32)
    for i in range(n):
        Sig_f = np.zeros((K, K), dtype=float)
        for f in m_fmri:
            Sig = (Phi_true * lam_true[i, f][None, :]) @ Phi_true.T
            Sig_f += w_m[f] * Sig
        E = _sym_gaussian_noise(K, sigma_fmri, rng, params.include_diag)
        C_fmri[i] = (W_fmri * Sig_f + E).astype(np.float32)

    out = {
        "Phi_true": Phi_true.astype(np.float32),
        "lambda_true": lam_true.astype(np.float32),
        "C_eeg": C_eeg,
        "C_fmri": C_fmri,
        "W_eeg": W_eeg.astype(np.float32),
        "W_fmri": W_fmri.astype(np.float32),
        "omega": omega.astype(np.float32),
        "m_fmri": m_fmri.astype(np.int32),
        "w_m": w_m.astype(np.float32),
        "sigma_eeg": np.array(sigma_eeg, dtype=np.float64),
        "sigma_fmri": np.array(sigma_fmri, dtype=np.float64),
        "snr_target": np.array(snr, dtype=np.float64),
        "R": np.array(R, dtype=np.int32),
        "kmax_eeg": np.array(kmax_eeg, dtype=np.int32),
        "include_diag": np.array(int(params.include_diag), dtype=np.int32),
        "n": np.array(n, dtype=np.int32),
        "K": np.array(K, dtype=np.int32),
        "F": np.array(F, dtype=np.int32),
        "peak_width": np.array(params.peak_width, dtype=np.float32),
        "tau_beta": np.array(params.tau, dtype=np.float32),
    }

    if save_full:
        Sigma_true = np.zeros((n, F, K, K), dtype=np.float32)
        for i in range(n):
            for f in range(F):
                Sigma_true[i, f] = ((Phi_true * lam_true[i, f][None, :]) @ Phi_true.T).astype(np.float32)
        out["Sigma_true"] = Sigma_true

    return out


def main():
    ap = argparse.ArgumentParser("Generate simulated EEG/fMRI covariance data")
    ap.add_argument("--R", type=int, required=True)
    ap.add_argument("--kmax_eeg", type=int, required=True)
    ap.add_argument("--snr", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="", help="Output .npz path")

    ap.add_argument("--save_full", action="store_true", help="Also save Sigma_true")

    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--K", type=int, default=50)
    ap.add_argument("--F", type=int, default=20)
    ap.add_argument("--omega_min", type=float, default=1.0)
    ap.add_argument("--omega_max", type=float, default=40.0)
    ap.add_argument("--include_diag", type=int, default=1)

    ap.add_argument("--L_spline", type=int, default=4)
    ap.add_argument("--tau", type=float, default=0.4)
    ap.add_argument("--peak_width", type=float, default=0.8)

    args = ap.parse_args()

    params = SimParams(
        seed=args.seed,
        n=args.n,
        K=args.K,
        F=args.F,
        omega_min=args.omega_min,
        omega_max=args.omega_max,
        include_diag=bool(int(args.include_diag)),
        L_spline=args.L_spline,
        tau=args.tau,
        peak_width=args.peak_width,
    )

    data = generate_simulated_data(
        R=args.R,
        kmax_eeg=args.kmax_eeg,
        snr=args.snr,
        params=params,
        save_full=bool(args.save_full),
    )

    if args.out:
        outp = Path(args.out).expanduser().resolve()
        outp.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(outp, **data)
        print("Saved:", str(outp))
        print(f"  R={args.R}, kmax_eeg={args.kmax_eeg}, snr={args.snr}, seed={args.seed}")
        print(f"  sigma_eeg={float(data['sigma_eeg']):.6g}, sigma_fmri={float(data['sigma_fmri']):.6g}")
        if args.save_full:
            print("  save_full=True")
        else:
            print("  save_full=False")
    else:
        print("Generated in-memory (no --out provided).")


if __name__ == "__main__":
    main()