#!/usr/bin/env python3
"""
rank_screening_simulation.py

Covariance-level EEG/fMRI simulator for compact ARD robustness studies.
 

Residual models
---------------
gaussian
    Homoskedastic Gaussian residuals.
student_t
    Symmetric Student-t residuals standardized to unit variance.
freq_hetero
    Gaussian EEG residuals with frequency-dependent standard deviations.  

The output is compatible with fit_map_fixedR_cov_fast.py and additionally
stores lambda_fmri_true for the separate-fMRI fitting mode.
"""

from __future__ import annotations

import argparse
import json
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
    noise_model: str = "gaussian"
    t_df: float = 5.0
    hetero_gamma: float = 0.5


def _make_simple_spline_basis(omega: np.ndarray, L: int) -> np.ndarray:
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
    return B / (np.sqrt(np.sum(B**2, axis=0, keepdims=True)) + 1e-12)


def _make_mu_templates(
    omega: np.ndarray,
    R: int,
    peak_hz: Optional[List[float]],
    peak_width: float,
    rng: np.random.Generator,
) -> np.ndarray:
    logw = np.log(omega)
    if peak_hz is None:
        base = np.array([6.0, 10.0, 20.0, 30.0])
        if R <= len(base):
            peaks = base[:R]
        else:
            extra = rng.uniform(5.0, 35.0, size=R - len(base))
            peaks = np.concatenate([base, extra])
    else:
        if len(peak_hz) < R:
            raise ValueError("peak_hz must have length >= R")
        peaks = np.asarray(peak_hz[:R], dtype=float)

    mu = np.zeros((R, omega.size), dtype=float)
    for r in range(R):
        c = np.log(peaks[r])
        mu[r] = -0.5 * ((logw - c) / peak_width) ** 2
    return mu


def _make_eeg_mask(K: int, kmax_eeg: int, include_diag: bool) -> np.ndarray:
    if not (1 <= kmax_eeg <= K):
        raise ValueError("kmax_eeg must be in [1,K]")
    ok = np.arange(K) < kmax_eeg
    W = np.outer(ok.astype(float), ok.astype(float))
    if not include_diag:
        np.fill_diagonal(W, 0.0)
    return W


def _make_full_mask(K: int, include_diag: bool) -> np.ndarray:
    W = np.ones((K, K), dtype=float)
    if not include_diag:
        np.fill_diagonal(W, 0.0)
    return W


def _expected_noise_fro2_unit(K: int, mask: np.ndarray, include_diag: bool) -> float:
    if include_diag:
        iu = np.triu_indices(K, k=0)
        w = mask[iu]
        is_diag = iu[0] == iu[1]
        return float(np.sum(w[is_diag] ** 2) + 2.0 * np.sum(w[~is_diag] ** 2))
    iu = np.triu_indices(K, k=1)
    return float(2.0 * np.sum(mask[iu] ** 2))


def _sigma_for_target_snr(
    signal_fro2_mean: float,
    K: int,
    mask: np.ndarray,
    snr: float,
    include_diag: bool,
) -> float:
    if snr <= 0:
        raise ValueError("snr must be positive")
    denom_unit = _expected_noise_fro2_unit(K, mask, include_diag)
    if denom_unit <= 0:
        raise ValueError("Mask has no observable entries")
    return float(np.sqrt(max(signal_fro2_mean / (snr * denom_unit), 1e-16)))


def _unit_variance_draw(
    size: int,
    rng: np.random.Generator,
    noise_model: str,
    t_df: float,
) -> np.ndarray:
    if noise_model in {"gaussian", "freq_hetero"}:
        return rng.normal(size=size)
    if noise_model == "student_t":
        if t_df <= 2:
            raise ValueError("Student-t degrees of freedom must exceed 2")
        z = rng.standard_t(df=t_df, size=size)
        return z * np.sqrt((t_df - 2.0) / t_df)
    raise ValueError(f"Unsupported noise_model: {noise_model}")


def _sym_noise(
    K: int,
    sigma: float,
    rng: np.random.Generator,
    include_diag: bool,
    noise_model: str,
    t_df: float,
) -> np.ndarray:
    E = np.zeros((K, K), dtype=float)
    iu = np.triu_indices(K, k=0 if include_diag else 1)
    E[iu] = sigma * _unit_variance_draw(
        iu[0].size, rng, noise_model=noise_model, t_df=t_df
    )
    if include_diag:
        return E + E.T - np.diag(np.diag(E))
    return E + E.T


def _hetero_multipliers(F: int, gamma: float) -> np.ndarray:
    z = np.linspace(-1.0, 1.0, F)
    raw = np.exp(gamma * z)
    return raw / np.sqrt(np.mean(raw**2))


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
    lam_true = np.transpose(np.exp(loglam), (0, 2, 1))

    W_eeg = _make_eeg_mask(K, kmax_eeg, params.include_diag)
    W_fmri = _make_full_mask(K, params.include_diag)

    m_fmri = np.asarray(params.m_fmri, dtype=int)
    if np.any(m_fmri < 0) or np.any(m_fmri >= F):
        raise ValueError("m_fmri indices must lie in [0,F-1]")
    w_m = np.zeros(F, dtype=float)
    w_m[m_fmri] = 1.0 / len(m_fmri)
    lambda_fmri_true = np.einsum("f,nfr->nr", w_m, lam_true)

    eeg_signal_vals = []
    fmri_signal_vals = []
    for i in range(n):
        for f in range(F):
            Sig = (Phi_true * lam_true[i, f][None, :]) @ Phi_true.T
            eeg_signal_vals.append(np.sum((W_eeg * Sig) ** 2))
        Sig_fmri = (Phi_true * lambda_fmri_true[i][None, :]) @ Phi_true.T
        fmri_signal_vals.append(np.sum((W_fmri * Sig_fmri) ** 2))

    sigma_eeg = _sigma_for_target_snr(
        float(np.mean(eeg_signal_vals)), K, W_eeg, snr, params.include_diag
    )
    sigma_fmri = _sigma_for_target_snr(
        float(np.mean(fmri_signal_vals)), K, W_fmri, snr, params.include_diag
    )
    if params.common_sigma:
        sigma_common = float(np.sqrt(sigma_eeg * sigma_fmri))
        sigma_eeg = sigma_common
        sigma_fmri = sigma_common

    g_f = (
        _hetero_multipliers(F, params.hetero_gamma)
        if params.noise_model == "freq_hetero"
        else np.ones(F, dtype=float)
    )

    C_eeg = np.zeros((n, F, K, K), dtype=np.float32)
    for i in range(n):
        for f in range(F):
            Sig = (Phi_true * lam_true[i, f][None, :]) @ Phi_true.T
            E = _sym_noise(
                K,
                sigma_eeg * g_f[f],
                rng,
                params.include_diag,
                params.noise_model,
                params.t_df,
            )
            C_eeg[i, f] = (W_eeg * Sig + E).astype(np.float32)

    fmri_noise_model = "student_t" if params.noise_model == "student_t" else "gaussian"
    C_fmri = np.zeros((n, K, K), dtype=np.float32)
    for i in range(n):
        Sig_fmri = (Phi_true * lambda_fmri_true[i][None, :]) @ Phi_true.T
        E = _sym_noise(
            K,
            sigma_fmri,
            rng,
            params.include_diag,
            fmri_noise_model,
            params.t_df,
        )
        C_fmri[i] = (W_fmri * Sig_fmri + E).astype(np.float32)

    metadata = {
        "seed": int(params.seed),
        "noise_model": params.noise_model,
        "t_df": float(params.t_df),
        "hetero_gamma": float(params.hetero_gamma),
        "common_sigma": bool(params.common_sigma),
        "n": int(n),
        "K": int(K),
        "F": int(F),
        "R_true": int(R),
        "kmax_eeg": int(kmax_eeg),
        "snr_target": float(snr),
    }

    out: Dict[str, np.ndarray] = {
        "Phi_true": Phi_true.astype(np.float32),
        "lambda_true": lam_true.astype(np.float32),
        "lambda_fmri_true": lambda_fmri_true.astype(np.float32),
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
        "noise_model": np.array(params.noise_model),
        "t_df": np.array(params.t_df, dtype=np.float64),
        "hetero_gamma": np.array(params.hetero_gamma, dtype=np.float64),
        "hetero_multipliers": g_f.astype(np.float32),
        "seed": np.array(params.seed, dtype=np.int64),
        "metadata_json": np.array(json.dumps(metadata)),
    }

    if save_full:
        Sigma_true = np.zeros((n, F, K, K), dtype=np.float32)
        for i in range(n):
            for f in range(F):
                Sigma_true[i, f] = (
                    (Phi_true * lam_true[i, f][None, :]) @ Phi_true.T
                ).astype(np.float32)
        out["Sigma_true"] = Sigma_true

    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        "Generate covariance-level EEG/fMRI simulations for ARD robustness."
    )
    ap.add_argument("--R", type=int, required=True)
    ap.add_argument("--kmax_eeg", type=int, required=True)
    ap.add_argument("--snr", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--K", type=int, default=50)
    ap.add_argument("--F", type=int, default=20)
    ap.add_argument("--omega_min", type=float, default=1.0)
    ap.add_argument("--omega_max", type=float, default=40.0)
    ap.add_argument("--include_diag", type=int, default=1)
    ap.add_argument("--L_spline", type=int, default=4)
    ap.add_argument("--tau", type=float, default=0.4)
    ap.add_argument("--peak_width", type=float, default=0.8)
    ap.add_argument(
        "--noise_model",
        choices=["gaussian", "student_t", "freq_hetero"],
        default="gaussian",
    )
    ap.add_argument("--t_df", type=float, default=5.0)
    ap.add_argument("--hetero_gamma", type=float, default=0.5)
    ap.add_argument("--common_sigma", type=int, default=1)
    ap.add_argument("--save_full", action="store_true")
    args = ap.parse_args()

    params = SimParams(
        seed=args.seed,
        n=args.n,
        K=args.K,
        F=args.F,
        omega_min=args.omega_min,
        omega_max=args.omega_max,
        include_diag=bool(args.include_diag),
        L_spline=args.L_spline,
        tau=args.tau,
        peak_width=args.peak_width,
        common_sigma=bool(args.common_sigma),
        noise_model=args.noise_model,
        t_df=args.t_df,
        hetero_gamma=args.hetero_gamma,
    )
    data = generate_simulated_data(
        R=args.R,
        kmax_eeg=args.kmax_eeg,
        snr=args.snr,
        params=params,
        save_full=args.save_full,
    )

    outp = Path(args.out).expanduser().resolve()
    outp.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outp, **data)
    print("Saved:", outp)
    print(
        f"  R={args.R}, n={args.n}, kmax={args.kmax_eeg}, "
        f"SNR={args.snr}, noise={args.noise_model}, seed={args.seed}"
    )
    print(
        f"  sigma_eeg={float(data['sigma_eeg']):.6g}, "
        f"sigma_fmri={float(data['sigma_fmri']):.6g}"
    )
    if args.noise_model == "freq_hetero":
        g = np.asarray(data["hetero_multipliers"])
        print(
            f"  hetero multiplier min/max={g.min():.4f}/{g.max():.4f}; "
            f"RMS={np.sqrt(np.mean(g*g)):.4f}"
        )


if __name__ == "__main__":
    main()
