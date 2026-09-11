#!/usr/bin/env python3
"""
rank_screening_evaluation.py

Evaluate adaptive ARD-motivated rank screening under known generating rank.

Primary outputs 


- Full factor-energy vector for post-hoc threshold sensitivity.
- Leading-subspace recovery using the R_true highest-energy fitted  factors.
- Threshold-specific active-subspace recovery for the fit's default threshold.
- High-energy factor similarity after Hungarian matching.
- EEG observed-support, EEG unobserved-region, fMRI, and combined latent
  reconstruction errors.
 
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import scipy.linalg as la
from scipy.optimize import linear_sum_assignment


def sigma_from_phi_lambda(Phi: np.ndarray, lam: np.ndarray) -> np.ndarray:
    return (Phi * lam[None, :]) @ Phi.T


def compute_energy_from_fit(fit: Dict[str, np.ndarray]) -> np.ndarray:
    if "energy_r" in fit:
        return np.asarray(fit["energy_r"], dtype=float)
    lam = np.asarray(fit["lambda_hat"], dtype=float)
    energy = np.mean(lam * lam, axis=(0, 1))
    if "lambda_fmri_hat" in fit:
        lf = np.asarray(fit["lambda_fmri_hat"], dtype=float)
        energy = energy + np.mean(lf * lf, axis=0)
    return energy


def compute_R_eff_from_energy(energy: np.ndarray, rel_thresh: float) -> int:
    if energy.size == 0:
        return 0
    emax = float(np.max(energy))
    if emax <= 0:
        return 0
    return int(np.sum(energy / (emax + 1e-12) >= rel_thresh))


def select_top(energy: np.ndarray, r_keep: int) -> np.ndarray:
    r_keep = max(1, min(int(r_keep), energy.size))
    return np.argsort(-energy)[:r_keep]


def subspace_metrics(Phi_true: np.ndarray, Phi_hat: np.ndarray) -> Dict[str, Any]:
    Ptrue = Phi_true @ Phi_true.T
    Phat = Phi_hat @ Phi_hat.T
    raw = la.norm(Ptrue - Phat, ord="fro")
    rel = raw / max(la.norm(Ptrue, ord="fro"), 1e-12)
    s = la.svdvals(Phi_true.T @ Phi_hat)
    s = np.clip(s, 0.0, 1.0)
    angles = np.degrees(np.arccos(s))
    return {
        "projection_error_raw": float(raw),
        "projection_error_relative": float(rel),
        "principal_angles_deg": angles,
        "angle_mean_deg": float(np.mean(angles)) if angles.size else np.nan,
        "angle_max_deg": float(np.max(angles)) if angles.size else np.nan,
    }


def matched_factor_similarity(
    Phi_true: np.ndarray, Phi_hat_same_rank: np.ndarray
) -> Dict[str, Any]:
    M = Phi_true.T @ Phi_hat_same_rank
    rows, cols = linear_sum_assignment(-np.abs(M))
    order = cols[np.argsort(rows)]
    vals = np.abs(M[np.arange(Phi_true.shape[1]), order])
    return {
        "matched_hat_indices": order,
        "matched_abs_inner_products": vals,
        "matched_abs_inner_mean": float(np.mean(vals)),
        "matched_abs_inner_min": float(np.min(vals)),
    }


def reconstruction_metrics(
    sim: Dict[str, np.ndarray],
    fit: Dict[str, np.ndarray],
    Phi_true: np.ndarray,
    Phi_hat: np.ndarray,
) -> Dict[str, float]:
    lam_true = np.asarray(sim["lambda_true"], dtype=float)
    lam_hat = np.asarray(fit["lambda_hat"], dtype=float)
    W_eeg = np.asarray(sim["W_eeg"], dtype=float)
    W_fmri = np.asarray(sim["W_fmri"], dtype=float)
    include_diag = bool(int(np.asarray(sim.get("include_diag", 1)).item()))

    n, F, _ = lam_true.shape
    K = Phi_true.shape[0]
    iu = np.triu_indices(K, k=0 if include_diag else 1)
    eeg_obs = W_eeg[iu] > 0
    eeg_miss = ~eeg_obs

    num_eeg_obs = den_eeg_obs = 0.0
    num_eeg_miss = den_eeg_miss = 0.0
    num_fmri = den_fmri = 0.0

    for i in range(n):
        for f in range(F):
            St = sigma_from_phi_lambda(Phi_true, lam_true[i, f])
            Sh = sigma_from_phi_lambda(Phi_hat, lam_hat[i, f])
            diff = (Sh - St)[iu]
            tru = St[iu]
            num_eeg_obs += np.sum(diff[eeg_obs] ** 2)
            den_eeg_obs += np.sum(tru[eeg_obs] ** 2)
            if np.any(eeg_miss):
                num_eeg_miss += np.sum(diff[eeg_miss] ** 2)
                den_eeg_miss += np.sum(tru[eeg_miss] ** 2)

        if "lambda_fmri_hat" in fit and "lambda_fmri_true" in sim:
            lf_true = np.asarray(sim["lambda_fmri_true"][i], dtype=float)
            lf_hat = np.asarray(fit["lambda_fmri_hat"][i], dtype=float)
        else:
            w_m = np.asarray(sim["w_m"], dtype=float)
            lf_true = np.einsum("f,fr->r", w_m, lam_true[i])
            lf_hat = np.einsum("f,fr->r", w_m, lam_hat[i])

        Stf = sigma_from_phi_lambda(Phi_true, lf_true)
        Shf = sigma_from_phi_lambda(Phi_hat, lf_hat)
        diff_f = (Shf - Stf)[iu]
        tru_f = Stf[iu]
        mask_f = W_fmri[iu] > 0
        num_fmri += np.sum(diff_f[mask_f] ** 2)
        den_fmri += np.sum(tru_f[mask_f] ** 2)

    eeg_obs_err = np.sqrt(num_eeg_obs / max(den_eeg_obs, 1e-12))
    eeg_miss_err = np.sqrt(num_eeg_miss / max(den_eeg_miss, 1e-12))
    fmri_err = np.sqrt(num_fmri / max(den_fmri, 1e-12))
    combined = np.sqrt(
        (num_eeg_obs + num_fmri) / max(den_eeg_obs + den_fmri, 1e-12)
    )
    return {
        "RelErr_eeg_observed": float(eeg_obs_err),
        "RelErr_eeg_missing": float(eeg_miss_err),
        "RelErr_fmri": float(fmri_err),
        "RelErr_combined": float(combined),
    }


def jsonable(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.integer):
        return int(x)
    return x


def main() -> None:
    ap = argparse.ArgumentParser(
        "Evaluate an ARD fit for covariance-level robustness simulations."
    )
    ap.add_argument("--sim", required=True)
    ap.add_argument("--fit", required=True)
    ap.add_argument("--json_out", required=True)
    ap.add_argument("--ard_rel_thresh", type=float, default=1e-2)
    args = ap.parse_args()

    sim = dict(np.load(args.sim, allow_pickle=True))
    fit = dict(np.load(args.fit, allow_pickle=True))

    Phi_true = np.asarray(sim["Phi_true"], dtype=float)
    Phi_hat = np.asarray(fit["Phi_hat"], dtype=float)
    R_true = Phi_true.shape[1]
    R_hat = Phi_hat.shape[1]

    energy = compute_energy_from_fit(fit)
    energy_order = np.argsort(-energy)
    R_eff = (
        int(np.asarray(fit["R_eff"]).item())
        if "R_eff" in fit
        else compute_R_eff_from_energy(energy, args.ard_rel_thresh)
    )
    rel_thresh = (
        float(np.asarray(fit["R_eff_rel_thresh"]).item())
        if "R_eff_rel_thresh" in fit
        else float(args.ard_rel_thresh)
    )

    idx_top_true = select_top(energy, R_true)
    Phi_top_true = Phi_hat[:, idx_top_true]
    met_top = subspace_metrics(Phi_true, Phi_top_true)
    match = matched_factor_similarity(Phi_true, Phi_top_true)

    idx_active = select_top(energy, max(R_eff, 1))
    Phi_active = Phi_hat[:, idx_active]
    met_active = subspace_metrics(Phi_true, Phi_active)

    rec = reconstruction_metrics(sim, fit, Phi_true, Phi_hat)

    def scalar(name: str, default=np.nan):
        if name not in sim:
            return default
        return np.asarray(sim[name]).item()

    out: Dict[str, Any] = {
        "trueR": int(R_true),
        "Rmax": int(R_hat),
        "R_eff": int(R_eff),
        "R_eff_rel_thresh": float(rel_thresh),
        "n": int(scalar("n")),
        "K": int(scalar("K")),
        "F": int(scalar("F")),
        "KMAX": int(scalar("kmax_eeg")),
        "SNR": float(scalar("snr_target")),
        "seed": int(scalar("seed", -1)),
        "noise_model": str(scalar("noise_model", "gaussian")),
        "t_df": float(scalar("t_df", np.nan)),
        "hetero_gamma": float(scalar("hetero_gamma", np.nan)),
        "energy_r": energy,
        "energy_relative": energy / (np.max(energy) + 1e-12),
        "energy_order": energy_order,
        "topRtrue_indices": idx_top_true,
        "active_indices": idx_active,
        "topRtrue_projection_error_raw": met_top["projection_error_raw"],
        "topRtrue_projection_error_relative": met_top["projection_error_relative"],
        "topRtrue_principal_angles_deg": met_top["principal_angles_deg"],
        "topRtrue_angle_mean_deg": met_top["angle_mean_deg"],
        "topRtrue_angle_max_deg": met_top["angle_max_deg"],
        "topRtrue_matched_abs_inner_products": match["matched_abs_inner_products"],
        "topRtrue_matched_abs_inner_mean": match["matched_abs_inner_mean"],
        "topRtrue_matched_abs_inner_min": match["matched_abs_inner_min"],
        "active_projection_error_raw": met_active["projection_error_raw"],
        "active_projection_error_relative": met_active["projection_error_relative"],
        "active_principal_angles_deg": met_active["principal_angles_deg"],
        "active_angle_mean_deg": met_active["angle_mean_deg"],
        "active_angle_max_deg": met_active["angle_max_deg"],
        **rec,
    }

    if "tau_hat" in fit:
        tau = np.asarray(fit["tau_hat"], dtype=float)
        out.update(
            {
                "tau_min": float(np.min(tau)),
                "tau_median": float(np.median(tau)),
                "tau_max": float(np.max(tau)),
            }
        )

    outp = Path(args.json_out).expanduser().resolve()
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w") as f:
        json.dump({k: jsonable(v) for k, v in out.items()}, f, indent=2)

    print("Saved:", outp)
    print(
        f"noise={out['noise_model']}, trueR={R_true}, n={out['n']}, "
        f"R_eff={R_eff}, threshold={rel_thresh:g}"
    )
    print(
        "top-Rtrue: max angle={:.3f}, rel projection error={:.4f}, "
        "matched similarity={:.4f}".format(
            out["topRtrue_angle_max_deg"],
            out["topRtrue_projection_error_relative"],
            out["topRtrue_matched_abs_inner_mean"],
        )
    )
    print(
        "reconstruction: EEG obs={:.4f}, EEG miss={:.4f}, "
        "fMRI={:.4f}, combined={:.4f}".format(
            out["RelErr_eeg_observed"],
            out["RelErr_eeg_missing"],
            out["RelErr_fmri"],
            out["RelErr_combined"],
        )
    )


if __name__ == "__main__":
    main()
