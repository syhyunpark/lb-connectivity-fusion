#!/usr/bin/env python3
"""
observation_model_evaluation.py

Evaluate full and EEG-observed subspace recovery, reconstruction error,
connectivity-level SNR, oracle benchmarks  and  fit stability 
for the observation-model validation simulations.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import scipy.linalg as la
from scipy.optimize import linear_sum_assignment, lsq_linear


Array = np.ndarray


def ensure_parent(path: str | Path) -> None:
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def load_metadata(z: np.lib.npyio.NpzFile) -> Dict[str, Any]:
    if "metadata_json" in z:
        return json.loads(str(np.asarray(z["metadata_json"]).item()))
    return {}


def get_fit_array(fit: np.lib.npyio.NpzFile, keys: Iterable[str]) -> Array | None:
    for key in keys:
        if key in fit:
            return np.asarray(fit[key], dtype=float)
    return None


def as_scalar(z: np.lib.npyio.NpzFile, key: str, default: Any = None) -> Any:
    if key not in z:
        return default
    arr = np.asarray(z[key])
    if arr.size == 1:
        value = arr.reshape(-1)[0]
        return value.item() if hasattr(value, "item") else value
    return default


def sigma_from_phi_lambda(Phi: Array, lam: Array) -> Array:
    return (Phi * lam[None, :]) @ Phi.T


def orthonormal_basis(A: Array, tol: float = 1e-10) -> Array:
    U, s, _ = la.svd(np.asarray(A, dtype=float), full_matrices=False)
    if s.size == 0 or s[0] <= 0:
        return np.zeros((A.shape[0], 0), dtype=float)
    rank = int(np.sum(s > tol * s[0]))
    return U[:, :rank]


def subspace_metrics(Phi_ref: Array, Phi_hat: Array, prefix: str = "") -> Dict[str, Any]:
    A = orthonormal_basis(Phi_ref)
    B = orthonormal_basis(Phi_hat)
    if A.shape[1] == 0 or B.shape[1] == 0:
        return {
            f"{prefix}principal_angles_deg": [],
            f"{prefix}mean_principal_angle_deg": float("nan"),
            f"{prefix}max_principal_angle_deg": float("nan"),
            f"{prefix}subspace_miss_error": float("nan"),
            f"{prefix}projection_error_rel": float("nan"),
            f"{prefix}reference_rank": int(A.shape[1]),
            f"{prefix}estimated_rank": int(B.shape[1]),
        }
    s = np.clip(la.svdvals(A.T @ B), 0.0, 1.0)
    angles = np.degrees(np.arccos(s))
    P_ref = A @ A.T
    P_hat = B @ B.T
    captured = float(np.sum(s**2))
    miss = np.sqrt(max(0.0, A.shape[1] - captured)) / np.sqrt(A.shape[1])
    proj = float(la.norm(P_ref - P_hat, ord="fro") / max(la.norm(P_ref, ord="fro"), 1e-12))
    return {
        f"{prefix}principal_angles_deg": angles.tolist(),
        f"{prefix}mean_principal_angle_deg": float(np.mean(angles)),
        f"{prefix}max_principal_angle_deg": float(np.max(angles)),
        f"{prefix}subspace_miss_error": float(miss),
        f"{prefix}projection_error_rel": proj,
        f"{prefix}reference_rank": int(A.shape[1]),
        f"{prefix}estimated_rank": int(B.shape[1]),
    }


def weighted_subspace_metrics(Phi_ref: Array, Phi_hat: Array, q: Array, prefix: str) -> Dict[str, Any]:
    q = np.asarray(q, dtype=float).reshape(-1)
    return subspace_metrics(q[:, None] * Phi_ref, q[:, None] * Phi_hat, prefix=prefix)


def match_components(Phi_true: Array, Phi_hat: Array) -> Tuple[Array, Array, Array, Array]:
    M = Phi_true.T @ Phi_hat
    row, col = linear_sum_assignment(-np.abs(M))
    order = np.argsort(row)
    row, col = row[order], col[order]
    matched = Phi_hat[:, col]
    signs = np.sign(np.sum(Phi_true[:, row] * matched, axis=0))
    signs[signs == 0] = 1.0
    matched *= signs[None, :]
    inner = np.sum(Phi_true[:, row] * matched, axis=0)
    return row, col, signs, inner


def masked_stack_sums(A: Array, B: Array, W: Array) -> Tuple[float, float]:
    diff = (A - B) * W
    truth = A * W
    return float(np.sum(diff**2)), float(np.sum(truth**2))


def masked_relerr(A: Array, B: Array, W: Array) -> float:
    num, den = masked_stack_sums(A, B, W)
    return float(np.sqrt(num / max(den, 1e-12)))


def masked_summary_snr(C_clean: Array, C_noisy: Array, W: Array) -> float:
    sig2 = float(np.sum((C_clean * W) ** 2))
    err2 = float(np.sum(((C_noisy - C_clean) * W) ** 2))
    return float(sig2 / max(err2, 1e-12))


def reconstruct_eeg(Phi: Array, lam: Array) -> Array:
    n, F, _ = lam.shape
    K = Phi.shape[0]
    out = np.empty((n, F, K, K), dtype=np.float32)
    for i in range(n):
        for f in range(F):
            out[i, f] = sigma_from_phi_lambda(Phi, lam[i, f]).astype(np.float32)
    return out


def reconstruct_fmri(Phi: Array, lam: Array) -> Array:
    return np.stack([sigma_from_phi_lambda(Phi, x) for x in lam], axis=0).astype(np.float32)


def upper_design(Phi: Array, W: Array) -> Tuple[Array, Tuple[Array, Array], Array, Array]:
    K, R = Phi.shape
    iu = np.triu_indices(K, k=0)
    keep = np.where(W[iu] > 0)[0]
    frob = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
    cols = []
    for r in range(R):
        Br = np.outer(Phi[:, r], Phi[:, r])[iu]
        cols.append((frob * W[iu] * Br)[keep])
    return np.stack(cols, axis=1), iu, keep, frob


def fixed_phi_nnls_reconstruction(C: Array, Phi: Array, W: Array) -> Tuple[float, Array]:
    A, iu, keep, frob = upper_design(Phi, W)
    lead_shape = C.shape[:-2]
    C2 = C.reshape((-1,) + C.shape[-2:])
    recon = np.empty_like(C2, dtype=np.float32)
    weights = np.empty((C2.shape[0], Phi.shape[1]), dtype=float)
    obs_weight = (frob * W[iu])[keep]
    for j, M in enumerate(C2):
        y = (obs_weight * M[iu][keep]).astype(float)
        fit = lsq_linear(A, y, bounds=(0.0, np.inf), method="bvls", verbose=0)
        weights[j] = fit.x
        recon[j] = sigma_from_phi_lambda(Phi, fit.x).astype(np.float32)
    recon = recon.reshape(C.shape)
    return masked_relerr(C, recon, W), weights.reshape(lead_shape + (Phi.shape[1],))


def fixed_band_phi_nnls_reconstruction(C_eeg: Array, Phi_bands: Array, W: Array) -> float:
    n, F = C_eeg.shape[:2]
    if Phi_bands.shape[0] != F:
        raise ValueError("Phi_bands frequency dimension does not match C_eeg")
    num = den = 0.0
    for f in range(F):
        A, iu, keep, frob = upper_design(Phi_bands[f], W)
        obs_weight = (frob * W[iu])[keep]
        for M in C_eeg[:, f]:
            y = obs_weight * M[iu][keep]
            fit = lsq_linear(A, y, bounds=(0.0, np.inf), method="bvls", verbose=0)
            H = sigma_from_phi_lambda(Phi_bands[f], fit.x)
            n0, d0 = masked_stack_sums(M, H, W)
            num += n0
            den += d0
    return float(np.sqrt(num / max(den, 1e-12)))


def best_rank_r_psd(M: Array, R: int) -> Array:
    vals, vecs = la.eigh(0.5 * (M + M.T))
    order = np.argsort(vals)[::-1]
    vals = np.maximum(vals[order[:R]], 0.0)
    vecs = vecs[:, order[:R]]
    return (vecs * vals[None, :]) @ vecs.T


def independent_rank_r_psd_relerr(C: Array, R: int, W: Array) -> float:
    C2 = C.reshape((-1,) + C.shape[-2:])
    num = den = 0.0
    for M in C2:
        H = best_rank_r_psd(M, R)
        n0, d0 = masked_stack_sums(M, H, W)
        num += n0
        den += d0
    return float(np.sqrt(num / max(den, 1e-12)))


def compute_benchmarks(data: np.lib.npyio.NpzFile, cache_path: Path | None) -> Dict[str, float]:
    if cache_path is not None and cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    Phi_true = np.asarray(data["Phi_true"], dtype=float)
    C_eeg_clean = np.asarray(data["C_eeg_clean"], dtype=float)
    C_fmri_clean = np.asarray(data["C_fmri_clean"], dtype=float)
    W_eeg = np.asarray(data["W_eeg"], dtype=float)
    W_fmri = np.asarray(data["W_fmri"], dtype=float)
    R = Phi_true.shape[1]

    eeg_common, _ = fixed_phi_nnls_reconstruction(C_eeg_clean, Phi_true, W_eeg)
    fmri_common, _ = fixed_phi_nnls_reconstruction(C_fmri_clean, Phi_true, W_fmri)
    out = {
        "eeg_common_phi_oracle_relerr": float(eeg_common),
        "fmri_common_phi_oracle_relerr": float(fmri_common),
        "eeg_independent_rankR_psd_relerr": float(independent_rank_r_psd_relerr(C_eeg_clean, R, W_eeg)),
        "fmri_independent_rankR_psd_relerr": float(independent_rank_r_psd_relerr(C_fmri_clean, R, W_fmri)),
    }
    if "Phi_band_true" in data:
        out["eeg_band_truth_oracle_relerr"] = float(
            fixed_band_phi_nnls_reconstruction(
                C_eeg_clean,
                np.asarray(data["Phi_band_true"], dtype=float),
                W_eeg,
            )
        )
    else:
        out["eeg_band_truth_oracle_relerr"] = float(eeg_common)

    if cache_path is not None:
        ensure_parent(cache_path)
        with open(cache_path, "w") as f:
            json.dump(out, f, indent=2)
    return out


def corr_safe(x: Array, y: Array) -> float:
    x, y = np.asarray(x).ravel(), np.asarray(y).ravel()
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate redesigned observation-model validation recovery")
    ap.add_argument("--data", required=True)
    ap.add_argument("--fit", required=True)
    ap.add_argument("--fit_mode", choices=["fused", "eeg_only", "fmri_only"], required=True)
    ap.add_argument("--fit_variant", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--benchmark_cache", default="")
    ap.add_argument("--clean_reference_fit", default="")
    args = ap.parse_args()

    data = np.load(args.data, allow_pickle=True)
    fit = np.load(args.fit, allow_pickle=True)
    meta = load_metadata(data)

    Phi_true = np.asarray(data["Phi_true"], dtype=float)
    Phi_hat = get_fit_array(fit, ("Phi_hat", "Phi", "B_hat"))
    if Phi_hat is None:
        raise KeyError("Fit does not contain Phi_hat")
    if Phi_true.shape[0] != Phi_hat.shape[0]:
        raise ValueError("Phi dimensions do not match")

    out: Dict[str, Any] = dict(meta)
    out.update(
        {
            "data": Path(args.data).name,
            "fit": Path(args.fit).name,
            "fit_mode": args.fit_mode,
            "fit_variant": args.fit_variant,
            "fit_target": str(as_scalar(fit, "fit_target", "noisy")),
            "K": int(Phi_true.shape[0]),
            "R_true": int(Phi_true.shape[1]),
            "R_hat": int(Phi_hat.shape[1]),
            "J_final": float(as_scalar(fit, "J_final", np.nan)),
            "multistart_best_method": str(as_scalar(fit, "multistart_best_method", "")),
            "multistart_best_seed": int(as_scalar(fit, "multistart_best_seed", -1)),
        }
    )
    out.update(subspace_metrics(Phi_true, Phi_hat))

    q_eeg = np.asarray(data["q_eeg"], dtype=float)
    out.update(weighted_subspace_metrics(Phi_true, Phi_hat, q_eeg, prefix="eeg_observed_"))

    row, col, signs, inner = match_components(Phi_true, Phi_hat)
    out["matched_hat_factors"] = [int(x + 1) for x in col.tolist()]
    out["mean_abs_phi_inner_product"] = float(np.mean(np.abs(inner)))
    out["min_abs_phi_inner_product"] = float(np.min(np.abs(inner)))

    C_eeg_clean = np.asarray(data["C_eeg_clean"], dtype=float)
    C_fmri_clean = np.asarray(data["C_fmri_clean"], dtype=float)
    C_eeg = np.asarray(data["C_eeg"], dtype=float)
    C_fmri = np.asarray(data["C_fmri"], dtype=float)
    W_eeg = np.asarray(data["W_eeg"], dtype=float)
    W_fmri = np.asarray(data["W_fmri"], dtype=float)

    out["summary_snr_eeg_recomputed"] = masked_summary_snr(C_eeg_clean, C_eeg, W_eeg)
    out["summary_snr_fmri_recomputed"] = masked_summary_snr(C_fmri_clean, C_fmri, W_fmri)

    cache = Path(args.benchmark_cache).expanduser().resolve() if args.benchmark_cache else None
    out.update(compute_benchmarks(data, cache))

    lam_eeg_hat = get_fit_array(fit, ("lambda_hat", "lambda_eeg_hat", "L_hat"))
    lam_fmri_hat = get_fit_array(fit, ("lambda_fmri_hat", "lambda_f_hat"))
    out["eeg_fit_clean_relerr"] = float("nan")
    out["fmri_fit_clean_relerr"] = float("nan")
    out["eeg_fit_noisy_relerr"] = float("nan")
    out["fmri_fit_noisy_relerr"] = float("nan")

    if args.fit_mode in {"fused", "eeg_only"} and lam_eeg_hat is not None:
        C_hat = reconstruct_eeg(Phi_hat, lam_eeg_hat)
        out["eeg_fit_clean_relerr"] = masked_relerr(C_eeg_clean, C_hat, W_eeg)
        out["eeg_fit_noisy_relerr"] = masked_relerr(C_eeg, C_hat, W_eeg)
        out["eeg_excess_over_common_phi_oracle"] = (
            out["eeg_fit_clean_relerr"] - out["eeg_common_phi_oracle_relerr"]
        )
        out["eeg_excess_over_band_truth_oracle"] = (
            out["eeg_fit_clean_relerr"] - out["eeg_band_truth_oracle_relerr"]
        )

    if args.fit_mode in {"fused", "fmri_only"} and lam_fmri_hat is not None:
        C_hat = reconstruct_fmri(Phi_hat, lam_fmri_hat)
        out["fmri_fit_clean_relerr"] = masked_relerr(C_fmri_clean, C_hat, W_fmri)
        out["fmri_fit_noisy_relerr"] = masked_relerr(C_fmri, C_hat, W_fmri)
        out["fmri_excess_over_common_phi_oracle"] = (
            out["fmri_fit_clean_relerr"] - out["fmri_common_phi_oracle_relerr"]
        )

    if args.clean_reference_fit:
        with np.load(args.clean_reference_fit, allow_pickle=True) as z:
            Phi_clean_fit = np.asarray(z["Phi_hat"], dtype=float)
        out.update(subspace_metrics(Phi_clean_fit, Phi_hat, prefix="noisy_to_clean_"))
        out.update(
            weighted_subspace_metrics(
                Phi_clean_fit,
                Phi_hat,
                q_eeg,
                prefix="eeg_observed_noisy_to_clean_",
            )
        )

    per_factor = []
    lam_eeg_true = np.asarray(data["lambda_eeg_true"], dtype=float) if "lambda_eeg_true" in data else None
    lam_fmri_true = np.asarray(data["lambda_fmri_true"], dtype=float) if "lambda_fmri_true" in data else None
    for j, (rt, rh) in enumerate(zip(row, col)):
        item: Dict[str, Any] = {
            "true_factor": int(rt + 1),
            "matched_hat_factor": int(rh + 1),
            "sign": float(signs[j]),
            "phi_inner_product": float(inner[j]),
        }
        if lam_eeg_true is not None and lam_eeg_hat is not None and args.fit_mode in {"fused", "eeg_only"}:
            item["eeg_lambda_corr_secondary"] = corr_safe(lam_eeg_true[:, :, rt], lam_eeg_hat[:, :, rh])
        if lam_fmri_true is not None and lam_fmri_hat is not None and args.fit_mode in {"fused", "fmri_only"}:
            item["fmri_lambda_corr_secondary"] = corr_safe(lam_fmri_true[:, rt], lam_fmri_hat[:, rh])
        per_factor.append(item)

    if "Phi_band_true" in data:
        Phi_bands = np.asarray(data["Phi_band_true"], dtype=float)
        full_to_common_mean = []
        full_to_common_max = []
        fit_to_band_mean = []
        fit_to_band_max = []
        observed_fit_to_band_mean = []
        observed_fit_to_band_max = []
        for Pb in Phi_bands:
            m_truth = subspace_metrics(Phi_true, Pb, prefix="x_")
            m_fit = subspace_metrics(Pb, Phi_hat, prefix="x_")
            m_obs = weighted_subspace_metrics(Pb, Phi_hat, q_eeg, prefix="x_")
            full_to_common_mean.append(m_truth["x_mean_principal_angle_deg"])
            full_to_common_max.append(m_truth["x_max_principal_angle_deg"])
            fit_to_band_mean.append(m_fit["x_mean_principal_angle_deg"])
            fit_to_band_max.append(m_fit["x_max_principal_angle_deg"])
            observed_fit_to_band_mean.append(m_obs["x_mean_principal_angle_deg"])
            observed_fit_to_band_max.append(m_obs["x_max_principal_angle_deg"])
        out["true_band_to_common_mean_angle_deg"] = float(np.mean(full_to_common_mean))
        out["true_band_to_common_max_angle_deg"] = float(np.max(full_to_common_max))
        out["fit_to_eeg_bands_mean_angle_deg"] = float(np.mean(fit_to_band_mean))
        out["fit_to_eeg_bands_max_angle_deg"] = float(np.max(fit_to_band_max))
        out["eeg_observed_fit_to_bands_mean_angle_deg"] = float(np.mean(observed_fit_to_band_mean))
        out["eeg_observed_fit_to_bands_max_angle_deg"] = float(np.max(observed_fit_to_band_max))

    ensure_parent(args.out_json)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2, allow_nan=True)

    ensure_parent(args.out_csv)
    fields = sorted({k for row0 in per_factor for k in row0})
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(per_factor)

    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_csv}")
    print("scenario:", out.get("scenario_label", out.get("scenario")))
    print("fit_variant:", args.fit_variant)
    print("max principal angle:", round(out["max_principal_angle_deg"], 4))
    print("EEG-observed max angle:", round(out["eeg_observed_max_principal_angle_deg"], 4))
    print("summary SNR EEG/fMRI:", round(out["summary_snr_eeg_recomputed"], 3), round(out["summary_snr_fmri_recomputed"], 3))


if __name__ == "__main__":
    main()
