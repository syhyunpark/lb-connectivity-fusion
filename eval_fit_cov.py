#eval_fit_cov.py
#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from typing import Dict, Tuple, Optional, Any

import numpy as np
import scipy.linalg as la


def sigma_from_Phi_lambda(Phi: np.ndarray, lam: np.ndarray) -> np.ndarray:
    return (Phi * lam[None, :]) @ Phi.T


def procrustes_align(Phi_true: np.ndarray, Phi_hat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    M = Phi_hat.T @ Phi_true
    U, _, Vt = la.svd(M, full_matrices=False)
    Q = U @ Vt
    Phi_hat_al = Phi_hat @ Q
    return Phi_hat_al, Q


def rotate_lambda_diagonal(lam_hat: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rotate diagonal lambda weights into the aligned basis.

    Returns:
      lam_al_diag: (n, M, R)
      offdiag_norm: (n, M)
    """
    n, M, R = lam_hat.shape
    lam_al = np.zeros_like(lam_hat)
    offn = np.zeros((n, M), dtype=float)

    for i in range(n):
        for m in range(M):
            D = np.diag(lam_hat[i, m])
            A = Q.T @ D @ Q
            lam_al[i, m] = np.diag(A)
            A_off = A - np.diag(np.diag(A))
            offn[i, m] = la.norm(A_off, ord="fro")
    return lam_al, offn


def subspace_metrics(Phi_true: np.ndarray, Phi_hat: np.ndarray) -> Dict[str, Any]:
    Ptrue = Phi_true @ Phi_true.T
    Phat = Phi_hat @ Phi_hat.T
    dsub = la.norm(Ptrue - Phat, ord="fro")

    M = Phi_true.T @ Phi_hat
    U, s, Vt = la.svd(M, full_matrices=False)
    s = np.clip(s, -1.0, 1.0)
    angles = np.degrees(np.arccos(s))

    return {
        "dsub": float(dsub),
        "angles_deg": angles,
        "angle_mean_deg": float(np.mean(angles)) if angles.size else float("nan"),
        "angle_max_deg": float(np.max(angles)) if angles.size else float("nan"),
    }


def lambda_metrics(lambda_true: np.ndarray, lambda_hat: np.ndarray, omega: np.ndarray) -> Dict[str, Any]:
    rmse = float(np.sqrt(np.mean((lambda_hat - lambda_true) ** 2)))

    n, M, R = lambda_true.shape
    corr_mr = np.zeros((M, R))
    for m in range(M):
        for r in range(R):
            x = lambda_hat[:, m, r]
            y = lambda_true[:, m, r]
            corr_mr[m, r] = np.corrcoef(x, y)[0, 1]
    corr_r = np.nanmean(corr_mr, axis=0)

    bands = {"theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0)}
    band_corr = {}
    for name, (lo, hi) in bands.items():
        idx = np.where((omega >= lo) & (omega < hi))[0]
        if idx.size == 0:
            band_corr[name] = np.nan
            continue
        L_true = lambda_true[:, idx, :].sum(axis=1)
        L_hat = lambda_hat[:, idx, :].sum(axis=1)
        cc = []
        for r in range(R):
            cc.append(np.corrcoef(L_hat[:, r], L_true[:, r])[0, 1])
        band_corr[name] = float(np.nanmean(cc))

    return {
        "rmse_lambda": rmse,
        "corr_r_mean_over_omega": corr_r,
        "band_corr_mean_over_r": band_corr,
    }


def region_errors_general(
    Phi_true: np.ndarray,
    lambda_true: np.ndarray,
    Phi_hat: np.ndarray,
    lambda_hat: np.ndarray,
    W_eeg: np.ndarray,
    W_fmri: np.ndarray,
    m_fmri: np.ndarray,
    kmax_eeg: int,
    include_diag: bool,
) -> Dict[str, float]:
    n, M, Rt = lambda_true.shape
    Rh = lambda_hat.shape[2]
    K = Phi_true.shape[0]
    iu = np.triu_indices(K, k=0 if include_diag else 1)

    eeg_obs_edges = np.where(W_eeg[iu] > 0)[0]

    idx = np.arange(K)
    ok = idx < kmax_eeg
    W_block = np.outer(ok.astype(float), ok.astype(float))
    if not include_diag:
        np.fill_diagonal(W_block, 0.0)
    eeg_miss_edges = np.where(W_block[iu] == 0)[0]

    m_fmri_set = set(int(x) for x in m_fmri.tolist())

    rel_obs, rel_eegmiss, rel_fmrimiss = [], [], []

    for i in range(n):
        num_obs = den_obs = 0.0
        num_eeg = den_eeg = 0.0
        num_fm = den_fm = 0.0

        for m in range(M):
            S_true = sigma_from_Phi_lambda(Phi_true, lambda_true[i, m])
            S_hat = sigma_from_Phi_lambda(Phi_hat, lambda_hat[i, m])

            if m in m_fmri_set:
                diff = (S_hat - S_true)[iu]
                tru = S_true[iu]
                num_obs += np.sum(diff**2)
                den_obs += np.sum(tru**2)
            else:
                diff = (S_hat - S_true)[iu][eeg_obs_edges]
                tru = S_true[iu][eeg_obs_edges]
                num_obs += np.sum(diff**2)
                den_obs += np.sum(tru**2)

            diff_e = (S_hat - S_true)[iu][eeg_miss_edges]
            tru_e = S_true[iu][eeg_miss_edges]
            num_eeg += np.sum(diff_e**2)
            den_eeg += np.sum(tru_e**2)

            if m not in m_fmri_set:
                diff_f = (S_hat - S_true)[iu]
                tru_f = S_true[iu]
                num_fm += np.sum(diff_f**2)
                den_fm += np.sum(tru_f**2)

        rel_obs.append(np.sqrt(num_obs / max(den_obs, 1e-12)))
        rel_eegmiss.append(np.sqrt(num_eeg / max(den_eeg, 1e-12)))
        rel_fmrimiss.append(np.sqrt(num_fm / max(den_fm, 1e-12)))

    return {
        "RelErr_obs_mean": float(np.mean(rel_obs)),
        "RelErr_eeg_miss_mean": float(np.mean(rel_eegmiss)),
        "RelErr_fmri_miss_mean": float(np.mean(rel_fmrimiss)),
    }


def compute_energy_from_fit(fit: Dict[str, np.ndarray]) -> np.ndarray:
    if "energy_r" in fit:
        return np.asarray(fit["energy_r"], dtype=float)
    lam = fit["lambda_hat"]
    return np.mean(lam * lam, axis=(0, 1))


def compute_R_eff_from_energy(energy: np.ndarray, rel_thresh: float) -> int:
    if energy.size == 0:
        return 0
    emax = float(np.max(energy))
    if emax <= 0:
        return 0
    return int(np.sum((energy / (emax + 1e-12)) >= rel_thresh))


def select_top_factors_by_energy(energy: np.ndarray, R_keep: int) -> np.ndarray:
    if energy.size == 0:
        return np.array([], dtype=int)
    order = np.argsort(-energy)
    R_keep = int(max(1, min(R_keep, energy.size)))
    return order[:R_keep]


def ard_summary(fit: Dict[str, np.ndarray], R_true: int, rel_thresh_default: float = 1e-2) -> Dict[str, Any]:
    lam_hat = fit["lambda_hat"]
    Rh = lam_hat.shape[2]
    out: Dict[str, Any] = {"is_ard": True, "Rmax": int(Rh)}

    energy = compute_energy_from_fit(fit)
    out["energy_r"] = energy

    tau = fit.get("tau_hat", None)
    if tau is not None:
        tau = np.asarray(tau, dtype=float)
        out["tau_min"] = float(np.min(tau))
        out["tau_med"] = float(np.median(tau))
        out["tau_max"] = float(np.max(tau))
    else:
        out["tau_min"] = out["tau_med"] = out["tau_max"] = float("nan")

    rel_thresh = float(fit.get("R_eff_rel_thresh", rel_thresh_default))
    out["R_eff_rel_thresh"] = rel_thresh

    if "R_eff" in fit:
        out["R_eff"] = int(fit["R_eff"])
    else:
        out["R_eff"] = compute_R_eff_from_energy(energy, rel_thresh)

    order = np.argsort(-energy)
    out["energy_order"] = order
    out["energy_top"] = energy[order[:min(10, Rh)]].tolist()
    out["R_true"] = int(R_true)
    return out


def to_jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.float32, np.float64)):
        return float(x)
    if isinstance(x, (np.int32, np.int64)):
        return int(x)
    return x


def main():
    ap = argparse.ArgumentParser("Evaluate fit vs truth for covariance simulations")
    ap.add_argument("--sim", type=str, required=True)
    ap.add_argument("--fit", type=str, required=True)
    ap.add_argument("--json_out", type=str, default="")
    ap.add_argument(
        "--align_procrustes",
        action="store_true",
        help="Align Phi_hat to Phi_true before lambda comparisons",
    )
    ap.add_argument(
        "--ard_rel_thresh",
        type=float,
        default=1e-2,
        help="Relative threshold for R_eff if not stored in fit",
    )
    args = ap.parse_args()

    sim = dict(np.load(args.sim, allow_pickle=True))
    fit = dict(np.load(args.fit, allow_pickle=True))

    Phi_true = sim["Phi_true"]
    lambda_true = sim["lambda_true"]
    omega = sim["omega"]
    W_eeg = sim["W_eeg"]
    W_fmri = sim["W_fmri"]
    m_fmri = sim["m_fmri"].astype(int)
    kmax_eeg = int(sim["kmax_eeg"])
    include_diag = bool(int(sim.get("include_diag", 1)))

    Phi_hat_full = fit["Phi_hat"]
    lambda_hat_full = fit["lambda_hat"]

    K, R_true = Phi_true.shape
    R_hat = Phi_hat_full.shape[1]

    out: Dict[str, Any] = {}

    is_ard = ("tau_hat" in fit) or ("R_eff" in fit) or (R_hat != R_true)
    out["is_ard"] = bool(is_ard)

    out.update(subspace_metrics(Phi_true, Phi_hat_full))

    out.update(
        region_errors_general(
            Phi_true,
            lambda_true,
            Phi_hat_full,
            lambda_hat_full,
            W_eeg,
            W_fmri,
            m_fmri,
            kmax_eeg,
            include_diag,
        )
    )

    if is_ard:
        out["ard"] = ard_summary(fit, R_true=R_true, rel_thresh_default=args.ard_rel_thresh)

        energy = np.asarray(out["ard"]["energy_r"], dtype=float)
        R_eff = int(out["ard"]["R_eff"])
        idx_active = select_top_factors_by_energy(energy, R_eff)

        out["ard_topR_eff_idx"] = idx_active.tolist()

        Phi_hat_active = Phi_hat_full[:, idx_active]
        lam_hat_active = lambda_hat_full[:, :, idx_active]

        met_act = subspace_metrics(Phi_true, Phi_hat_active)
        out["dsub_active"] = met_act["dsub"]
        out["angles_active_deg"] = met_act["angles_deg"]
        out["angle_active_mean_deg"] = met_act["angle_mean_deg"]
        out["angle_active_max_deg"] = met_act["angle_max_deg"]

    if (not is_ard) and (lambda_hat_full.shape[2] == R_true):
        Phi_hat = Phi_hat_full
        lam_hat = lambda_hat_full

        if args.align_procrustes:
            Phi_hat_al, Q = procrustes_align(Phi_true, Phi_hat)
            lam_hat_al, offn = rotate_lambda_diagonal(lam_hat, Q)
            out["procrustes_offdiag_mean"] = float(np.mean(offn))
            out["procrustes_offdiag_median"] = float(np.median(offn))
            out.update(lambda_metrics(lambda_true, lam_hat_al, omega))
        else:
            out.update(lambda_metrics(lambda_true, lam_hat, omega))
    else:
        if is_ard:
            energy = np.asarray(out["ard"]["energy_r"], dtype=float)
            idx_top_true = select_top_factors_by_energy(energy, R_true)
            out["ard_topRtrue_idx"] = idx_top_true.tolist()

            Phi_hat_sel = Phi_hat_full[:, idx_top_true]
            lam_hat_sel = lambda_hat_full[:, :, idx_top_true]

            if args.align_procrustes:
                Phi_hat_al, Q = procrustes_align(Phi_true, Phi_hat_sel)
                lam_hat_al, offn = rotate_lambda_diagonal(lam_hat_sel, Q)
                out["procrustes_offdiag_mean"] = float(np.mean(offn))
                out["procrustes_offdiag_median"] = float(np.median(offn))
                out.update(lambda_metrics(lambda_true, lam_hat_al, omega))
            else:
                out.update(lambda_metrics(lambda_true, lam_hat_sel, omega))

    print("\n=== Evaluation report ===")
    print("is_ard:", bool(out["is_ard"]))
    print("Subspace ||P*-P̂||_F (full):", out["dsub"])
    print("Principal angles (deg) (full):", np.array2string(np.asarray(out["angles_deg"]), precision=3))

    if is_ard:
        ard = out["ard"]
        print("ARD: Rmax=", ard["Rmax"], " R_eff=", ard["R_eff"], " (thr=", ard["R_eff_rel_thresh"], ")")
        print("ARD: tau_min/med/max =", ard["tau_min"], ard["tau_med"], ard["tau_max"])
        print("ARD: top energies =", ard["energy_top"])
        print("ARD: top-R_eff factor idx:", out["ard_topR_eff_idx"])
        print("Subspace ||P*-P̂||_F (top-R_eff):", out["dsub_active"])
        print(
            "Principal angles (deg) (top-R_eff):",
            np.array2string(np.asarray(out["angles_active_deg"]), precision=3),
        )

    if "procrustes_offdiag_mean" in out:
        print("Procrustes offdiag mean:", out["procrustes_offdiag_mean"])
    if "rmse_lambda" in out:
        print("RMSE(lambda):", out["rmse_lambda"])
        print("Mean corr over omega (per r):", np.array2string(np.asarray(out["corr_r_mean_over_omega"]), precision=3))
        print("Band corr (mean over r):", out["band_corr_mean_over_r"])

    print("RelErr observed mean:", out["RelErr_obs_mean"])
    print("RelErr EEG-miss mean:", out["RelErr_eeg_miss_mean"])
    print("RelErr fMRI-miss mean:", out["RelErr_fmri_miss_mean"])
    print("========================\n")

    if args.json_out:
        out2 = {}
        for k, v in out.items():
            if isinstance(v, dict):
                out2[k] = {kk: to_jsonable(vv) for kk, vv in v.items()}
            else:
                out2[k] = to_jsonable(v)
        with open(args.json_out, "w") as f:
            json.dump(out2, f, indent=2)
        print("Saved JSON:", args.json_out)


if __name__ == "__main__":
    main()