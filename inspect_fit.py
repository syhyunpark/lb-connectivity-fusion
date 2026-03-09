#!/usr/bin/env python3
# inspect_fit.py
#
# Inspect a fitted fusion model and optionally append baseline (single-modality) feature blocks
# from separate fit files.
# Example:
# python3 inspect_fit.py \
#   --data /.../realdata_fusion_K50_EEGK20_EO.npz \
#   --fit  /.../fit_real_EO_sep_R12_default.npz \
#   --outdir /.../inspect_real_EO_sep_R12_default \
#   --prefix LEMON_EO_R12 \
#   --formats png,pdf \
#   --sort_by_energy median \
#   --reff_mode max --reff_eps 0.01 \
#   --plot_title "MPI-LEMON (EO): fused EEG-fMRI networks (R=12)" \
#   --baseline_fmri_fit /.../fit_fmri_only_R12.npz \
#   --baseline_eeg_fit  /.../fit_eeg_only_R12.npz \
#   --write_sorted_fit
#!/usr/bin/env python3
# inspect_fit.py

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import scipy.linalg as la

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pandas as pd


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def load_npz(path: str) -> dict:
    return dict(np.load(path, allow_pickle=True))


def frob(A: np.ndarray) -> float:
    return float(np.sqrt(np.sum(A * A)))


def masked_relerr(A_true: np.ndarray, A_hat: np.ndarray, W: np.ndarray, offdiag_only: bool) -> float:
    Wm = W
    if offdiag_only:
        Wm = W.copy()
        np.fill_diagonal(Wm, 0.0)
    num = np.sum((Wm * (A_hat - A_true)) ** 2)
    den = np.sum((Wm * A_true) ** 2)
    return float(np.sqrt(num / max(den, 1e-12)))


def sigma_from_phi_lambda(Phi: np.ndarray, lam_r: np.ndarray) -> np.ndarray:
    return (Phi * lam_r[None, :]) @ Phi.T


def fmri_rankR_baseline(C_fmri: np.ndarray, R: int) -> np.ndarray:
    Cbar = np.mean(C_fmri, axis=0)
    Cbar = 0.5 * (Cbar + Cbar.T)
    evals, evecs = la.eigh(Cbar)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]
    evals_pos = np.clip(evals[:R], 0.0, None)
    Ur = evecs[:, :R]
    Cbar_R = (Ur * evals_pos[None, :]) @ Ur.T
    return 0.5 * (Cbar_R + Cbar_R.T)


def _normalized_wm(data: dict) -> tuple[np.ndarray, np.ndarray]:
    m_fmri = np.asarray(data.get("m_fmri", []), dtype=int)
    w_m = np.asarray(data.get("w_m", []), dtype=float)
    if m_fmri.size == 0 or w_m.size == 0:
        return m_fmri, w_m
    ws = float(np.sum(w_m[m_fmri]))
    if ws > 0:
        w_m = w_m / ws
    return m_fmri, w_m


def compute_fmri_pred_from_fit(data: dict, fit: dict, Phi: np.ndarray, lambdas_eeg: np.ndarray) -> np.ndarray:
    """
    Return subject-level fMRI predictions with shape (n, K, K).

    If lambda_fmri_hat is present, use it directly.
    Otherwise aggregate the EEG-side weights over the fMRI bins in data.
    """
    C_fmri = data["C_fmri"]
    n, _, _ = C_fmri.shape

    if "lambda_fmri_hat" in fit:
        lam_fmri = np.asarray(fit["lambda_fmri_hat"], dtype=float)
        if lam_fmri.shape[0] != n:
            raise ValueError("lambda_fmri_hat has wrong n dimension.")
        return np.stack([sigma_from_phi_lambda(Phi, lam_fmri[i]) for i in range(n)], axis=0)

    m_fmri, w_m = _normalized_wm(data)
    if m_fmri.size == 0:
        raise ValueError("aggregate fMRI mode requires m_fmri and w_m in data.")
    lam_agg = np.sum(lambdas_eeg[:, m_fmri, :] * w_m[m_fmri][None, :, None], axis=1)
    return np.stack([sigma_from_phi_lambda(Phi, lam_agg[i]) for i in range(n)], axis=0)


def compute_eeg_pred_from_fit(data: dict, Phi: np.ndarray, lambdas: np.ndarray) -> np.ndarray:
    n, F, _, _ = data["C_eeg"].shape
    pred = np.zeros((n, F, Phi.shape[0], Phi.shape[0]), dtype=float)
    for i in range(n):
        for f in range(F):
            pred[i, f] = sigma_from_phi_lambda(Phi, lambdas[i, f])
    return pred


def _band_indices(omega: np.ndarray) -> Dict[str, np.ndarray]:
    """Canonical frequency bands on the model grid."""
    omega = np.asarray(omega, dtype=float).reshape(-1)
    wmax = float(np.nanmax(omega)) if omega.size else 0.0
    gamma_hi = min(45.0, wmax + 1e-9)

    bands = {
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "beta": (13.0, 30.0),
        "gamma": (30.0, gamma_hi),
    }
    out: Dict[str, np.ndarray] = {}
    for name, (lo, hi) in bands.items():
        out[name] = np.where((omega >= lo) & (omega < hi))[0]
    return out


def _spec_center_of_mass(omega: np.ndarray, lam_f: np.ndarray) -> np.ndarray:
    """Spectral center of mass for one factor across frequency."""
    denom = np.sum(lam_f, axis=1) + 1e-12
    return (lam_f @ omega) / denom


def plot_energy_bar(energy: np.ndarray, outbase: Path, title: str, formats=("png",)):
    x = np.arange(1, len(energy) + 1)
    plt.figure(figsize=(6, 3))
    plt.bar(x, energy)
    plt.xlabel("Network index (sorted)")
    plt.ylabel("Median energy")
    if title:
        plt.title(title)
    plt.tight_layout()
    for ext in formats:
        plt.savefig(str(outbase.with_suffix(f".{ext}")), dpi=220)
    plt.close()


def plot_energy_stacked(energy_eeg: np.ndarray, energy_fmri: np.ndarray, outbase: Path, title: str, formats=("png",)):
    x = np.arange(1, len(energy_eeg) + 1)
    plt.figure(figsize=(6.6, 3.2))
    plt.bar(x, energy_eeg, label="EEG")
    plt.bar(x, energy_fmri, bottom=energy_eeg, label="fMRI")
    plt.xlabel("Network index (sorted)")
    plt.ylabel("Median energy (stacked)")
    if title:
        plt.title(title)
    plt.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    for ext in formats:
        plt.savefig(str(outbase.with_suffix(f".{ext}")), dpi=220)
    plt.close()


def plot_lambda_median_curves(omega: np.ndarray, lam: np.ndarray, outbase: Path, title: str, formats=("png",)):
    med = np.median(lam, axis=0)
    plt.figure(figsize=(7.2, 4.0))
    for r in range(med.shape[1]):
        plt.plot(omega, med[:, r], label=f"r={r+1}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel(r"Median $\lambda_r(\omega)$")
    if title:
        plt.title(title)
    plt.legend(ncol=2, fontsize=8, frameon=False)
    plt.tight_layout()
    for ext in formats:
        plt.savefig(str(outbase.with_suffix(f".{ext}")), dpi=220)
    plt.close()


def plot_hist(x: np.ndarray, outbase: Path, title: str, xlabel: str, formats=("png",)):
    plt.figure(figsize=(6, 3))
    plt.hist(x, bins=30)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    if title:
        plt.title(title)
    plt.tight_layout()
    for ext in formats:
        plt.savefig(str(outbase.with_suffix(f".{ext}")), dpi=220)
    plt.close()


def extract_feature_block_from_fit(
    data: dict,
    fit_path: str,
    prefix: str,
    kind: str,
    sort_by_energy: str = "median",
) -> pd.DataFrame:
    """
    Extract a baseline-only feature block aligned to the subject order in data.

    kind="fmri" -> {prefix}__fMRI_strength_r*
    kind="eeg"  -> {prefix}__EEG_theta_r*, ..., {prefix}__EEG_specCOM_r*
    """
    if kind not in ("fmri", "eeg"):
        raise ValueError("kind must be one of {'fmri','eeg'}")

    fit = load_npz(fit_path)

    Phi = np.asarray(fit["Phi_hat"], dtype=float)
    lambdas = np.asarray(fit["lambda_hat"], dtype=float)
    R = Phi.shape[1]
    n = lambdas.shape[0]

    omega = np.asarray(data.get("omega", np.arange(lambdas.shape[1])), dtype=float).reshape(-1)
    band_idx = _band_indices(omega)

    energy_eeg_med = np.median(lambdas * lambdas, axis=(0, 1))
    if "lambda_fmri_hat" in fit:
        lam_fmri = np.asarray(fit["lambda_fmri_hat"], dtype=float)
        energy_fmri_med = np.median(lam_fmri * lam_fmri, axis=0)
    else:
        energy_fmri_med = np.zeros(R, dtype=float)
    energy_total_med = energy_eeg_med + energy_fmri_med

    order = np.arange(R)
    if sort_by_energy == "median":
        order = np.argsort(-energy_total_med)
    elif sort_by_energy == "mean":
        energy_eeg_mean = np.mean(lambdas * lambdas, axis=(0, 1))
        if "lambda_fmri_hat" in fit:
            energy_fmri_mean = np.mean(lam_fmri * lam_fmri, axis=0)
        else:
            energy_fmri_mean = np.zeros(R, dtype=float)
        order = np.argsort(-(energy_eeg_mean + energy_fmri_mean))

    lambdas_s = lambdas[:, :, order]

    subj = data.get("subject_ids", np.arange(n)).astype(str)
    out = {"subject_id": subj}

    if "lambda_fmri_hat" in fit:
        fmri_strength = np.asarray(fit["lambda_fmri_hat"], dtype=float)[:, order]
    else:
        m_fmri, w_m = _normalized_wm(data)
        if m_fmri.size == 0:
            fmri_strength = np.full((n, R), np.nan)
        else:
            fmri_strength = np.sum(lambdas_s[:, m_fmri, :] * w_m[m_fmri][None, :, None], axis=1)

    if kind == "fmri":
        for r in range(R):
            out[f"{prefix}__fMRI_strength_r{r+1}"] = fmri_strength[:, r]
        return pd.DataFrame(out)

    for r in range(R):
        for name, idx in band_idx.items():
            if idx.size:
                out[f"{prefix}__EEG_{name}_r{r+1}"] = np.sum(lambdas_s[:, idx, r], axis=1)
            else:
                out[f"{prefix}__EEG_{name}_r{r+1}"] = np.nan
        out[f"{prefix}__EEG_specCOM_r{r+1}"] = _spec_center_of_mass(omega, lambdas_s[:, :, r])

    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser("Inspect fitted fusion model on real/sim data (+ optional baseline feature blocks).")
    ap.add_argument("--data", required=True)
    ap.add_argument("--fit", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--prefix", default="fit")
    ap.add_argument("--formats", default="png", help="comma list, e.g. png,pdf")
    ap.add_argument("--sort_by_energy", choices=["none", "median", "mean"], default="none")

    ap.add_argument("--reff_mode", choices=["max", "share"], default="max")
    ap.add_argument("--reff_eps", type=float, default=0.01)

    ap.add_argument("--plot_title", default="")
    ap.add_argument("--write_sorted_fit", action="store_true")

    ap.add_argument("--baseline_fmri_fit", default="", help="Optional: path to fMRI-only fit .npz")
    ap.add_argument("--baseline_eeg_fit", default="", help="Optional: path to EEG-only fit .npz")

    args = ap.parse_args()

    outdir = Path(args.outdir).expanduser().resolve()
    ensure_dir(outdir)
    fmts = tuple([x.strip() for x in args.formats.split(",") if x.strip()])

    data = load_npz(args.data)
    fit = load_npz(args.fit)

    C_eeg = data["C_eeg"].astype(float)
    C_fmri = data["C_fmri"].astype(float)
    W_eeg = data["W_eeg"].astype(float)
    W_fmri = data["W_fmri"].astype(float)

    n, F, K, _ = C_eeg.shape

    omega = np.asarray(data.get("omega", np.arange(F, dtype=float)), dtype=float).reshape(-1)
    if omega.size != F:
        omega = np.linspace(1.0, float(F), F)

    band_idx = _band_indices(omega)

    Phi = np.asarray(fit["Phi_hat"], dtype=float)
    lambdas = np.asarray(fit["lambda_hat"], dtype=float)
    R = Phi.shape[1]

    fmri_mode = str(fit.get("fmri_mode", "aggregate"))
    has_sep_fmri = ("lambda_fmri_hat" in fit)

    ortho_err = frob(Phi.T @ Phi - np.eye(R))

    energy_eeg_med0 = np.median(lambdas * lambdas, axis=(0, 1))
    if has_sep_fmri:
        lam_fmri0 = np.asarray(fit["lambda_fmri_hat"], dtype=float)
        energy_fmri_med0 = np.median(lam_fmri0 * lam_fmri0, axis=0)
    else:
        energy_fmri_med0 = np.zeros(R, dtype=float)
    energy_total_med0 = energy_eeg_med0 + energy_fmri_med0

    order = np.arange(R)
    if args.sort_by_energy == "median":
        order = np.argsort(-energy_total_med0)
    elif args.sort_by_energy == "mean":
        energy_eeg_mean0 = np.mean(lambdas * lambdas, axis=(0, 1))
        if has_sep_fmri:
            energy_fmri_mean0 = np.mean(lam_fmri0 * lam_fmri0, axis=0)
        else:
            energy_fmri_mean0 = np.zeros(R, dtype=float)
        order = np.argsort(-(energy_eeg_mean0 + energy_fmri_mean0))

    Phi_s = Phi[:, order]
    lambdas_s = lambdas[:, :, order]

    energy_eeg_med = np.median(lambdas_s * lambdas_s, axis=(0, 1))
    if has_sep_fmri:
        lam_fmri = np.asarray(fit["lambda_fmri_hat"], dtype=float)[:, order]
        energy_fmri_med = np.median(lam_fmri * lam_fmri, axis=0)
    else:
        lam_fmri = None
        energy_fmri_med = np.zeros(R, dtype=float)
    energy_total_med = energy_eeg_med + energy_fmri_med

    if args.reff_mode == "max":
        denom = float(np.max(energy_total_med)) if energy_total_med.size else 0.0
        share = energy_total_med / (denom + 1e-12) if denom > 0 else np.zeros_like(energy_total_med)
    else:
        denom = float(np.sum(energy_total_med)) if energy_total_med.size else 0.0
        share = energy_total_med / (denom + 1e-12) if denom > 0 else np.zeros_like(energy_total_med)
    R_eff = int(np.sum(share >= args.reff_eps))

    if has_sep_fmri:
        fit_view = dict(fit)
        fit_view["lambda_fmri_hat"] = lam_fmri
        C_fmri_hat = compute_fmri_pred_from_fit(data, fit_view, Phi_s, lambdas_s)
    else:
        C_fmri_hat = compute_fmri_pred_from_fit(data, fit, Phi_s, lambdas_s)
    C_eeg_hat = compute_eeg_pred_from_fit(data, Phi_s, lambdas_s)

    fmri_rel = np.array([masked_relerr(C_fmri[i], C_fmri_hat[i], W_fmri, offdiag_only=False) for i in range(n)])
    fmri_rel_off = np.array([masked_relerr(C_fmri[i], C_fmri_hat[i], W_fmri, offdiag_only=True) for i in range(n)])

    eeg_rel = np.zeros(n, dtype=float)
    eeg_rel_off = np.zeros(n, dtype=float)
    for i in range(n):
        rr = []
        rr_off = []
        for f in range(F):
            rr.append(masked_relerr(C_eeg[i, f], C_eeg_hat[i, f], W_eeg, offdiag_only=False))
            rr_off.append(masked_relerr(C_eeg[i, f], C_eeg_hat[i, f], W_eeg, offdiag_only=True))
        eeg_rel[i] = float(np.mean(rr))
        eeg_rel_off[i] = float(np.mean(rr_off))

    Cbar_fmri = 0.5 * (np.mean(C_fmri, axis=0) + np.mean(C_fmri, axis=0).T)
    fmri_base = np.array([masked_relerr(C_fmri[i], Cbar_fmri, W_fmri, offdiag_only=False) for i in range(n)])
    fmri_base_off = np.array([masked_relerr(C_fmri[i], Cbar_fmri, W_fmri, offdiag_only=True) for i in range(n)])

    Cbar_eeg = np.mean(C_eeg, axis=0)
    eeg_base = np.zeros(n, dtype=float)
    eeg_base_off = np.zeros(n, dtype=float)
    for i in range(n):
        rr = []
        rr_off = []
        for f in range(F):
            rr.append(masked_relerr(C_eeg[i, f], Cbar_eeg[f], W_eeg, offdiag_only=False))
            rr_off.append(masked_relerr(C_eeg[i, f], Cbar_eeg[f], W_eeg, offdiag_only=True))
        eeg_base[i] = float(np.mean(rr))
        eeg_base_off[i] = float(np.mean(rr_off))

    CbarR = fmri_rankR_baseline(C_fmri, R=R)
    fmri_rankR = np.array([masked_relerr(C_fmri[i], CbarR, W_fmri, offdiag_only=False) for i in range(n)])
    fmri_rankR_off = np.array([masked_relerr(C_fmri[i], CbarR, W_fmri, offdiag_only=True) for i in range(n)])

    subj = data.get("subject_ids", np.arange(n)).astype(str)

    if has_sep_fmri:
        fmri_strength = lam_fmri
    else:
        m_fmri, w_m = _normalized_wm(data)
        fmri_strength = np.sum(lambdas_s[:, m_fmri, :] * w_m[m_fmri][None, :, None], axis=1)

    feat: Dict[str, np.ndarray] = {"subject_id": subj}

    for r in range(R):
        feat[f"fMRI_strength_r{r+1}"] = fmri_strength[:, r]
        for name, idx in band_idx.items():
            feat[f"EEG_{name}_r{r+1}"] = np.sum(lambdas_s[:, idx, r], axis=1) if idx.size else np.nan
        feat[f"EEG_specCOM_r{r+1}"] = _spec_center_of_mass(omega, lambdas_s[:, :, r])

    feat["fmri_relerr"] = fmri_rel
    feat["fmri_relerr_offdiag"] = fmri_rel_off
    feat["eeg_relerr_obs"] = eeg_rel
    feat["eeg_relerr_obs_offdiag"] = eeg_rel_off

    feat["fmri_baseline_relerr"] = fmri_base
    feat["fmri_baseline_relerr_offdiag"] = fmri_base_off
    feat["fmri_rankR_baseline_relerr"] = fmri_rankR
    feat["fmri_rankR_baseline_relerr_offdiag"] = fmri_rankR_off
    feat["eeg_baseline_relerr"] = eeg_base
    feat["eeg_baseline_relerr_offdiag"] = eeg_base_off

    df_feat = pd.DataFrame(feat)

    if args.baseline_fmri_fit.strip():
        df_b = extract_feature_block_from_fit(
            data=data,
            fit_path=args.baseline_fmri_fit.strip(),
            prefix="FMRIonly",
            kind="fmri",
            sort_by_energy="median",
        )
        df_feat = df_feat.merge(df_b, on="subject_id", how="left")

    if args.baseline_eeg_fit.strip():
        df_b = extract_feature_block_from_fit(
            data=data,
            fit_path=args.baseline_eeg_fit.strip(),
            prefix="EEGonly",
            kind="eeg",
            sort_by_energy="median",
        )
        df_feat = df_feat.merge(df_b, on="subject_id", how="left")

    feat_csv = outdir / f"{args.prefix}_subject_features.csv"
    df_feat.to_csv(feat_csv, index=False)

    summary = {
        "n": int(n),
        "F": int(F),
        "K": int(K),
        "R": int(R),
        "fmri_mode": fmri_mode,
        "has_lambda_fmri_hat": bool(has_sep_fmri),
        "ortho_err_fro": float(ortho_err),
        "sort_by_energy": args.sort_by_energy,
        "sort_order": order.tolist(),
        "reff_mode": args.reff_mode,
        "reff_eps": float(args.reff_eps),
        "R_eff": int(R_eff),
        "energy_eeg_median": energy_eeg_med.tolist(),
        "energy_fmri_median": energy_fmri_med.tolist(),
        "energy_total_median": energy_total_med.tolist(),
        "energy_total_max": float(np.max(energy_total_med)) if energy_total_med.size else float("nan"),
        "energy_total_sum": float(np.sum(energy_total_med)) if energy_total_med.size else float("nan"),
        "fmri_relerr_median": float(np.median(fmri_rel)),
        "fmri_relerr_offdiag_median": float(np.median(fmri_rel_off)),
        "eeg_relerr_obs_median": float(np.median(eeg_rel)),
        "eeg_relerr_obs_offdiag_median": float(np.median(eeg_rel_off)),
        "fmri_baseline_relerr_median": float(np.median(fmri_base)),
        "fmri_rankR_baseline_relerr_median": float(np.median(fmri_rankR)),
        "eeg_baseline_relerr_median": float(np.median(eeg_base)),
        "scale_eeg": float(data.get("scale_eeg", np.nan)),
        "scale_fmri": float(data.get("scale_fmri", np.nan)),
        "eeg_cond": str(data.get("eeg_cond", "")),
        "feature_naming": "paper_friendly_v2 (+gamma band mass)",
        "baseline_fmri_fit": args.baseline_fmri_fit,
        "baseline_eeg_fit": args.baseline_eeg_fit,
    }
    summary_path = outdir / f"{args.prefix}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    title = args.plot_title.strip() or args.prefix

    plot_energy_bar(
        energy_total_med,
        outdir / f"{args.prefix}_energy_total",
        title=f"{title}: network energy (total)",
        formats=fmts,
    )
    plot_energy_stacked(
        energy_eeg_med,
        energy_fmri_med,
        outdir / f"{args.prefix}_energy_stacked",
        title=f"{title}: network energy (EEG + fMRI)",
        formats=fmts,
    )
    plot_energy_bar(
        energy_eeg_med,
        outdir / f"{args.prefix}_energy_eeg",
        title=f"{title}: network energy (EEG)",
        formats=fmts,
    )
    plot_energy_bar(
        energy_fmri_med,
        outdir / f"{args.prefix}_energy_fmri",
        title=f"{title}: network energy (fMRI)",
        formats=fmts,
    )

    plot_lambda_median_curves(
        omega,
        lambdas_s,
        outdir / f"{args.prefix}_lambda_median_curves",
        title=f"{title}: median spectral factor weights",
        formats=fmts,
    )
    plot_hist(
        fmri_rel,
        outdir / f"{args.prefix}_fmri_relerr_hist",
        title=f"{title}: fMRI relative error",
        xlabel="relative error",
        formats=fmts,
    )
    plot_hist(
        eeg_rel,
        outdir / f"{args.prefix}_eeg_relerr_obs_hist",
        title=f"{title}: EEG relative error (observed block)",
        xlabel="relative error",
        formats=fmts,
    )

    if args.write_sorted_fit:
        sorted_fit = outdir / f"{args.prefix}_fit_sorted.npz"
        out = dict(fit)
        out["Phi_hat"] = Phi_s
        out["lambda_hat"] = lambdas_s
        if has_sep_fmri:
            out["lambda_fmri_hat"] = lam_fmri
        np.savez_compressed(sorted_fit, **out)

    print("Wrote:", feat_csv)
    print("Wrote:", summary_path)
    print("Quick checks:")
    print("  fmri_mode =", fmri_mode, " separate_lambda =", has_sep_fmri)
    print("  ||Phi^T Phi - I||_F =", ortho_err)
    print(f"  R_eff ({args.reff_mode}, eps={args.reff_eps}):", R_eff)
    print("  fMRI relerr median =", float(np.median(fmri_rel)), " baseline =", float(np.median(fmri_base)))
    print("  EEG  relerr median =", float(np.median(eeg_rel)), " baseline =", float(np.median(eeg_base)))


if __name__ == "__main__":
    main()