#!/usr/bin/env python3
"""Shared-spatial-factor estimator for EEG--fMRI connectivity.

This code is the estimation engine. 

The model represents modality-specific connectivity summaries in common
Laplace--Beltrami (LB) coordinates using shared orthonormal spatial factors
Phi and separate nonnegative modality-specific strengths. EEG strengths are
frequency resolved; fMRI strengths are modality specific.

The same implementation can also
be used by the simulation scripts, including adaptive ARD-motivated rank
screening.
"""
import argparse
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Literal, List

import numpy as np
import scipy.linalg as la
from scipy.optimize import lsq_linear

try:
    from joblib import Parallel, delayed
    _HAS_JOBLIB = True
except Exception:
    _HAS_JOBLIB = False

 
# Helpers 
def sym(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.T)


def project_tangent(Phi: np.ndarray, G: np.ndarray) -> np.ndarray:
    # Riemannian gradient on Stiefel: G - Phi sym(Phi^T G)
    return G - Phi @ sym(Phi.T @ G)


def qr_retraction(Phi: np.ndarray) -> np.ndarray:
    Q, _ = la.qr(Phi, mode="economic")
    return Q


def sigma_from_Phi_lambda(Phi: np.ndarray, lam: np.ndarray) -> np.ndarray:
    # Phi diag(lam) Phi^T
    return (Phi * lam[None, :]) @ Phi.T


def _frob(A: np.ndarray) -> float:
    return float(np.sqrt(np.sum(A * A)))

 
# Small caches: (F,R) -> (D, Ksmooth)
# ---- 
_CACHE_DR: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}


def get_D_Ksmooth(F: int, R: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return the second-difference operator used for EEG spectral smoothing.

    For F >= 3, D has shape (F-2, F) and computes second differences across
    frequency bins. For F < 3, no second difference is defined, so we return
    empty operators. 
    """
    key = (F, R)
    if key in _CACHE_DR:
        return _CACHE_DR[key]
    if F < 1:
        raise ValueError("Need F>=1")
    if F < 3:
        D = np.zeros((0, F), dtype=float)
        Ksmooth = np.zeros((0, F * R), dtype=float)
    else:
        D = np.zeros((F - 2, F), dtype=float)
        for i in range(F - 2):
            D[i, i] = 1.0
            D[i, i + 1] = -2.0
            D[i, i + 2] = 1.0
        Ksmooth = np.kron(D, np.eye(R))  # ((F-2)R, FR), frequency-major
    _CACHE_DR[key] = (D, Ksmooth)
    return D, Ksmooth


 
# Config 
FMriMode = Literal["aggregate", "separate"]
SortMode = Literal["none", "median", "mean"]
InitExpand = Literal["random", "fmri_eig"]


@dataclass
class FitConfig:
    R: int = 5

    # penalties
    alpha_lambda: float = 0.0
    alpha0: float = 1e-6

    # modality weights (allow 0 for single-modality fits)
    w_eeg: float = -1.0   # default 1/F
    w_fmri: float = 1.0

    # fMRI observation mode
    fmri_mode: FMriMode = "aggregate"   # DEFAULT keeps simulation identical

    # Phi update
    step_phi: float = 1e-2
    max_iter: int = 20
    tol: float = 1e-6

    backtrack: bool = True
    backtrack_factor: float = 0.5
    backtrack_max: int = 12
    enforce_monotone: bool = True

    # lambda solver
    lam_solver: str = "bvls"   # "lsq" or "bvls"

    # parallel lambda solves
    n_jobs: int = 1

    # ----- ARD options  
    use_ard: bool = False
    Rmax: int = 15
    ard_a: float = 1e-6
    ard_b: float = 1e-6
    ard_eta: float = 0.2
    ard_burnin: int = 2
    ard_tau_floor: float = 1e-6
    ard_tau_ceiling: float = 1e9
    ard_energy_rel_thresh: float = 1e-2

    #  end-of-fit sorting 
    sort_factors: SortMode = "none"  # default none => simulation outputs unchanged

    # init Phi from an anchor fit  
    init_fit: str = ""              # path to .npz with Phi_hat
    init_expand: InitExpand = "random"
    init_seed: int = 0
    init_method: str = "fmri_eig"
    use_clean_data: bool = False

    verbose: bool = True

 
# Edge compression 
def get_edge_indices(W: np.ndarray, include_diag: bool) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    K = W.shape[0]
    iu = np.triu_indices(K, k=0 if include_diag else 1)
    keep = np.where(W[iu] > 0)[0]
    if keep.size == 0:
        raise ValueError("Mask has no observed entries.")
    return iu, keep


def build_A_columns(Phi: np.ndarray, W: np.ndarray, iu, keep: np.ndarray) -> np.ndarray:
    """Compressed symmetric design equivalent to the full Frobenius norm."""
    _, R = Phi.shape
    frob_w = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
    cols = []
    for r in range(R):
        Br_u = np.outer(Phi[:, r], Phi[:, r])[iu]
        cols.append((frob_w * W[iu] * Br_u)[keep])
    return np.stack(cols, axis=1)

 
# NNLS solver 
def solve_nnls(A_aug: np.ndarray, y_aug: np.ndarray, solver: str) -> np.ndarray:
    if solver == "bvls":
        res = lsq_linear(A_aug, y_aug, bounds=(0.0, np.inf), method="bvls", verbose=0)
    else:
        res = lsq_linear(A_aug, y_aug, bounds=(0.0, np.inf), lsmr_tol="auto", verbose=0)
    return res.x


def build_ridge_ard_diag(tau: np.ndarray, F: int, add_fmri_bin: bool) -> np.ndarray:
    """
    ARD factor-wise ridge for variable layout:
      aggregate: x = vec(lambda_eeg) length FR
      separate : x = [vec(lambda_eeg) length FR, lambda_fmri length R] => length FR+R
    """
    tau = np.asarray(tau, dtype=float)
    ridge_w = np.tile(np.sqrt(tau), F)  # length FR
    if add_fmri_bin:
        ridge_w = np.concatenate([ridge_w, np.sqrt(tau)], axis=0)  # length FR+R
    return np.diag(ridge_w)

 
# Lambda update (aggregate|separate) 
def lambda_update_all_subjects(
    C_eeg: np.ndarray,
    C_fmri: np.ndarray,
    Phi: np.ndarray,
    W_eeg: np.ndarray,
    W_fmri: np.ndarray,
    F_fmri: np.ndarray,
    w_f: np.ndarray,
    sigma_eeg: float,
    sigma_fmri: float,
    cfg: FitConfig,
    include_diag: bool,
    tau: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Returns:
      lambdas_eeg:  (n, F, R)
      lambdas_fmri: (n, R) if fmri_mode='separate' else None
    """
    n, F, K, _ = C_eeg.shape
    R = Phi.shape[1]

    w_eeg = (1.0 / F) if cfg.w_eeg < 0 else float(cfg.w_eeg)
    w_fmri = float(cfg.w_fmri)

    use_eeg = (w_eeg > 0.0)
    use_fmri = (w_fmri > 0.0)

    seeg = (np.sqrt(w_eeg) / sigma_eeg) if use_eeg else 0.0
    sfmri = (np.sqrt(w_fmri) / sigma_fmri) if use_fmri else 0.0

    separate = (cfg.fmri_mode == "separate") and use_fmri
    P = F * R + (R if separate else 0)

    iu, eeg_keep = get_edge_indices(W_eeg, include_diag)
    _, fmri_keep = get_edge_indices(W_fmri, include_diag)
    iu0, iu1 = iu

    frob_u = np.where(iu0 == iu1, 1.0, np.sqrt(2.0))
    Weeg_u_keep = (frob_u * W_eeg[iu])[eeg_keep]
    Wfmri_u_keep = (frob_u * W_fmri[iu])[fmri_keep]

    Eeeg = eeg_keep.size
    Efmri = fmri_keep.size

    # smoothness penalty only for EEG part; pad zeros if separate
    _, Ksmooth = get_D_Ksmooth(F, R)  # ((F-2)R, FR)
    if separate:
        Ksmooth_full = np.hstack([Ksmooth, np.zeros((Ksmooth.shape[0], R), dtype=float)])
    else:
        Ksmooth_full = Ksmooth
    A_smooth = np.sqrt(cfg.alpha_lambda) * Ksmooth_full

    # ridge
    if cfg.use_ard:
        if tau is None or tau.shape != (R,):
            raise ValueError("ARD enabled but tau not provided or wrong shape.")
        A_ridge = build_ridge_ard_diag(tau, F, add_fmri_bin=separate)
    else:
        A_ridge = np.sqrt(cfg.alpha0) * np.eye(P, dtype=float)

    A_blocks = []
    y_layout = []  # list of tuples (name, length)

    if use_eeg:
        A_eeg = build_A_columns(Phi, W_eeg, iu, eeg_keep)  # (Eeeg,R)
        A_eeg_block = np.zeros((F * Eeeg, P), dtype=float)
        for f in range(F):
            A_eeg_block[f * Eeeg:(f + 1) * Eeeg, f * R:(f + 1) * R] = A_eeg
        A_blocks.append(seeg * A_eeg_block)
        y_layout.append(("eeg", F * Eeeg))

    if use_fmri:
        A_fmri = build_A_columns(Phi, W_fmri, iu, fmri_keep)  # (Efmri,R)
        A_fmri_block = np.zeros((Efmri, P), dtype=float)
        if separate:
            A_fmri_block[:, F * R:F * R + R] = A_fmri
        else:
            for f in F_fmri:
                A_fmri_block[:, f * R:(f + 1) * R] = w_f[f] * A_fmri
        A_blocks.append(sfmri * A_fmri_block)
        y_layout.append(("fmri", Efmri))

    if not A_blocks:
        return np.zeros((n, F, R), dtype=float), (np.zeros((n, R), dtype=float) if separate else None)

    A_data = np.vstack(A_blocks)
    A_aug = np.vstack([A_data, A_smooth, A_ridge])
    y_aug_len = A_aug.shape[0]

    def build_y_aug(i: int) -> np.ndarray:
        y = np.zeros(y_aug_len, dtype=float)
        pos = 0
        for name, length in y_layout:
            if name == "eeg":
                Ce_u_all = C_eeg[i][:, iu0, iu1]
                Ce_u_all = Ce_u_all[:, eeg_keep]
                y_eeg = (Ce_u_all * Weeg_u_keep[None, :]).ravel(order="C")
                y[pos:pos + length] = seeg * y_eeg
                pos += length
            elif name == "fmri":
                Cf_u = C_fmri[i][iu0, iu1]
                Cf_u = Cf_u[fmri_keep]
                y_fm = Wfmri_u_keep * Cf_u
                y[pos:pos + length] =  sfmri * y_fm
                pos +=  length
        return y

    def solve_one(i: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        x = solve_nnls(A_aug, build_y_aug(i), solver=cfg.lam_solver)  # length P
        lam_eeg = x[:F * R].reshape(F, R)
        lam_fm = x[F * R:F * R + R] if separate else None
        return lam_eeg, lam_fm

    if cfg.n_jobs > 1 and _HAS_JOBLIB:
        out_list = Parallel(n_jobs=cfg.n_jobs, prefer="threads")(delayed(solve_one)(i) for i in range(n))
    else:
        out_list = [solve_one(i) for i in range(n)]

    lambdas_eeg = np.stack([t[0] for t in out_list], axis=0)
    lambdas_fmri = np.stack([t[1] for t in out_list], axis=0) if separate else None
    return lambdas_eeg, lambdas_fmri

 
# Lambda penalties 
def lambda_penalties(
    lambdas_eeg: np.ndarray,
    alpha_lambda: float,
    alpha0: float,
    use_ard: bool,
    tau: Optional[np.ndarray],
    lambdas_fmri: Optional[np.ndarray],
) -> float:
    n, F, R = lambdas_eeg.shape
    pen = 0.0

    if alpha_lambda > 0:
        D, _ = get_D_Ksmooth(F, R)
        for i in range(n):
            for r in range(R):
                dv = D @ lambdas_eeg[i, :, r]
                pen += alpha_lambda * float(dv @ dv)

    if use_ard:
        if tau is None:
            raise ValueError("use_ard=True but tau is None.")
        energy_r = np.sum(lambdas_eeg * lambdas_eeg, axis=(0, 1))
        if lambdas_fmri is not None:
            energy_r = energy_r + np.sum(lambdas_fmri * lambdas_fmri, axis=0)
        pen += float(np.sum(tau * energy_r))
    else:
        if alpha0 > 0:
            pen += alpha0 * float(np.sum(lambdas_eeg * lambdas_eeg))
            if lambdas_fmri is not None:
                pen += alpha0 * float(np.sum(lambdas_fmri * lambdas_fmri))

    return float(pen)

 
# Data-fit + gradient 
def datafit_and_grad(
    C_eeg, C_fmri, Phi, lambdas_eeg, lambdas_fmri, W_eeg, W_fmri, F_fmri, w_f,
    sigma_eeg, sigma_fmri, w_eeg, w_fmri, fmri_mode: FMriMode
) -> Tuple[float, np.ndarray]:
    n, F, K, _ =  C_eeg.shape
    Rdim = Phi.shape[1]

    use_eeg = (w_eeg > 0.0)
    use_fmri = (w_fmri > 0.0)

    J = 0.0
    G = np.zeros((K, Rdim), dtype=float)

    for i in range(n):
        if use_eeg:
            for f in range(F):
                lam = lambdas_eeg[i, f]
                S = sigma_from_Phi_lambda(Phi, lam)
                resid = S - C_eeg[i, f]
                Rm = W_eeg * resid
                J += w_eeg * (1.0 / (2.0 * sigma_eeg**2)) * np.sum(Rm * Rm)
                # The objective uses ||W ⊙ resid||_F^2, so its derivative contains W^2.
                tmp = ((W_eeg * W_eeg) * resid) @ Phi
                G += w_eeg * (2.0 / (sigma_eeg**2)) * (tmp * lam[None, :])

        if use_fmri:
            if fmri_mode == "separate":
                if lambdas_fmri is None:
                    raise RuntimeError("fmri_mode='separate' but lambdas_fmri is None.")
                lam_fm = lambdas_fmri[i]
            else:
                lam_fm = np.zeros(Rdim, dtype=float)
                for ff in F_fmri:
                    lam_fm += w_f[ff] * lambdas_eeg[i, ff]

            Sf = sigma_from_Phi_lambda(Phi, lam_fm)
            resid_f = Sf - C_fmri[i]
            Rf = W_fmri * resid_f
            J += w_fmri * (1.0 / (2.0 * sigma_fmri**2)) * np.sum(Rf * Rf)
            tmpf = ((W_fmri * W_fmri) * resid_f) @ Phi
            G += w_fmri * (2.0 / (sigma_fmri**2)) * (tmpf * lam_fm[None, :])

    return float(J), G

 
# ARD tau update (damped) 
def ard_update_tau(
    lambdas_eeg: np.ndarray,
    lambdas_fmri: Optional[np.ndarray],
    tau: np.ndarray,
    a: float,
    b: float,
    eta: float,
    tau_floor: float,
    tau_ceiling: float,
) -> np.ndarray:
    n, F, R = lambdas_eeg.shape

    # sum-of-squares per factor (matches penalty definition)
    ss_r = np.sum(lambdas_eeg * lambdas_eeg, axis=(0, 1))
    n_coef = n * F
    if lambdas_fmri is not None:
        ss_r = ss_r + np.sum(lambdas_fmri * lambdas_fmri, axis=0)
        n_coef = n_coef + n  # extra n coefficients per factor in separate mode

    raw = (a + n_coef) / (b + ss_r)
    tau_new = (1.0 - eta) * tau + eta * raw
    return np.clip(tau_new, tau_floor, tau_ceiling)


def effective_rank_energy(
    lambdas_eeg: np.ndarray,
    lambdas_fmri: Optional[np.ndarray],
    fmri_mode: FMriMode,
    rel_thresh: float,
) -> Tuple[int, np.ndarray]:
    """
    Returns (R_eff, energy_r) where energy_r is used for thresholding.

    - aggregate: energy_r = mean_{i,f} lambda_eeg^2  (UNCHANGED from simulations)
    - separate : energy_r = mean_{i,f} lambda_eeg^2 + mean_i lambda_fmri^2
    """
    E_eeg = np.mean(lambdas_eeg * lambdas_eeg, axis=(0, 1))
    if fmri_mode == "separate" and (lambdas_fmri is not None):
        E_fm = np.mean(lambdas_fmri * lambdas_fmri, axis=0)
        E = E_eeg + E_fm
    else:
        E = E_eeg

    Emax = float(np.max(E)) if E.size else 0.0
    if Emax <= 0:
        return 0, E
    keep = (E / (Emax + 1e-12)) >= rel_thresh
    return int(np.sum(keep)), E

 
#  end-of-fit factor sorting (optional) 
def factor_energy_for_sort(
    lambdas_eeg: np.ndarray,
    lambdas_fmri: Optional[np.ndarray],
    fmri_mode: FMriMode,
    stat: SortMode,
) -> np.ndarray:
    """
    Energy used ONLY to order factors in the saved output.
    Does NOT affect estimation.

    - aggregate: energy = stat_{i,f}(lambda_eeg^2)
    - separate : energy = stat_{i,f}(lambda_eeg^2) + stat_i(lambda_fmri^2)
    """
    if stat == "none":
        raise ValueError("stat cannot be 'none' here.")

    sq_eeg = lambdas_eeg * lambdas_eeg
    if stat == "median":
        E_eeg = np.median(sq_eeg, axis=(0, 1))
    else:
        E_eeg = np.mean(sq_eeg, axis=(0, 1))

    if fmri_mode == "separate" and (lambdas_fmri is not None):
        sq_fm = lambdas_fmri * lambdas_fmri
        if stat == "median":
            E_fm = np.median(sq_fm, axis=0)
        else:
            E_fm = np.mean(sq_fm, axis=0)
        return E_eeg + E_fm

    return E_eeg


def apply_factor_permutation(
    Phi: np.ndarray,
    lambdas_eeg: np.ndarray,
    lambdas_fmri: Optional[np.ndarray],
    tau: Optional[np.ndarray],
    perm: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    Phi2 = Phi[:, perm]
    lam2 = lambdas_eeg[:, :, perm]
    lamf2 = lambdas_fmri[:, perm] if lambdas_fmri is not None else None
    tau2 = tau[perm] if tau is not None else None
    return Phi2, lam2, lamf2, tau2

 
#   init Phi from an anchor fit (optional) 
def init_phi_from_anchor(
    data: Dict[str, np.ndarray],
    R: int,
    init_fit_path: str,
    expand: InitExpand,
    seed: int,
) -> np.ndarray:
    z = dict(np.load(init_fit_path, allow_pickle=True))
    if "Phi_hat" not in z:
        raise ValueError(f"--init_fit file missing Phi_hat: {init_fit_path}")
    Phi0 = np.asarray(z["Phi_hat"], dtype=float)
    K = int(data["C_fmri"].shape[1])

    if Phi0.shape[0] != K:
        raise ValueError(f"--init_fit Phi_hat has K={Phi0.shape[0]} but data has K={K}")

    R0 = Phi0.shape[1]
    if R0 == R:
        return qr_retraction(Phi0)
    if R0 > R:
        return qr_retraction(Phi0[:, :R])

    # Need to expand to larger rank
    need = R - R0
    if expand == "random":
        rng = np.random.default_rng(seed)
        A = rng.normal(size=(K, need))
    else:
        # expand using fMRI eigenvectors not already spanned
        Cbar = np.mean(data["C_fmri"], axis=0)
        evals, evecs = la.eigh(0.5 * (Cbar + Cbar.T))
        U = evecs[:, ::-1]  # descending
        # pick candidates and orthogonalize against current Phi0
        A = U[:, :max(need * 2, need)]
        # remove projection onto Phi0
        A = A - Phi0 @ (Phi0.T @ A)

        # keep columns with nontrivial norm
        cols = []
        for j in range(A.shape[1]):
            v = A[:, j]
            nv = np.linalg.norm(v)
            if nv > 1e-10:
                cols.append(v / nv)
            if len(cols) >= need:
                break
        if len(cols) < need:
            rng = np.random.default_rng(seed)
            B = rng.normal(size=(K, need - len(cols)))
            cols.extend([B[:, j] for j in range(B.shape[1])])
        A = np.stack(cols[:need], axis=1)

    Phi_init = np.hstack([Phi0, A])
    return qr_retraction(Phi_init)

 
# Main fit
# -------------------------
def fit_model(data: Dict[str, np.ndarray], cfg: FitConfig) -> Dict[str, np.ndarray]:
    if cfg.use_clean_data:
        if "C_eeg_clean" not in data or "C_fmri_clean" not in data:
            raise KeyError("--use_clean_data requires C_eeg_clean and C_fmri_clean")
        C_eeg = data["C_eeg_clean"]
        C_fmri = data["C_fmri_clean"]
    else:
        C_eeg = data["C_eeg"]
        C_fmri = data["C_fmri"]
    W_eeg = data["W_eeg"]
    W_fmri = data["W_fmri"]
    F_fmri = data["m_fmri"].astype(int)
    w_f = data["w_m"].astype(float)

    sigma_eeg = float(data.get("sigma_eeg", 1.0))
    sigma_fmri = float(data.get("sigma_fmri", 1.0))
    include_diag = bool(int(data.get("include_diag", 1)))

    n, F, K, _ = C_eeg.shape

    # normalize weights over fMRI bins (robust; preserves simulations if already normalized)
    ws = float(np.sum(w_f[F_fmri]))
    if ws <= 0:
        raise ValueError("Sum of w_m over m_fmri is nonpositive.")
    w_f = w_f / ws

    # choose working rank
    R = int(cfg.Rmax) if cfg.use_ard else int(cfg.R)

    w_eeg = (1.0 / F) if cfg.w_eeg < 0 else float(cfg.w_eeg)
    w_fmri = float(cfg.w_fmri)

    # Initialize Phi from an anchor fit, the group-mean fMRI eigenspace, or random QR.
    if cfg.init_fit:
        Phi = init_phi_from_anchor(
            data=data, R=R, init_fit_path=cfg.init_fit,
            expand=cfg.init_expand, seed=cfg.init_seed
        )
    elif cfg.init_method == "fmri_eig":
        if w_fmri <= 0:
            raise ValueError("init_method='fmri_eig' requires w_fmri > 0")
        Cbar = np.mean(C_fmri, axis=0)
        _, evecs = la.eigh(0.5 * (Cbar + Cbar.T))
        Phi = qr_retraction(evecs[:, -R:])
    elif cfg.init_method == "random":
        rng = np.random.default_rng(cfg.init_seed)
        Phi = qr_retraction(rng.normal(size=(K, R)))
    else:
        raise ValueError("init_method must be 'fmri_eig' or 'random'")

    lambdas_eeg = np.full((n, F, R), 1e-2, dtype=float)
    lambdas_fmri = np.full((n, R), 1e-2, dtype=float) if (cfg.fmri_mode == "separate" and w_fmri > 0) else None

    # ARD init
    tau = None
    if cfg.use_ard:
        tau_floor = float(cfg.ard_tau_floor) if cfg.ard_tau_floor > 0 else float(cfg.alpha0)
        tau = np.full(R, max(tau_floor, cfg.alpha0), dtype=float)

    if cfg.verbose:
        print(
            f"[init] use_ard={cfg.use_ard} R={R} w_eeg={w_eeg}, w_fmri={w_fmri}, "
            f"fmri_mode={cfg.fmri_mode}, n_jobs={cfg.n_jobs}, solver={cfg.lam_solver}, "
            f"init_method={cfg.init_method}, init_fit={'yes' if cfg.init_fit else 'no'}, "
            f"fit_target={'clean' if cfg.use_clean_data else 'noisy'}"
        )
        if cfg.use_ard:
            print(
                f"       ARD: eta={cfg.ard_eta}, burnin={cfg.ard_burnin}, a=b={cfg.ard_a}, "
                f"tau_floor={cfg.ard_tau_floor}, tau_ceiling={cfg.ard_tau_ceiling}, rel_thresh={cfg.ard_energy_rel_thresh}"
            )

    J_new = np.inf

    for it in range(cfg.max_iter):
        tau_old = tau.copy() if (cfg.use_ard and tau is not None) else None

        # (A) lambda update
        lambdas_eeg, lambdas_fmri = lambda_update_all_subjects(
            C_eeg, C_fmri, Phi, W_eeg, W_fmri, F_fmri, w_f,
            sigma_eeg, sigma_fmri, cfg, include_diag, tau=tau_old
        )

        # (B) penalty constant
        pen_const = lambda_penalties(
            lambdas_eeg, cfg.alpha_lambda, cfg.alpha0, cfg.use_ard, tau_old, lambdas_fmri
        )

        # (C) data-fit + grad
        J_data, G = datafit_and_grad(
            C_eeg, C_fmri, Phi, lambdas_eeg, lambdas_fmri, W_eeg, W_fmri, F_fmri, w_f,
            sigma_eeg, sigma_fmri, w_eeg, w_fmri, cfg.fmri_mode
        )
        J_curr = J_data + pen_const

        # (D) Phi step + backtracking
        G = project_tangent(Phi, G)
        step = cfg.step_phi
        Phi_new = qr_retraction(Phi - step * G)

        J_data_new, _ = datafit_and_grad(
            C_eeg, C_fmri, Phi_new, lambdas_eeg, lambdas_fmri, W_eeg, W_fmri, F_fmri, w_f,
            sigma_eeg, sigma_fmri, w_eeg, w_fmri, cfg.fmri_mode
        )
        J_new = J_data_new + pen_const

        if cfg.backtrack:
            bt = 0
            while (J_new > J_curr) and (bt < cfg.backtrack_max):
                step *= cfg.backtrack_factor
                Phi_new = qr_retraction(Phi - step * G)
                J_data_new, _ = datafit_and_grad(
                    C_eeg, C_fmri, Phi_new, lambdas_eeg, lambdas_fmri, W_eeg, W_fmri, F_fmri, w_f,
                    sigma_eeg, sigma_fmri, w_eeg, w_fmri, cfg.fmri_mode
                )
                J_new = J_data_new + pen_const
                bt += 1

        if cfg.enforce_monotone and (J_new > J_curr):
            Phi_new = Phi
            J_new = J_curr
            step = 0.0

        Phi = Phi_new

        # (E) ARD tau update for NEXT iteration
        if cfg.use_ard and tau is not None and it >= cfg.ard_burnin:
            tau_floor = float(cfg.ard_tau_floor) if cfg.ard_tau_floor > 0 else float(cfg.alpha0)
            tau = ard_update_tau(
                lambdas_eeg, lambdas_fmri, tau,
                a=float(cfg.ard_a), b=float(cfg.ard_b), eta=float(cfg.ard_eta),
                tau_floor=tau_floor, tau_ceiling=float(cfg.ard_tau_ceiling),
            )

        rel = abs(J_curr - J_new) / max(1e-12, abs(J_curr))
        if cfg.verbose:
            msg = f"[it {it+1:02d}] J={J_new:.6e}  rel={rel:.3e}  step={step:.2e}"
            if cfg.use_ard and tau is not None:
                msg += f"  tau_min={tau.min():.2e} tau_med={np.median(tau):.2e} tau_max={tau.max():.2e}"
            print(msg)

        if rel < cfg.tol:
            break
 
    # end-of-fit sorting (optional ) 
    sort_order = np.arange(R)
    sort_energy = None
    if cfg.sort_factors != "none":
        sort_energy = factor_energy_for_sort(lambdas_eeg, lambdas_fmri, cfg.fmri_mode, cfg.sort_factors)
        sort_order = np.argsort(-sort_energy)
        Phi, lambdas_eeg, lambdas_fmri, tau = apply_factor_permutation(Phi, lambdas_eeg, lambdas_fmri, tau, sort_order)

    out: Dict[str, np.ndarray] = {
        "Phi_hat": Phi,
        "lambda_hat": lambdas_eeg,
        "J_final": np.array(J_new),
        "fmri_mode": np.array(cfg.fmri_mode, dtype=object),
        "sort_by_energy": np.array(cfg.sort_factors, dtype=object),
        "sort_order": sort_order.astype(int),
        "w_eeg": np.array(w_eeg, dtype=float),
        "w_fmri": np.array(w_fmri, dtype=float),
        "alpha_lambda": np.array(cfg.alpha_lambda, dtype=float),
        "alpha0": np.array(cfg.alpha0, dtype=float),
        "frobenius_symmetric_weighting": np.array(1, dtype=np.int32),
        "smooth_mask_gradient_uses_W2": np.array(1, dtype=np.int32),
        "init_method": np.array(cfg.init_method, dtype=object),
        "init_seed": np.array(cfg.init_seed, dtype=np.int64),
        "fit_target": np.array("clean" if cfg.use_clean_data else "noisy", dtype=object),
    }
    if lambdas_fmri is not None:
        out["lambda_fmri_hat"] = lambdas_fmri
    if sort_energy is not None:
        out["sort_energy"] = np.asarray(sort_energy, dtype=float)

    if cfg.use_ard and tau is not None:
        R_eff, E = effective_rank_energy(
            lambdas_eeg=lambdas_eeg,
            lambdas_fmri=lambdas_fmri,
            fmri_mode=cfg.fmri_mode,
            rel_thresh=cfg.ard_energy_rel_thresh,
        )
        out["tau_hat"] = tau
        out["energy_r"] = E
        out["R_eff"] = np.array(R_eff, dtype=int)
        out["R_eff_rel_thresh"] = np.array(cfg.ard_energy_rel_thresh, dtype=float)

    return out


def main():
    ap = argparse.ArgumentParser("Fit the shared-spatial-factor connectivity model.")
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)

    # fixed-R
    ap.add_argument("--R", type=int, default=5)

    # penalties
    ap.add_argument("--alpha_lambda", type=float, default=0.0)
    ap.add_argument("--alpha0", type=float, default=1e-6)

    # modality weights
    ap.add_argument("--w_fmri", type=float, default=1.0)
    ap.add_argument("--w_eeg", type=float, default=-1.0)

    # fMRI mode
    ap.add_argument("--fmri_mode", type=str, default="aggregate", choices=["aggregate", "separate"],
                    help="aggregate or separate modality-specific fMRI strengths.")

    # Phi update
    ap.add_argument("--step_phi", type=float, default=1e-2)
    ap.add_argument("--max_iter", type=int, default=20)
    ap.add_argument("--tol", type=float, default=1e-6)

    # parallel
    ap.add_argument("--n_jobs", type=int, default=1)
    ap.add_argument("--lam_solver", type=str, default="bvls", choices=["lsq", "bvls"])

    ap.add_argument("--no_backtrack", action="store_true")
    ap.add_argument("--no_monotone", action="store_true")

    # ARD
    ap.add_argument("--use_ard", action="store_true")
    ap.add_argument("--Rmax", type=int, default=15)
    ap.add_argument("--ard_eta", type=float, default=0.2)
    ap.add_argument("--ard_a", type=float, default=1e-6)
    ap.add_argument("--ard_b", type=float, default=1e-6)
    ap.add_argument("--ard_burnin", type=int, default=2)
    ap.add_argument("--ard_tau_floor", type=float, default=1e-6)
    ap.add_argument("--ard_tau_ceiling", type=float, default=1e9)
    ap.add_argument("--ard_energy_rel_thresh", type=float, default=1e-2)

    # Optional: end-of-fit sorting (off by default)
    ap.add_argument("--sort_factors", type=str, default="none", choices=["none", "median", "mean"],
                    help="End-of-fit factor sorting for stable indexing (reporting only). Default none.")

    # init from anchor fit ( Optional 
    ap.add_argument("--init_fit", type=str, default="", help="Warm-start Phi from an anchor fit .npz (truncate/expand).")
    ap.add_argument("--init_expand", type=str, default="random", choices=["random", "fmri_eig"],
                    help="If init_fit has fewer factors than needed, how to expand.")
    ap.add_argument("--init_seed", type=int, default=0, help="Seed for random init/expansion.")
    ap.add_argument("--init_method", choices=["fmri_eig", "random"], default="fmri_eig")
    ap.add_argument("--use_clean_data", action="store_true", help="Fit C_eeg_clean and C_fmri_clean instead of noisy summaries.")

    args = ap.parse_args()
    data = dict(np.load(args.data, allow_pickle=True))

    cfg = FitConfig(
        R=args.R,
        alpha_lambda=args.alpha_lambda,
        alpha0=args.alpha0,
        w_fmri=args.w_fmri,
        w_eeg=args.w_eeg,
        fmri_mode=args.fmri_mode,
        step_phi=args.step_phi,
        max_iter=args.max_iter,
        tol=args.tol,
        n_jobs=args.n_jobs,
        lam_solver=args.lam_solver,
        backtrack=not args.no_backtrack,
        enforce_monotone=not args.no_monotone,
        use_ard=bool(args.use_ard),
        Rmax=args.Rmax,
        ard_eta=args.ard_eta,
        ard_a=args.ard_a,
        ard_b=args.ard_b,
        ard_burnin=args.ard_burnin,
        ard_tau_floor=args.ard_tau_floor,
        ard_tau_ceiling=args.ard_tau_ceiling,
        ard_energy_rel_thresh=args.ard_energy_rel_thresh,
        sort_factors=args.sort_factors,
        init_fit=args.init_fit,
        init_expand=args.init_expand,
        init_seed=args.init_seed,
        init_method=args.init_method,
        use_clean_data=bool(args.use_clean_data),
        verbose=True,
    )

    fit = fit_model(data, cfg)
    np.savez_compressed(args.out, **fit)
    print("Saved fit to:", args.out)


if __name__ == "__main__":
    main()
