#!/usr/bin/env python3
"""Utilities for leakage-controlled (nested cross-validation) MPI--LEMON prediction and fitting.

Every operation that defines the imaging representation is performed using the
outer training subjects only:  modality scaling, dictionary fitting, factor 
ordering, held-out score estimation, feature standardization, and ridge tuning.
"""
from __future__ import annotations

import contextlib
import importlib.util
import inspect
import io
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.linalg as la
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

Array = np.ndarray


def scalar_value(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    arr = np.asarray(value)
    if arr.size != 1:
        return default
    out = arr.reshape(-1)[0]
    return out.item() if hasattr(out, "item") else out



def normalize_subject_id(value: Any) -> str:
    """Normalize MPI--LEMON identifiers to six digits when possible."""
    text = str(value.item() if hasattr(value, "item") else value).strip()
    text = text.replace("sub-", "").replace("SUB-", "")
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    if text.isdigit():
        return text.zfill(6)
    digits = "".join(ch for ch in text if ch.isdigit())
    if digits:
        return digits[-6:].zfill(6)
    return text


def load_npz(path: str | Path) -> Dict[str, Any]:
    path = Path(path).expanduser().resolve()
    with np.load(path, allow_pickle=True) as z:
        return {key: z[key] for key in z.files}


def infer_subject_ids(data: Mapping[str, Any]) -> Array:
    for key in ("subject_ids", "subject_id", "subjects", "participant_ids", "ids"):
        if key in data:
            values = np.asarray(data[key]).reshape(-1)
            return np.asarray([normalize_subject_id(x) for x in values], dtype=object)
    raise KeyError("The joint NPZ does not contain subject identifiers")


def infer_include_diag(data: Mapping[str, Any]) -> bool:
    if "include_diag" in data:
        return bool(int(scalar_value(data["include_diag"], 1)))
    eeg = int(scalar_value(data.get("include_diag_eeg"), 1))
    fmri = int(scalar_value(data.get("include_diag_fmri"), 1))
    if eeg != fmri:
        raise ValueError(
            "The existing fitter uses one diagonal-inclusion flag, but the EEG and fMRI flags differ"
        )
    return bool(eeg)



def validate_joint_npz(data: Mapping[str, Any]) -> Dict[str, Any]:
    required = ("C_eeg", "C_fmri", "W_eeg", "W_fmri", "omega", "m_fmri", "w_m")
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"Joint NPZ missing required keys: {missing}")

    C_eeg = np.asarray(data["C_eeg"])
    C_fmri = np.asarray(data["C_fmri"])
    W_eeg = np.asarray(data["W_eeg"])
    W_fmri = np.asarray(data["W_fmri"])
    omega = np.asarray(data["omega"]).reshape(-1)
    ids = infer_subject_ids(data)

    if C_eeg.ndim != 4:
        raise ValueError(f"C_eeg must have shape (n,F,K,K), found {C_eeg.shape}")
    n, F, K, K2 = C_eeg.shape
    if K != K2 or C_fmri.shape != (n, K, K):
        raise ValueError("EEG and fMRI covariance dimensions are inconsistent")
    if W_eeg.shape != (K, K) or W_fmri.shape != (K, K):
        raise ValueError("W_eeg and W_fmri must have shape (K,K)")
    if omega.size != F:
        raise ValueError(f"omega has length {omega.size}, expected F={F}")
    if ids.size != n or len(set(ids.tolist())) != n:
        raise ValueError("Subject identifiers are missing or duplicated")
    if not np.isfinite(C_eeg).all() or not np.isfinite(C_fmri).all():
        raise ValueError("Connectivity arrays contain nonfinite values")

    return {
        "n": int(n),
        "F": int(F),
        "K": int(K),
        "subject_ids": ids,
        "include_diag": infer_include_diag(data),
    }


def infer_column(df: pd.DataFrame, requested: str, candidates: Sequence[str]) -> str:
    if requested:
        if requested not in df.columns:
            raise KeyError(f"Column '{requested}' not found")
        return requested
    lower = {str(col).lower(): str(col) for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    raise KeyError(f"Could not infer a column from {list(candidates)}")



def load_phenotypes(
    path: str | Path,
    outcome: str = "age_mid",
    sex_column: str = "male",
    id_column: str = "",
) -> pd.DataFrame:
    df = pd.read_csv(Path(path).expanduser().resolve())
    id_col = infer_column(
        df,
        id_column,
        ("subject_id", "participant_id", "subject", "ID", "id", "file_id"),
    )
    outcome_col = infer_column(df, outcome, ("age_mid", "age", "age_years"))
    if sex_column not in df.columns:
        raise KeyError(f"Sex covariate '{sex_column}' not found")

    out = pd.DataFrame(
        {
            "subject_id": df[id_col].map(normalize_subject_id),
            "outcome": pd.to_numeric(df[outcome_col], errors="coerce"),
            "male": pd.to_numeric(df[sex_column], errors="coerce"),
        }
    )
    out = out.dropna(subset=["subject_id", "outcome", "male"])
    if out["subject_id"].duplicated().any():
        raise ValueError("Phenotype table contains duplicate subject IDs")
    return out


def align_condition(
    data: Mapping[str, Any],
    phenotype: pd.DataFrame,
    fold_subjects: Sequence[str],
) -> Dict[str, Any]:
    info = validate_joint_npz(data)
    data_index = {sid: i for i, sid in enumerate(info["subject_ids"].tolist())}
    pheno = phenotype.set_index("subject_id")
    ordered = [normalize_subject_id(x) for x in fold_subjects]
    missing_data = [sid for sid in ordered if sid not in data_index]
    missing_pheno = [sid for sid in ordered if sid not in pheno.index]
    if missing_data or missing_pheno:
        raise ValueError(
            f"Fold subjects missing from condition data={missing_data[:5]} or phenotypes={missing_pheno[:5]}"
        )
    return {
        "subject_ids": np.asarray(ordered, dtype=object),
        "npz_indices": np.asarray([data_index[sid] for sid in ordered], dtype=int),
        "outcome": pheno.loc[ordered, "outcome"].to_numpy(float),
        "male": pheno.loc[ordered, "male"].to_numpy(float)[:, None],
        "info": info,
    }


def masked_frobenius_norms(C: Array, W: Array) -> Array:
    leading = int(np.prod(C.shape[:-2]))
    weighted = (np.asarray(C, float) * np.asarray(W, float)).reshape(leading, -1)
    return np.sqrt(np.sum(weighted * weighted, axis=1))


def training_scales(C_eeg: Array, C_fmri: Array, W_eeg: Array, W_fmri: Array) -> Tuple[float, float]:
    n, F = C_eeg.shape[:2]
    eeg_norm = masked_frobenius_norms(C_eeg, W_eeg).reshape(n, F).mean(axis=1)
    fmri_norm = masked_frobenius_norms(C_fmri, W_fmri)
    return max(float(np.median(eeg_norm)), 1e-12), max(float(np.median(fmri_norm)), 1e-12)


def scaled_subset(
    data: Mapping[str, Any],
    indices: Sequence[int],
    eeg_scale: float,
    fmri_scale: float,
) -> Dict[str, Any]:
    idx = np.asarray(indices, dtype=int)
    ids = infer_subject_ids(data)
    return {
        "C_eeg": np.asarray(data["C_eeg"], float)[idx] / eeg_scale,
        "C_fmri": np.asarray(data["C_fmri"], float)[idx] / fmri_scale,
        "W_eeg": np.asarray(data["W_eeg"], float),
        "W_fmri": np.asarray(data["W_fmri"], float),
        "omega": np.asarray(data["omega"], float),
        "m_fmri": np.asarray(data["m_fmri"], int),
        "w_m": np.asarray(data["w_m"], float),
        "include_diag": np.array(int(infer_include_diag(data)), dtype=int),
        "sigma_eeg": np.array(1.0),
        "sigma_fmri": np.array(1.0),
        "subject_ids": ids[idx],
    }



def import_fitter(path: str | Path):
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    name = f"public_fitter_{abs(hash(str(path)))}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load fitter: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    for required in ("FitConfig", "fit_model", "lambda_update_all_subjects"):
        if not hasattr(module, required):
            raise AttributeError(f"Fitter missing required object: {required}")
    return module


def make_fit_config(module, **kwargs):
    fields = getattr(module.FitConfig, "__dataclass_fields__", {})
    if fields:
        kwargs = {key: value for key, value in kwargs.items() if key in fields}
    else:
        params = inspect.signature(module.FitConfig).parameters
        kwargs = {key: value for key, value in kwargs.items() if key in params}
    return module.FitConfig(**kwargs)


@dataclass
class FitResult:
    fit: Dict[str, Any]
    start_method: str
    start_seed: int
    objective: float
    iterations: int
    last_relative_change: float
    converged: bool
    log_text: str


def random_basis(K: int, R: int, seed: int) -> Array:
    rng = np.random.default_rng(seed)
    Q, _ = la.qr(rng.normal(size=(K, R)), mode="economic")
    return Q[:, :R]


def parse_fit_log(text: str, tolerance: float) -> Tuple[int, float, bool]:
    hits = re.findall(r"\[it\s+(\d+)\].*?rel=([0-9.eE+\-]+)", text)
    if not hits:
        return 0, float("nan"), False
    iterations = int(hits[-1][0])
    relative = float(hits[-1][1])
    return iterations, relative, bool(relative < tolerance)


def fit_training_model(
    fitter,
    data: Mapping[str, Any],
    rank: int,
    mode: str,
    seed: int,
    n_starts: int,
    alpha_lambda: float,
    alpha0: float,
    step_phi: float,
    max_iter: int,
    tolerance: float,
    n_jobs: int,
    solver: str,
    workdir: str | Path,
) -> Tuple[FitResult, List[FitResult]]:
    if mode not in {"fused", "eeg_only", "fmri_only"}:
        raise ValueError(f"Unknown fit mode: {mode}")
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    K = int(np.asarray(data["C_fmri"]).shape[1])

    starts: List[Tuple[str, int]] = []
    if mode in {"fused", "fmri_only"}:
        starts.append(("fmri_eigenvectors", seed))
        starts.extend(("random", seed + 100000 * j) for j in range(1, n_starts))
    else:
        starts.extend(("random", seed + 100000 * (j + 1)) for j in range(n_starts))

    results: List[FitResult] = []
    for number, (method, start_seed) in enumerate(starts, start=1):
        init_fit = ""
        if method == "random" and mode in {"fused", "fmri_only"}:
            init_path = workdir / f"random_start_{number}.npz"
            np.savez_compressed(init_path, Phi_hat=random_basis(K, rank, start_seed))
            init_fit = str(init_path)

        config = make_fit_config(
            fitter,
            R=rank,
            alpha_lambda=alpha_lambda,
            alpha0=alpha0,
            w_eeg=(-1.0 if mode in {"fused", "eeg_only"} else 0.0),
            w_fmri=(1.0 if mode in {"fused", "fmri_only"} else 0.0),
            fmri_mode="separate",
            step_phi=step_phi,
            max_iter=max_iter,
            tol=tolerance,
            n_jobs=n_jobs,
            lam_solver=solver,
            backtrack=True,
            enforce_monotone=True,
            use_ard=False,
            sort_factors="none",
            init_fit=init_fit,
            init_expand="random",
            init_seed=start_seed,
            init_method=("fmri_eig" if method == "fmri_eigenvectors" else "random"),
            verbose=True,
        )
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            fit = fitter.fit_model(dict(data), config)
        log_text = buffer.getvalue()
        iterations, last_rel, converged = parse_fit_log(log_text, tolerance)
        result = FitResult(
            fit=fit,
            start_method=method,
            start_seed=start_seed,
            objective=float(np.asarray(fit.get("J_final", np.nan))),
            iterations=iterations,
            last_relative_change=last_rel,
            converged=converged,
            log_text=log_text,
        )
        results.append(result)
        (workdir / f"start_{number}_{method}.log").write_text(log_text)

    valid = [result for result in results if np.isfinite(result.objective)]
    if not valid:
        raise RuntimeError(f"No finite {mode} fit")
    return min(valid, key=lambda item: item.objective), results



def score_fixed_dictionary(
    fitter,
    data: Mapping[str, Any],
    Phi: Array,
    rank: int,
    mode: str,
    alpha_lambda: float,
    alpha0: float,
    n_jobs: int,
    solver: str,
) -> Tuple[Array, Optional[Array]]:
    config = make_fit_config(
        fitter,
        R=rank,
        alpha_lambda=alpha_lambda,
        alpha0=alpha0,
        w_eeg=(-1.0 if mode in {"fused", "eeg_only"} else 0.0),
        w_fmri=(1.0 if mode in {"fused", "fmri_only"} else 0.0),
        fmri_mode="separate",
        n_jobs=n_jobs,
        lam_solver=solver,
        use_ard=False,
        sort_factors="none",
        verbose=False,
    )
    C_eeg = np.asarray(data["C_eeg"], float)
    C_fmri = np.asarray(data["C_fmri"], float)
    W_eeg = np.asarray(data["W_eeg"], float)
    W_fmri = np.asarray(data["W_fmri"], float)
    m_fmri = np.asarray(data["m_fmri"], int)
    w_m = np.asarray(data["w_m"], float)
    w_m = w_m / float(np.sum(w_m[m_fmri]))
    lambdas_eeg, lambdas_fmri = fitter.lambda_update_all_subjects(
        C_eeg,
        C_fmri,
        np.asarray(Phi, float),
        W_eeg,
        W_fmri,
        m_fmri,
        w_m,
        1.0,
        1.0,
        config,
        bool(int(scalar_value(data["include_diag"], 1))),
        tau=None,
    )
    return np.asarray(lambdas_eeg, float), (
        None if lambdas_fmri is None else np.asarray(lambdas_fmri, float)
    )


def order_factors(
    Phi: Array,
    train_eeg: Array,
    train_fmri: Optional[Array],
    test_eeg: Array,
    test_fmri: Optional[Array],
    mode: str,
) -> Dict[str, Any]:
    R = train_eeg.shape[-1]
    energy = np.zeros(R, float)
    if mode in {"fused", "eeg_only"}:
        energy += np.median(train_eeg**2, axis=(0, 1))
    if mode in {"fused", "fmri_only"}:
        if train_fmri is None:
            raise ValueError("Missing fMRI strengths")
        energy += np.median(train_fmri**2, axis=0)
    order = np.argsort(-energy)
    return {
        "Phi": np.asarray(Phi)[:, order],
        "train_eeg": train_eeg[:, :, order],
        "test_eeg": test_eeg[:, :, order],
        "train_fmri": None if train_fmri is None else train_fmri[:, order],
        "test_fmri": None if test_fmri is None else test_fmri[:, order],
        "energy": energy[order],
        "order": order,
    }


def band_indices(omega: Array) -> Dict[str, Array]:
    omega = np.asarray(omega, float).reshape(-1)
    gamma_hi = float(np.max(omega)) + 1e-9
    return {
        "theta": np.where((omega >= 4.0) & (omega < 8.0))[0],
        "alpha": np.where((omega >= 8.0) & (omega < 13.0))[0],
        "beta": np.where((omega >= 13.0) & (omega < 30.0))[0],
        "gamma": np.where((omega >= 30.0) & (omega < gamma_hi))[0],
    }


def eeg_summary_features(lambdas: Array, omega: Array, k: int) -> Tuple[Array, List[str]]:
    lambdas = np.asarray(lambdas, float)[:, :, :k]
    omega = np.asarray(omega, float).reshape(-1)
    bands = band_indices(omega)
    blocks: List[Array] = []
    names: List[str] = []
    for r in range(k):
        factor = lambdas[:, :, r]
        for band in ("theta", "alpha", "beta", "gamma"):
            idx = bands[band]
            if idx.size == 0:
                raise ValueError(f"No frequency bins were found for the {band} band")
            blocks.append(np.sum(factor[:, idx], axis=1)[:, None])
            names.append(f"EEG_{band}_r{r + 1}")
        denominator = np.sum(factor, axis=1) + 1e-12
        blocks.append(((factor @ omega) / denominator)[:, None])
        names.append(f"EEG_specCOM_r{r + 1}")
    return np.concatenate(blocks, axis=1), names


def fmri_features(lambdas: Array, k: int) -> Tuple[Array, List[str]]:
    X = np.asarray(lambdas, float)[:, :k]
    return X, [f"fMRI_strength_r{r + 1}" for r in range(k)]


def prediction_models(
    ordered: Mapping[str, Mapping[str, Any]],
    omega: Array,
    k: int,
    male_train: Array,
    male_test: Array,
) -> Dict[str, Tuple[Array, Array, int]]:
    fused = ordered["fused"]
    eeg_only = ordered["eeg_only"]
    fmri_only = ordered["fmri_only"]

    fused_eeg_train, _ = eeg_summary_features(fused["train_eeg"], omega, k)
    fused_eeg_test, _ = eeg_summary_features(fused["test_eeg"], omega, k)
    fused_fmri_train, _ = fmri_features(fused["train_fmri"], k)
    fused_fmri_test, _ = fmri_features(fused["test_fmri"], k)

    separate_eeg_train, _ = eeg_summary_features(eeg_only["train_eeg"], omega, k)
    separate_eeg_test, _ = eeg_summary_features(eeg_only["test_eeg"], omega, k)
    separate_fmri_train, _ = fmri_features(fmri_only["train_fmri"], k)
    separate_fmri_test, _ = fmri_features(fmri_only["test_fmri"], k)

    def with_male(brain_train: Optional[Array], brain_test: Optional[Array]) -> Tuple[Array, Array, int]:
        if brain_train is None:
            return male_train.copy(), male_test.copy(), 0
        return (
            np.concatenate([male_train, brain_train], axis=1),
            np.concatenate([male_test, brain_test], axis=1),
            int(brain_train.shape[1]),
        )

    return {
        "M0": with_male(None, None),
        "Mf": with_male(fused_fmri_train, fused_fmri_test),
        "Me": with_male(fused_eeg_train, fused_eeg_test),
        "Mfused": with_male(
            np.concatenate([fused_fmri_train, fused_eeg_train], axis=1),
            np.concatenate([fused_fmri_test, fused_eeg_test], axis=1),
        ),
        "Mconcat": with_male(
            np.concatenate([separate_fmri_train, separate_eeg_train], axis=1),
            np.concatenate([separate_fmri_test, separate_eeg_test], axis=1),
        ),
    }


def regression_strata(y: Array, male: Array, n_splits: int) -> Optional[Array]:
    y = np.asarray(y, float)
    male = np.asarray(male, float).reshape(-1)
    for bins in range(5, 1, -1):
        try:
            age_bin = np.asarray(pd.qcut(y, q=bins, labels=False, duplicates="drop"), int)
        except ValueError:
            continue
        labels = np.asarray([f"{a}_{int(round(s))}" for a, s in zip(age_bin, male)], object)
        counts = pd.Series(labels).value_counts()
        if not counts.empty and int(counts.min()) >= n_splits:
            return labels
    return None


def inner_splits(y: Array, male: Array, n_splits: int, seed: int):
    strata = regression_strata(y, male, n_splits)
    if strata is not None:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(splitter.split(np.zeros(len(y)), strata)), "stratified_age_sex"
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(y)))), "kfold"


def tune_ridge(
    X: Array,
    y: Array,
    male: Array,
    alphas: Sequence[float],
    n_splits: int,
    seed: int,
) -> Tuple[float, pd.DataFrame, str]:
    splits, split_method = inner_splits(y, male, n_splits, seed)
    rows: List[Dict[str, Any]] = []
    for alpha in alphas:
        scores = []
        for fold, (train, valid) in enumerate(splits, start=1):
            model = Pipeline(
                [("standardize", StandardScaler()), ("ridge", Ridge(alpha=float(alpha)))]
            )
            model.fit(X[train], y[train])
            prediction = model.predict(X[valid])
            score = float(r2_score(y[valid], prediction))
            scores.append(score)
            rows.append({"alpha": float(alpha), "inner_fold": fold, "r2": score})
        rows.append({"alpha": float(alpha), "inner_fold": 0, "r2": float(np.mean(scores))})
    table = pd.DataFrame(rows)
    means = table.loc[table["inner_fold"] == 0].sort_values(
        ["r2", "alpha"], ascending=[False, True]
    )
    return float(means.iloc[0]["alpha"]), table, split_method



def parse_factor_sets(text: str, rank: int) -> List[Tuple[str, int]]:
    out: List[Tuple[str, int]] = []
    for token in [piece.strip().lower() for piece in text.split(",") if piece.strip()]:
        if token in {"all", "full"}:
            label, k = "all", rank
        else:
            k = int(token)
            if not 1 <= k <= rank:
                raise ValueError(f"Top-K value {k} is outside 1..{rank}")
            label = str(k)
        if label not in [existing[0] for existing in out]:
            out.append((label, k))
    if not out:
        raise ValueError("At least one factor set is required")
    return out
