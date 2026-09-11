#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

PRIMARY_FEATURE_TYPES = ("fmri", "eeg_total", "theta", "alpha", "beta", "gamma", "centroid")
EEG_FEATURE_TYPES = ("eeg_total", "theta", "alpha", "beta", "gamma", "centroid")
MULTIVARIATE_TYPES = ("fmri", "theta", "alpha", "beta", "gamma", "centroid")


def bh_adjust(values: Sequence[float]) -> np.ndarray:
    p = np.asarray(values, float)
    out = np.full(p.shape, np.nan, dtype=float)
    valid = np.isfinite(p)
    if not valid.any():
        return out
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    q = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    restored = np.empty_like(q)
    restored[order] = q
    out[valid] = restored
    return out


def usable_covariates(table: pd.DataFrame, candidates: Sequence[str]) -> List[str]:
    result = []
    for variable in candidates:
        if variable not in table.columns:
            continue
        values = pd.to_numeric(table[variable], errors="coerce")
        if values.notna().sum() == 0 or values.dropna().nunique() <= 1:
            continue
        result.append(variable)
    return result


def complete_cases(
    table: pd.DataFrame,
    variables: Sequence[str],
    mask: pd.Series | np.ndarray | None = None,
) -> pd.DataFrame:
    data = table.loc[:, variables].copy() if mask is None else table.loc[mask, variables].copy()
    for variable in variables:
        data[variable] = pd.to_numeric(data[variable], errors="coerce")
    return data.dropna()


def standardize(
    data: pd.DataFrame,
    columns: Sequence[str],
    binary_columns: Sequence[str] = ("male", "education_gymnasium"),
) -> pd.DataFrame:
    out = data.copy()
    binary = set(binary_columns)
    for variable in columns:
        values = pd.to_numeric(out[variable], errors="coerce")
        if variable in binary:
            out[variable] = values
            continue
        sd = values.std(ddof=1)
        out[variable] = (values - values.mean()) / sd if np.isfinite(sd) and sd > 0 else values
    return out


def fit_age_model(
    table: pd.DataFrame,
    outcome: str,
    covariates: Sequence[str],
    mask: pd.Series | np.ndarray | None = None,
) -> Dict[str, float | int]:
    variables = [outcome, "age_mid", *covariates]
    data = complete_cases(table, variables, mask)
    if len(data) < len(covariates) + 4:
        return {k: np.nan for k in [
            "beta_age_std", "se_age_std", "ci_low_std", "ci_high_std",
            "p_age", "beta_age_raw_per_year", "r_squared"
        ]} | {"n": int(len(data))}

    z = standardize(data, [outcome, "age_mid", *covariates])
    Xz = sm.add_constant(z[["age_mid", *covariates]], has_constant="add")
    fitz = sm.OLS(z[outcome], Xz).fit(cov_type="HC3")
    ci = fitz.conf_int().loc["age_mid"]

    X = sm.add_constant(data[["age_mid", *covariates]], has_constant="add")
    fit = sm.OLS(data[outcome], X).fit(cov_type="HC3")

    return {
        "n": int(len(data)),
        "beta_age_std": float(fitz.params["age_mid"]),
        "se_age_std": float(fitz.bse["age_mid"]),
        "ci_low_std": float(ci.iloc[0]),
        "ci_high_std": float(ci.iloc[1]),
        "p_age": float(fitz.pvalues["age_mid"]),
        "beta_age_raw_per_year": float(fit.params["age_mid"]),
        "r_squared": float(fitz.rsquared),
    }


def comparison_fields(base_beta: float, adjusted_beta: float, threshold: float) -> Dict[str, object]:
    if not np.isfinite(base_beta) or not np.isfinite(adjusted_beta):
        return {
            "sign_stable": np.nan,
            "substantive_base": np.nan,
            "sign_stable_if_substantive": np.nan,
            "absolute_beta_change": np.nan,
            "percent_attenuation": np.nan,
        }
    stable = bool(np.sign(base_beta) == np.sign(adjusted_beta))
    substantive = bool(abs(base_beta) >= threshold)
    attenuation = 100.0 * (1.0 - abs(adjusted_beta) / abs(base_beta)) if stable and abs(base_beta) > 1e-10 else np.nan
    return {
        "sign_stable": stable,
        "substantive_base": substantive,
        "sign_stable_if_substantive": stable if substantive else np.nan,
        "absolute_beta_change": abs(adjusted_beta - base_beta),
        "percent_attenuation": attenuation,
    }


def residuals(data: pd.DataFrame, outcome: str, covariates: Sequence[str]) -> np.ndarray:
    X = sm.add_constant(data[list(covariates)], has_constant="add")
    return sm.OLS(data[outcome], X).fit().resid.to_numpy(float)


def permutation_correlation(
    x: np.ndarray,
    y: np.ndarray,
    n_perm: int,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    observed = float(np.corrcoef(x, y)[0, 1])
    exceed = 0
    for _ in range(n_perm):
        value = float(np.corrcoef(x, y[rng.permutation(len(y))])[0, 1])
        exceed += abs(value) >= abs(observed)
    return observed, float((exceed + 1) / (n_perm + 1))


def multivariate_ols(Y: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    beta = np.linalg.lstsq(X, Y, rcond=None)[0]
    fitted = X @ beta
    resid = Y - fitted
    return beta, fitted, resid.T @ resid


def pillai_added_age(
    Y: np.ndarray,
    X_reduced: np.ndarray,
    age: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    X_full = np.column_stack([X_reduced, age])
    _, fitted_reduced, E_reduced = multivariate_ols(Y, X_reduced)
    beta_full, _, E_full = multivariate_ols(Y, X_full)
    H = (E_reduced - E_full)
    H = (H + H.T) / 2.0
    E = (E_full + E_full.T) / 2.0
    pillai = float(np.trace(H @ np.linalg.pinv(H + E)))
    return pillai, beta_full[-1, :], fitted_reduced, Y - fitted_reduced


def permutation_multivariate_age(
    Y: np.ndarray,
    X_reduced: np.ndarray,
    age: np.ndarray,
    n_perm: int,
    rng: np.random.Generator,
) -> Tuple[float, float, np.ndarray]:
    observed, age_vector, fitted_reduced, residual = pillai_added_age(Y, X_reduced, age)
    exceed = 0
    for _ in range(n_perm):
        Y_perm = fitted_reduced + residual[rng.permutation(len(residual)), :]
        value, _, _, _ = pillai_added_age(Y_perm, X_reduced, age)
        exceed += value >= observed
    return observed, float((exceed + 1) / (n_perm + 1)), age_vector


def run_multivariate_family(
    table: pd.DataFrame,
    covariates: Sequence[str],
    n_perm: int,
    rng: np.random.Generator,
    label: str,
) -> pd.DataFrame:
    rows = []
    for network in range(1, 13):
        outcomes = [f"{kind}_r{network:02d}" for kind in MULTIVARIATE_TYPES]
        data = complete_cases(table, [*outcomes, "age_mid", *covariates])
        Y = standardize(data, outcomes, binary_columns=())[outcomes].to_numpy(float)
        design = standardize(data, ["age_mid", *covariates])
        X_reduced = sm.add_constant(design[list(covariates)], has_constant="add").to_numpy(float)
        age = design["age_mid"].to_numpy(float)
        pillai, p_value, age_vector = permutation_multivariate_age(Y, X_reduced, age, n_perm, rng)
        rows.append({
            "analysis": label,
            "network": network,
            "n": len(data),
            "pillai_trace_age": pillai,
            "permutation_p_age": p_value,
            "age_vector_l2": float(np.linalg.norm(age_vector)),
            "age_beta_fmri": age_vector[0],
            "age_beta_theta": age_vector[1],
            "age_beta_alpha": age_vector[2],
            "age_beta_beta": age_vector[3],
            "age_beta_gamma": age_vector[4],
            "age_beta_centroid": age_vector[5],
            "reduced_covariates": ";".join(covariates),
        })
    result = pd.DataFrame(rows)
    result["q_age"] = bh_adjust(result["permutation_p_age"])
    return result


def normalize_subject_id(value: object) -> str:
    text = str(value).strip().replace('sub-', '').replace('SUB-', '')
    digits = ''.join(ch for ch in text if ch.isdigit())
    return digits[-6:].zfill(6) if digits else text


def build_public_analysis_table(
    features_csv: str | Path,
    metadata_csv: str | Path,
    fitted_npz: str | Path,
    model_input_npz: str | Path,
) -> pd.DataFrame:
    features = pd.read_csv(features_csv, dtype={'subject_id': str})
    metadata = pd.read_csv(metadata_csv, dtype={'subject_id': str})
    features['subject_id'] = features['subject_id'].map(normalize_subject_id)
    metadata['subject_id'] = metadata['subject_id'].map(normalize_subject_id)
    table = features.merge(metadata, on='subject_id', how='left', validate='one_to_one')

    if 'has_eo' in table.columns:
        table = table.loc[pd.to_numeric(table['has_eo'], errors='coerce').fillna(0) == 1].copy()

    # Compute the below-30-Hz centroid directly from the released fitted strengths.
    with np.load(fitted_npz, allow_pickle=True) as fit, np.load(model_input_npz, allow_pickle=True) as inp:
        ids = np.asarray([normalize_subject_id(x) for x in fit['subject_ids']], object)
        lam = np.asarray(fit['lambda_eeg_all'], float)
        omega = np.asarray(inp['omega'], float).reshape(-1)
    below = omega < 30.0
    numerator = np.sum(lam[:, below, :] * omega[below][None, :, None], axis=1)
    denominator = np.sum(lam[:, below, :], axis=1)
    centroid = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)
    extra = pd.DataFrame({'subject_id': ids})
    for r in range(centroid.shape[1]):
        extra[f'centroid_lt30_r{r+1:02d}'] = centroid[:, r]
    table = table.merge(extra, on='subject_id', how='left', validate='one_to_one')
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description='Age-association robustness and multimodal coordination analyses.')
    parser.add_argument('--features', default='lemon_eo_subject_features.csv')
    parser.add_argument('--metadata', default='lemon_subject_metadata.csv')
    parser.add_argument('--fitted', default='lemon_eo_fitted_representation.npz')
    parser.add_argument('--model-input', default='lemon_eo_model_inputs.npz')
    parser.add_argument('--outdir', default='age_association_results')
    parser.add_argument('--n-perm', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--high-emg-quantile', type=float, default=0.90)
    parser.add_argument('--substantive-beta-threshold', type=float, default=0.10)
    args = parser.parse_args()

    table = build_public_analysis_table(args.features, args.metadata, args.fitted, args.model_input)
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    base_covariates = usable_covariates(table, ['male'])
    fmri_core = usable_covariates(table, [
        'male', 'education_gymnasium', 'log1p_mean_fd', 'mean_std_dvars'
    ])
    eeg_core = usable_covariates(table, [
        'male', 'education_gymnasium', 'log_n_windows', 'channel_retention'
    ])
    eeg_spectral = usable_covariates(table, [
        'aperiodic_exponent_median', 'hf_log_relative_p90'
    ])
    eeg_full = list(dict.fromkeys([*eeg_core, *eeg_spectral]))
    print("Base covariates:", base_covariates)
    print("fMRI core covariates:", fmri_core)
    print("EEG core covariates:", eeg_core)
    print("EEG spectral covariates:", eeg_spectral)

    core_rows: List[Dict[str, object]] = []
    spectral_rows: List[Dict[str, object]] = []

    for feature_type in PRIMARY_FEATURE_TYPES:
        modality = "fmri" if feature_type == "fmri" else "eeg"
        core_covariates = fmri_core if modality == "fmri" else eeg_core
        for network in range(1, 13):
            outcome = f"{feature_type}_r{network:02d}"
            matched_variables = [outcome, "age_mid", *core_covariates]
            matched_mask = table[matched_variables].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
            base_all = fit_age_model(table, outcome, base_covariates)
            matched_base = fit_age_model(table, outcome, base_covariates, matched_mask)
            core = fit_age_model(table, outcome, core_covariates, matched_mask)
            comp = comparison_fields(float(matched_base["beta_age_std"]), float(core["beta_age_std"]), args.substantive_beta_threshold)
            core_rows.append({
                "feature": outcome,
                "feature_type": feature_type,
                "modality": modality,
                "network": network,
                "base_all_n": base_all["n"],
                "base_all_beta_age_std": base_all["beta_age_std"],
                "base_all_p_age": base_all["p_age"],
                "matched_n": matched_base["n"],
                "matched_base_beta_age_std": matched_base["beta_age_std"],
                "matched_base_p_age": matched_base["p_age"],
                "core_n": core["n"],
                "core_beta_age_std": core["beta_age_std"],
                "core_se_age_std": core["se_age_std"],
                "core_ci_low_std": core["ci_low_std"],
                "core_ci_high_std": core["ci_high_std"],
                "core_p_age": core["p_age"],
                "core_beta_age_raw_per_year": core["beta_age_raw_per_year"],
                **comp,
                "core_covariates": ";".join(core_covariates),
            })

            if modality == "eeg":
                spectral_variables = [outcome, "age_mid", *eeg_full]
                spectral_mask = table[spectral_variables].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
                core_same = fit_age_model(table, outcome, eeg_core, spectral_mask)
                spectral = fit_age_model(table, outcome, eeg_full, spectral_mask)
                comp2 = comparison_fields(float(core_same["beta_age_std"]), float(spectral["beta_age_std"]), args.substantive_beta_threshold)
                spectral_rows.append({
                    "feature": outcome,
                    "feature_type": feature_type,
                    "network": network,
                    "core_n": core_same["n"],
                    "core_beta_age_std": core_same["beta_age_std"],
                    "core_p_age": core_same["p_age"],
                    "spectral_n": spectral["n"],
                    "spectral_beta_age_std": spectral["beta_age_std"],
                    "spectral_se_age_std": spectral["se_age_std"],
                    "spectral_ci_low_std": spectral["ci_low_std"],
                    "spectral_ci_high_std": spectral["ci_high_std"],
                    "spectral_p_age": spectral["p_age"],
                    **{f"core_to_spectral_{k}": v for k, v in comp2.items()},
                    "spectral_covariates": ";".join(eeg_full),
                })

    core_results = pd.DataFrame(core_rows)
    core_results["base_all_q_age"] = bh_adjust(core_results["base_all_p_age"])
    core_results["matched_base_q_age"] = bh_adjust(core_results["matched_base_p_age"])
    core_results["core_q_age"] = bh_adjust(core_results["core_p_age"])
    core_results.to_csv(outdir / "age_associations_core.csv", index=False)

    spectral_results = pd.DataFrame(spectral_rows)
    spectral_results["core_q_age"] = bh_adjust(spectral_results["core_p_age"])
    spectral_results["spectral_q_age"] = bh_adjust(spectral_results["spectral_p_age"])
    spectral_results.to_csv(outdir / "age_associations_spectral_sensitivity.csv", index=False)

    # Gamma sensitivity.
    gamma_rows = []
    hf = pd.to_numeric(table["hf_log_relative_p90"], errors="coerce")
    threshold = float(hf.dropna().quantile(args.high_emg_quantile))
    low_emg = hf < threshold
    for network in range(1, 13):
        outcome = f"gamma_r{network:02d}"
        core = fit_age_model(table, outcome, eeg_core)
        spectral = fit_age_model(table, outcome, eeg_full)
        excluded = fit_age_model(table, outcome, eeg_full, low_emg)
        gamma_rows.append({
            "network": network,
            "hf_threshold": threshold,
            "high_emg_quantile": args.high_emg_quantile,
            "core_n": core["n"],
            "core_beta_age_std": core["beta_age_std"],
            "core_p_age": core["p_age"],
            "spectral_n": spectral["n"],
            "spectral_beta_age_std": spectral["beta_age_std"],
            "spectral_p_age": spectral["p_age"],
            "excluded_n": excluded["n"],
            "excluded_beta_age_std": excluded["beta_age_std"],
            "excluded_p_age": excluded["p_age"],
        })
    gamma = pd.DataFrame(gamma_rows)
    gamma["core_q_age"] = bh_adjust(gamma["core_p_age"])
    gamma["spectral_q_age"] = bh_adjust(gamma["spectral_p_age"])
    gamma["excluded_q_age"] = bh_adjust(gamma["excluded_p_age"])
    gamma.to_csv(outdir / "gamma_combined_spectral_sensitivity.csv", index=False)

    # Centroid below 30 Hz.
    centroid_rows = []
    for network in range(1, 13):
        outcome = f"centroid_lt30_r{network:02d}"
        core = fit_age_model(table, outcome, eeg_core)
        spectral = fit_age_model(table, outcome, eeg_full)
        centroid_rows.append({
            "network": network,
            "core_n": core["n"],
            "core_beta_age_std": core["beta_age_std"],
            "core_p_age": core["p_age"],
            "spectral_n": spectral["n"],
            "spectral_beta_age_std": spectral["beta_age_std"],
            "spectral_p_age": spectral["p_age"],
        })
    centroid = pd.DataFrame(centroid_rows)
    centroid["core_q_age"] = bh_adjust(centroid["core_p_age"])
    centroid["spectral_q_age"] = bh_adjust(centroid["spectral_p_age"])
    centroid.to_csv(outdir / "centroid_below30_sensitivity.csv", index=False)

    # Coordination with modality-specific residualization.
    coordination_rows = []
    fmri_resid_cov = list(dict.fromkeys(["age_mid", *fmri_core]))
    eeg_core_resid_cov = list(dict.fromkeys(["age_mid", *eeg_core]))
    eeg_full_resid_cov = list(dict.fromkeys(["age_mid", *eeg_full]))

    for network in range(1, 13):
        f_feature = f"fmri_r{network:02d}"
        for eeg_type in EEG_FEATURE_TYPES:
            e_feature = f"{eeg_type}_r{network:02d}"
            raw = complete_cases(table, [f_feature, e_feature])
            raw_r, raw_p = stats.pearsonr(raw[f_feature], raw[e_feature])

            core_variables = list(dict.fromkeys([f_feature, e_feature, *fmri_resid_cov, *eeg_core_resid_cov]))
            core_data = complete_cases(table, core_variables)
            f_resid = residuals(core_data, f_feature, fmri_resid_cov)
            e_resid = residuals(core_data, e_feature, eeg_core_resid_cov)
            core_r, core_p = permutation_correlation(f_resid, e_resid, args.n_perm, rng)

            spectral_variables = list(dict.fromkeys([f_feature, e_feature, *fmri_resid_cov, *eeg_full_resid_cov]))
            spectral_data = complete_cases(table, spectral_variables)
            f_resid2 = residuals(spectral_data, f_feature, fmri_resid_cov)
            e_resid2 = residuals(spectral_data, e_feature, eeg_full_resid_cov)
            spectral_r, spectral_p = permutation_correlation(f_resid2, e_resid2, args.n_perm, rng)

            coordination_rows.append({
                "network": network,
                "eeg_feature_type": eeg_type,
                "fmri_feature": f_feature,
                "eeg_feature": e_feature,
                "raw_n": len(raw),
                "raw_r": raw_r,
                "raw_p": raw_p,
                "core_n": len(core_data),
                "core_partial_r": core_r,
                "core_permutation_p": core_p,
                "spectral_n": len(spectral_data),
                "spectral_partial_r": spectral_r,
                "spectral_permutation_p": spectral_p,
            })

    coordination = pd.DataFrame(coordination_rows)
    coordination["raw_q"] = bh_adjust(coordination["raw_p"])
    coordination["core_q"] = bh_adjust(coordination["core_permutation_p"])
    coordination["spectral_q"] = bh_adjust(coordination["spectral_permutation_p"])
    coordination.to_csv(outdir / "residual_crossmodal_correlations.csv", index=False)

    # Multivariate full-versus-reduced tests.
    mv_core_cov = usable_covariates(table, list(dict.fromkeys([*fmri_core, *eeg_core])))
    mv_spectral_cov = usable_covariates(table, list(dict.fromkeys([*fmri_core, *eeg_core, *eeg_spectral])))
    mv_core = run_multivariate_family(table, mv_core_cov, args.n_perm, rng, "core")
    mv_spectral = run_multivariate_family(table, mv_spectral_cov, args.n_perm, rng, "spectral_sensitivity")
    pd.concat([mv_core, mv_spectral], ignore_index=True).to_csv(
        outdir / "multivariate_age_tests.csv", index=False
    )

    substantive = core_results[core_results["substantive_base"] == True]
    base_sig = core_results[core_results["matched_base_q_age"] < 0.05]

    strongest = coordination.reindex(
        coordination["core_partial_r"].abs().sort_values(ascending=False).index
    ).head(12)

    lines = [
        "AGE-ASSOCIATION ROBUSTNESS SUMMARY",
        "=" * 72,
        f"Feature-level tests: {len(core_results)}",
        f"Core-adjusted q<0.05: {int((core_results['core_q_age'] < 0.05).sum())}",
        f"Core sign-stable among all features: {int(core_results['sign_stable'].sum())}/{len(core_results)}",
        f"Core sign-stable among |base beta|>=0.10: {int(substantive['sign_stable'].sum())}/{len(substantive)}",
        f"Core sign-stable among base q<0.05: {int(base_sig['sign_stable'].sum())}/{len(base_sig)}",
        f"Primary coordination q<0.05: {int((coordination['core_q'] < 0.05).sum())}/{len(coordination)}",
        f"Spectral-sensitivity coordination q<0.05: {int((coordination['spectral_q'] < 0.05).sum())}/{len(coordination)}",
        f"Core multivariate age tests q<0.05: {int((mv_core['q_age'] < 0.05).sum())}/12",
        f"Spectral-sensitivity multivariate age tests q<0.05: {int((mv_spectral['q_age'] < 0.05).sum())}/12",
        "",
        "Strongest primary adjusted coordination results",
        strongest[["network", "eeg_feature_type", "core_partial_r", "core_q"]].to_string(
            index=False, float_format=lambda x: f"{x:.4f}"
        ),
        "",
        "Core multivariate age tests",
        mv_core[["network", "n", "pillai_trace_age", "permutation_p_age", "q_age", "age_vector_l2"]].to_string(
            index=False, float_format=lambda x: f"{x:.4f}"
        ),
    ]
    summary = outdir / "age_association_summary.txt"
    summary.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\nWrote age-association results to:", outdir)


if __name__ == "__main__":
    main()
