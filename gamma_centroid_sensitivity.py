#!/usr/bin/env python3
"""Targeted gamma and spectral-centroid sensitivity analyses.

Gamma models for each of 12 networks :::  
M_core:
    age + sex + education + retained-window count + channel retention

M_hf:
    M_core + high-frequency sensor-power proxy

M_aperiodic:
    M_core + aperiodic exponent

M_both:
    M_core + high-frequency sensor power + aperiodic exponent

M_core_excluded:
    M_core after excluding participants at or above the prespecified
    high-frequency-power quantile


Centroid sensitivity
--------------------
The ordinary spectral centroid and the centroid restricted to frequencies
below 30 Hz are compared under the same core EEG adjustment model.

 Outcomes, age, and continuous covariates are standardized within each fitted sample; 
binary covariates are not standardized. 

BH q-values are calculated separately within each
12-network model family.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm


CORE_COVARIATES = [
    "male",
    "education_gymnasium",
    "log_n_windows",
    "channel_retention",
]
HF_COVARIATE = "hf_log_relative_p90"
APERIODIC_COVARIATE = "aperiodic_exponent_median"


def bh_adjust(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    result = np.full(p.shape, np.nan, dtype=float)
    valid = np.isfinite(p)
    if not valid.any():
        return result

    values = p[valid]
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    result[valid] = restored
    return result


def standardize_for_model(
    data: pd.DataFrame,
    columns: Sequence[str],
    binary_columns: Sequence[str] = (
        "male",
        "education_gymnasium",
    ),
) -> pd.DataFrame:
    result = data.copy()
    binary = set(binary_columns)

    for column in columns:
        values = pd.to_numeric(result[column], errors="coerce")
        if column in binary:
            result[column] = values
            continue

        sd = values.std(ddof=1)
        result[column] = (
            (values - values.mean()) / sd
            if np.isfinite(sd) and sd > 0
            else values
        )

    return result


def fit_age_model(
    table: pd.DataFrame,
    outcome: str,
    covariates: Sequence[str],
    mask: pd.Series | np.ndarray | None = None,
) -> Dict[str, float | int]:
    variables = [outcome, "age_mid", *covariates]
    data = (
        table.loc[:, variables].copy()
        if mask is None
        else table.loc[mask, variables].copy()
    )

    for variable in variables:
        data[variable] = pd.to_numeric(data[variable], errors="coerce")
    data = data.dropna()

    if len(data) <= len(covariates) + 3:
        return {
            "n": int(len(data)),
            "beta_age_std": np.nan,
            "se_age_std": np.nan,
            "ci_low_std": np.nan,
            "ci_high_std": np.nan,
            "p_age": np.nan,
            "beta_age_raw_per_year": np.nan,
            "r_squared": np.nan,
        }

    standardized = standardize_for_model(
        data,
        [outcome, "age_mid", *covariates],
    )
    X_std = sm.add_constant(
        standardized[["age_mid", *covariates]],
        has_constant="add",
    )
    fit_std = sm.OLS(
        standardized[outcome],
        X_std,
    ).fit(cov_type="HC3")
    ci = fit_std.conf_int().loc["age_mid"]

    X_raw = sm.add_constant(
        data[["age_mid", *covariates]],
        has_constant="add",
    )
    fit_raw = sm.OLS(
        data[outcome],
        X_raw,
    ).fit(cov_type="HC3")

    return {
        "n": int(len(data)),
        "beta_age_std": float(fit_std.params["age_mid"]),
        "se_age_std": float(fit_std.bse["age_mid"]),
        "ci_low_std": float(ci.iloc[0]),
        "ci_high_std": float(ci.iloc[1]),
        "p_age": float(fit_std.pvalues["age_mid"]),
        "beta_age_raw_per_year": float(fit_raw.params["age_mid"]),
        "r_squared": float(fit_std.rsquared),
    }


def comparison_fields(
    reference_beta: float,
    alternative_beta: float,
) -> Dict[str, float | bool]:
    if not np.isfinite(reference_beta) or not np.isfinite(alternative_beta):
        return {
            "sign_stable": np.nan,
            "absolute_beta_change": np.nan,
            "percent_attenuation_if_same_sign": np.nan,
        }

    sign_stable = bool(
        np.sign(reference_beta) == np.sign(alternative_beta)
    )
    attenuation = (
        100.0
        * (1.0 - abs(alternative_beta) / abs(reference_beta))
        if sign_stable and abs(reference_beta) > 1e-12
        else np.nan
    )

    return {
        "sign_stable": sign_stable,
        "absolute_beta_change": abs(
            alternative_beta - reference_beta
        ),
        "percent_attenuation_if_same_sign": attenuation,
    }


def main() -> None:
    from age_association_analysis import build_public_analysis_table

    parser = argparse.ArgumentParser(description='Targeted gamma and spectral-centroid sensitivity analyses.')
    parser.add_argument('--features', default='lemon_eo_subject_features.csv')
    parser.add_argument('--metadata', default='lemon_subject_metadata.csv')
    parser.add_argument('--fitted', default='lemon_eo_fitted_representation.npz')
    parser.add_argument('--model-input', default='lemon_eo_model_inputs.npz')
    parser.add_argument('--outdir', default='gamma_centroid_results')
    parser.add_argument('--high-frequency-quantile', type=float, default=0.90)
    args = parser.parse_args()

    table = build_public_analysis_table(args.features, args.metadata, args.fitted, args.model_input)
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    required = [
        'age_mid', *CORE_COVARIATES, HF_COVARIATE, APERIODIC_COVARIATE,
        *[f'gamma_r{network:02d}' for network in range(1, 13)],
        *[f'centroid_r{network:02d}' for network in range(1, 13)],
        *[f'centroid_lt30_r{network:02d}' for network in range(1, 13)],
    ]
    missing = [x for x in required if x not in table.columns]
    if missing:
        raise KeyError('Required variables missing from public analysis table: ' + ', '.join(missing))

    threshold_variables = ['age_mid', *CORE_COVARIATES, HF_COVARIATE]
    threshold_data = table[threshold_variables].apply(pd.to_numeric, errors='coerce')
    threshold_mask = threshold_data.notna().all(axis=1)
    hf_threshold = float(pd.to_numeric(
        table.loc[threshold_mask, HF_COVARIATE], errors='coerce'
    ).quantile(args.high_frequency_quantile))
    low_hf_mask = threshold_mask & (pd.to_numeric(table[HF_COVARIATE], errors='coerce') < hf_threshold)

    threshold_summary = pd.DataFrame([{
        'hf_proxy': HF_COVARIATE,
        'quantile': args.high_frequency_quantile,
        'threshold': hf_threshold,
        'n_threshold_sample': int(threshold_mask.sum()),
        'n_below_threshold': int(low_hf_mask.sum()),
        'n_excluded': int(threshold_mask.sum() - low_hf_mask.sum()),
    }])
    threshold_summary.to_csv(outdir / 'high_frequency_threshold.csv', index=False)

    gamma_model_specs = {
        'M_core': CORE_COVARIATES,
        'M_hf': [*CORE_COVARIATES, HF_COVARIATE],
        'M_aperiodic': [*CORE_COVARIATES, APERIODIC_COVARIATE],
        'M_both': [*CORE_COVARIATES, HF_COVARIATE, APERIODIC_COVARIATE],
    }

    gamma_rows: List[Dict[str, object]] = []

    for network in range(1, 13):
        outcome = f"gamma_r{network:02d}"

        for model_name, covariates in gamma_model_specs.items():
            result = fit_age_model(
                table,
                outcome,
                covariates,
            )
            gamma_rows.append(
                {
                    "network": network,
                    "outcome": outcome,
                    "model": model_name,
                    "sample": "full_complete_case",
                    "hf_threshold": hf_threshold,
                    "high_emg_quantile":
                        args.high_frequency_quantile,
                    "covariates": ";".join(covariates),
                    **result,
                }
            )

        excluded_result = fit_age_model(
            table,
            outcome,
            CORE_COVARIATES,
            low_hf_mask,
        )
        gamma_rows.append(
            {
                "network": network,
                "outcome": outcome,
                "model": "M_core_excluded",
                "sample": "below_high_emg_threshold",
                "hf_threshold": hf_threshold,
                "high_emg_quantile":
                    args.high_frequency_quantile,
                "covariates": ";".join(CORE_COVARIATES),
                **excluded_result,
            }
        )

    gamma = pd.DataFrame(gamma_rows)

    gamma["q_age"] = np.nan
    for _, index in gamma.groupby("model").groups.items():
        gamma.loc[index, "q_age"] = bh_adjust(
            gamma.loc[index, "p_age"]
        )

    gamma.to_csv(
        outdir / "gamma_targeted_models.csv",
        index=False,
    )

    gamma_wide = gamma.pivot(
        index="network",
        columns="model",
        values=["n", "beta_age_std", "p_age", "q_age"],
    )
    gamma_wide.columns = [
        f"{measure}_{model}"
        for measure, model in gamma_wide.columns
    ]
    gamma_wide = gamma_wide.reset_index()

    for alternative in [
        "M_hf",
        "M_aperiodic",
        "M_both",
        "M_core_excluded",
    ]:
        comparison = pd.DataFrame(
            [
                comparison_fields(
                    row["beta_age_std_M_core"],
                    row[f"beta_age_std_{alternative}"],
                )
                for _, row in gamma_wide.iterrows()
            ]
        )
        gamma_wide[
            f"sign_stable_core_vs_{alternative}"
        ] = comparison["sign_stable"]
        gamma_wide[
            f"absolute_beta_change_core_vs_{alternative}"
        ] = comparison["absolute_beta_change"]
        gamma_wide[
            f"percent_attenuation_core_vs_{alternative}"
        ] = comparison[
            "percent_attenuation_if_same_sign"
        ]

    gamma_wide.to_csv(
        outdir / "gamma_targeted_comparisons.csv",
        index=False,
    )

    centroid_rows: List[Dict[str, object]] = []

    for network in range(1, 13):
        ordinary_outcome = f"centroid_r{network:02d}"
        restricted_outcome = f"centroid_lt30_r{network:02d}"

        matched_variables = [
            ordinary_outcome,
            restricted_outcome,
            "age_mid",
            *CORE_COVARIATES,
        ]
        matched_data = table[matched_variables].apply(
            pd.to_numeric,
            errors="coerce",
        )
        matched_mask = matched_data.notna().all(axis=1)

        ordinary = fit_age_model(
            table,
            ordinary_outcome,
            CORE_COVARIATES,
            matched_mask,
        )
        restricted = fit_age_model(
            table,
            restricted_outcome,
            CORE_COVARIATES,
            matched_mask,
        )
        comparison = comparison_fields(
            float(ordinary["beta_age_std"]),
            float(restricted["beta_age_std"]),
        )

        feature_correlation = float(
            table.loc[
                matched_mask,
                [ordinary_outcome, restricted_outcome],
            ]
            .apply(pd.to_numeric, errors="coerce")
            .corr()
            .iloc[0, 1]
        )

        centroid_rows.append(
            {
                "network": network,
                "n": ordinary["n"],
                "ordinary_beta_age_std":
                    ordinary["beta_age_std"],
                "ordinary_se_age_std":
                    ordinary["se_age_std"],
                "ordinary_ci_low_std":
                    ordinary["ci_low_std"],
                "ordinary_ci_high_std":
                    ordinary["ci_high_std"],
                "ordinary_p_age": ordinary["p_age"],
                "ordinary_beta_age_raw_per_year":
                    ordinary["beta_age_raw_per_year"],
                "lt30_beta_age_std":
                    restricted["beta_age_std"],
                "lt30_se_age_std":
                    restricted["se_age_std"],
                "lt30_ci_low_std":
                    restricted["ci_low_std"],
                "lt30_ci_high_std":
                    restricted["ci_high_std"],
                "lt30_p_age": restricted["p_age"],
                "lt30_beta_age_raw_per_year":
                    restricted["beta_age_raw_per_year"],
                "ordinary_lt30_feature_correlation":
                    feature_correlation,
                **comparison,
            }
        )

    centroid = pd.DataFrame(centroid_rows)
    centroid["ordinary_q_age"] = bh_adjust(
        centroid["ordinary_p_age"]
    )
    centroid["lt30_q_age"] = bh_adjust(
        centroid["lt30_p_age"]
    )
    centroid.to_csv(
        outdir / "centroid_core_comparison.csv",
        index=False,
    )

    lines: List[str] = [
        "TARGETED GAMMA AND CENTROID SENSITIVITY",
        "=" * 78,
        "",
        "HIGH-FREQUENCY EXCLUSION THRESHOLD",
        threshold_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.5f}",
        ),
        "",
        "GAMMA MODEL SUMMARY",
    ]

    model_summary_rows = []
    for model_name, group in gamma.groupby("model"):
        model_summary_rows.append(
            {
                "model": model_name,
                "median_n": int(group["n"].median()),
                "q_lt_0p05": int(
                    (group["q_age"] < 0.05).sum()
                ),
                "median_beta_age_std":
                    group["beta_age_std"].median(),
                "minimum_beta_age_std":
                    group["beta_age_std"].min(),
                "maximum_beta_age_std":
                    group["beta_age_std"].max(),
            }
        )

    model_summary = pd.DataFrame(
        model_summary_rows
    ).sort_values("model")

    lines.append(
        model_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )
    lines.extend(
        [
            "",
            "GAMMA NETWORK-LEVEL RESULTS",
            gamma[
                [
                    "network",
                    "model",
                    "n",
                    "beta_age_std",
                    "p_age",
                    "q_age",
                ]
            ]
            .sort_values(["network", "model"])
            .to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            ),
            "",
            "CENTROID ORDINARY VERSUS BELOW 30 HZ",
            centroid[
                [
                    "network",
                    "n",
                    "ordinary_beta_age_std",
                    "ordinary_q_age",
                    "lt30_beta_age_std",
                    "lt30_q_age",
                    "sign_stable",
                    "absolute_beta_change",
                    "ordinary_lt30_feature_correlation",
                ]
            ].to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            ),
        ]
    )

    summary_path = (
        outdir
        / "gamma_centroid_targeted_summary.txt"
    )
    summary_path.write_text("\n".join(lines) + "\n")

    print("\n".join(lines))
    print("\nWrote results to:", outdir)


if __name__ == "__main__":
    main()
