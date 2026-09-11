#!/usr/bin/env python3
"""Summarize held-out prediction results and paired model comparisons."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def pearson_safe(y: np.ndarray, prediction: np.ndarray) -> float:
    if len(y) < 3 or np.std(y) <= 1e-12 or np.std(prediction) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(y, prediction)[0, 1])


def metrics(y: np.ndarray, prediction: np.ndarray) -> Dict[str, float]:
    return {
        "r2": float(r2_score(y, prediction)),
        "mae": float(mean_absolute_error(y, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(y, prediction))),
        "pearson_r": pearson_safe(y, prediction),
        "mean_error": float(np.mean(prediction - y)),
    }


def paired_table(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    keys = ["condition", "rank", "subject_id"]
    left = a[keys + ["observed", "predicted"]].rename(columns={"predicted": "pred_a"})
    right = b[keys + ["predicted"]].rename(columns={"predicted": "pred_b"})
    return left.merge(right, on=keys, how="inner")




def paired_table_across_ranks(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    keys = ["condition", "subject_id"]
    left = a[keys + ["observed", "predicted"]].rename(columns={"predicted": "pred_a"})
    right = b[keys + ["predicted"]].rename(columns={"predicted": "pred_b"})
    return left.merge(right, on=keys, how="inner")

def bootstrap_difference(
    paired: pd.DataFrame,
    n_boot: int,
    seed: int,
) -> Dict[str, float]:
    if paired.empty:
        return {}
    y = paired["observed"].to_numpy(float)
    pred_a = paired["pred_a"].to_numpy(float)
    pred_b = paired["pred_b"].to_numpy(float)
    observed = metrics(y, pred_a)
    reference = metrics(y, pred_b)

    rng = np.random.default_rng(seed)
    n = len(paired)
    deltas: List[Tuple[float, float, float]] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        if np.std(yb) <= 1e-12:
            continue
        ma = metrics(yb, pred_a[idx])
        mb = metrics(yb, pred_b[idx])
        deltas.append((ma["r2"] - mb["r2"], ma["mae"] - mb["mae"], ma["rmse"] - mb["rmse"]))
    values = np.asarray(deltas, float)

    out = {
        "n_subjects": int(n),
        "delta_r2_a_minus_b": observed["r2"] - reference["r2"],
        "delta_mae_a_minus_b": observed["mae"] - reference["mae"],
        "delta_rmse_a_minus_b": observed["rmse"] - reference["rmse"],
    }
    if values.size:
        for index, name in enumerate(("r2", "mae", "rmse")):
            out[f"delta_{name}_ci_low"] = float(np.quantile(values[:, index], 0.025))
            out[f"delta_{name}_ci_high"] = float(np.quantile(values[:, index], 0.975))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize leakage-controlled prediction")
    parser.add_argument("--prediction_csvs", nargs="+", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--n_boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    frames = [
        pd.read_csv(Path(path).expanduser().resolve(), dtype={"subject_id": str})
        for path in args.prediction_csvs
    ]
    predictions = pd.concat(frames, ignore_index=True)
    output = Path(args.outdir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output / "all_predictions.csv", index=False)

    performance_rows: List[Dict[str, Any]] = []
    group_columns = ["condition", "rank", "model", "factor_set", "n_factors"]
    for key, group in predictions.groupby(group_columns, dropna=False):
        row = dict(zip(group_columns, key))
        row["n_subjects"] = int(group["subject_id"].nunique())
        row.update(metrics(group["observed"].to_numpy(float), group["predicted"].to_numpy(float)))
        performance_rows.append(row)
    performance = pd.DataFrame(performance_rows).sort_values(group_columns)
    performance.to_csv(output / "prediction_performance.csv", index=False)

    fold_rows: List[Dict[str, Any]] = []
    fold_columns = group_columns + ["outer_fold"]
    for key, group in predictions.groupby(fold_columns, dropna=False):
        row = dict(zip(fold_columns, key))
        row["n_subjects"] = int(len(group))
        row.update(metrics(group["observed"].to_numpy(float), group["predicted"].to_numpy(float)))
        fold_rows.append(row)
    pd.DataFrame(fold_rows).sort_values(fold_columns).to_csv(
        output / "fold_performance.csv", index=False
    )

    comparison_rows: List[Dict[str, Any]] = []
    planned = [
        ("Mfused", "Mconcat", "shared_vs_separate_multimodal"),
        ("Mfused", "Mf", "EEG_increment_given_fMRI"),
        ("Mfused", "Me", "fMRI_increment_given_EEG"),
    ]
    for (condition, rank, factor_set, n_factors), block in predictions.loc[
        predictions["model"] != "M0"
    ].groupby(["condition", "rank", "factor_set", "n_factors"]):
        baseline = predictions.loc[
            (predictions["condition"] == condition)
            & (predictions["rank"] == rank)
            & (predictions["model"] == "M0")
        ]
        for model in ("Mf", "Me", "Mfused", "Mconcat"):
            a = block.loc[block["model"] == model]
            if a.empty:
                continue
            result = bootstrap_difference(
                paired_table(a, baseline), args.n_boot, args.seed + int(rank) + int(n_factors)
            )
            result.update(
                {
                    "condition": condition,
                    "rank": rank,
                    "factor_set": factor_set,
                    "n_factors": n_factors,
                    "comparison": f"{model}_vs_M0",
                    "model_a": model,
                    "model_b": "M0",
                }
            )
            comparison_rows.append(result)

        for model_a, model_b, label in planned:
            a = block.loc[block["model"] == model_a]
            b = block.loc[block["model"] == model_b]
            if a.empty or b.empty:
                continue
            result = bootstrap_difference(
                paired_table(a, b),
                args.n_boot,
                args.seed + 1000 + int(rank) + int(n_factors),
            )
            result.update(
                {
                    "condition": condition,
                    "rank": rank,
                    "factor_set": factor_set,
                    "n_factors": n_factors,
                    "comparison": label,
                    "model_a": model_a,
                    "model_b": model_b,
                }
            )
            comparison_rows.append(result)

    comparisons = pd.DataFrame(comparison_rows)
    comparisons.to_csv(output / "paired_model_comparisons.csv", index=False)

    # Rank 12 versus rank 7 on the same condition, model, and factor-set label.
    rank_rows: List[Dict[str, Any]] = []
    if {7, 12}.issubset(set(predictions["rank"].astype(int))):
        for (condition, model, factor_set), block in predictions.groupby(
            ["condition", "model", "factor_set"], dropna=False
        ):
            a = block.loc[block["rank"] == 12]
            b = block.loc[block["rank"] == 7]
            if a.empty or b.empty:
                continue
            result = bootstrap_difference(
                paired_table_across_ranks(a, b), args.n_boot, args.seed + 7000
            )
            result.update(
                {
                    "condition": condition,
                    "model": model,
                    "factor_set": factor_set,
                    "rank_a": 12,
                    "rank_b": 7,
                }
            )
            rank_rows.append(result)
    pd.DataFrame(rank_rows).to_csv(output / "rank_comparisons.csv", index=False)

    print("\nPrediction performance")
    display = performance[["condition", "rank", "model", "factor_set", "r2", "mae", "rmse", "pearson_r"]]
    print(display.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nWrote prediction summaries to {output}")


if __name__ == "__main__":
    main()
