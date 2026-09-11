#!/usr/bin/env python3
"""PCA-regularized CCA benchmark for EO, R=12, using independently fitted modality-specific factors.

The script only reads completed prediction analysis outputs. (It does not modify or rerun
the fused, EEG-only, or fMRI-only decompositions.)

For each outer fold, it reconstructs the all-factor EEG-only and fMRI-only
feature blocks used by Mconcat. Standardization, PCA, CCA, and ridge regression
are fitted inside every inner-training split. The selected  pipeline is then refitted on the complete outer-training fold and applied once
to the held-out subjects.
"""
from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from analysis_utils import (
    eeg_summary_features,
    fmri_features,
    inner_splits,
    load_npz,
    load_phenotypes,
    normalize_subject_id,
)
from prediction_summary import bootstrap_difference, metrics

Array = np.ndarray


def parse_floats(text: str) -> List[float]:
    values = sorted({float(x.strip()) for x in text.split(",") if x.strip()})
    if not values or any(x < 0 for x in values):
        raise ValueError("Ridge alphas must be a nonempty list of nonnegative values")
    return values


def parse_ints(text: str) -> List[int]:
    values = sorted({int(x.strip()) for x in text.split(",") if x.strip()})
    if not values or any(x < 1 for x in values):
        raise ValueError("CCA component counts must be positive integers")
    return values


def subject_ids(values: Array) -> Array:
    return np.asarray([normalize_subject_id(x) for x in np.asarray(values).reshape(-1)], object)


@dataclass
class FoldData:
    fold: int
    train_ids: Array
    test_ids: Array
    eeg_train: Array
    eeg_test: Array
    fmri_train: Array
    fmri_test: Array
    y_train: Array
    y_test: Array
    male_train: Array
    male_test: Array


def load_fold(prediction_dir: Path, fold: int, omega: Array, phenotype: pd.DataFrame) -> FoldData:
    fold_dir = prediction_dir / f"fold_{fold:02d}"
    eeg_path = fold_dir / "eeg_only" / "training_fit_and_fixed_scores.npz"
    fmri_path = fold_dir / "fmri_only" / "training_fit_and_fixed_scores.npz"
    for path in (eeg_path, fmri_path):
        if not path.exists():
            raise FileNotFoundError(path)

    eeg = load_npz(eeg_path)
    fmri = load_npz(fmri_path)
    train_ids = subject_ids(eeg["train_subject_ids"])
    test_ids = subject_ids(eeg["test_subject_ids"])
    if not np.array_equal(train_ids, subject_ids(fmri["train_subject_ids"])):
        raise ValueError(f"Training subject order differs between views in fold {fold}")
    if not np.array_equal(test_ids, subject_ids(fmri["test_subject_ids"])):
        raise ValueError(f"Test subject order differs between views in fold {fold}")
    missing = [sid for sid in np.r_[train_ids, test_ids] if sid not in phenotype.index]
    if missing:
        raise ValueError(f"Phenotypes missing for {missing[:5]}")

    eeg_train, _ = eeg_summary_features(eeg["lambda_eeg_train"], omega, 12)
    eeg_test, _ = eeg_summary_features(eeg["lambda_eeg_test"], omega, 12)
    fmri_train, _ = fmri_features(fmri["lambda_fmri_train"], 12)
    fmri_test, _ = fmri_features(fmri["lambda_fmri_test"], 12)

    arrays = (eeg_train, eeg_test, fmri_train, fmri_test)
    if not all(np.isfinite(x).all() for x in arrays):
        raise ValueError(f"Nonfinite prediction analysis features in fold {fold}")

    return FoldData(
        fold=fold,
        train_ids=train_ids,
        test_ids=test_ids,
        eeg_train=eeg_train,
        eeg_test=eeg_test,
        fmri_train=fmri_train,
        fmri_test=fmri_test,
        y_train=phenotype.loc[train_ids, "outcome"].to_numpy(float),
        y_test=phenotype.loc[test_ids, "outcome"].to_numpy(float),
        male_train=phenotype.loc[train_ids, "male"].to_numpy(float)[:, None],
        male_test=phenotype.loc[test_ids, "male"].to_numpy(float)[:, None],
    )


@dataclass
class PCACCA:
    eeg_scaler: StandardScaler
    fmri_scaler: StandardScaler
    eeg_pca: PCA
    fmri_pca: PCA
    cca: CCA
    convergence_warnings: int

    def transform(self, eeg: Array, fmri: Array) -> Array:
        eeg_pc = self.eeg_pca.transform(self.eeg_scaler.transform(eeg))
        fmri_pc = self.fmri_pca.transform(self.fmri_scaler.transform(fmri))
        eeg_c, fmri_c = self.cca.transform(eeg_pc, fmri_pc)
        out = np.c_[eeg_c, fmri_c]
        if not np.isfinite(out).all():
            raise ValueError("CCA produced nonfinite scores")
        return out


def fit_pca_cca(
    eeg: Array,
    fmri: Array,
    n_components: int,
    pca_variance: float,
    max_iter: int,
    tol: float,
) -> PCACCA:
    eeg_scaler = StandardScaler().fit(eeg)
    fmri_scaler = StandardScaler().fit(fmri)
    eeg_z = eeg_scaler.transform(eeg)
    fmri_z = fmri_scaler.transform(fmri)
    eeg_pca = PCA(n_components=pca_variance, svd_solver="full").fit(eeg_z)
    fmri_pca = PCA(n_components=pca_variance, svd_solver="full").fit(fmri_z)
    eeg_pc = eeg_pca.transform(eeg_z)
    fmri_pc = fmri_pca.transform(fmri_z)
    available = min(eeg_pc.shape[1], fmri_pc.shape[1], len(eeg_pc) - 1)
    if n_components > available:
        raise ValueError(f"{n_components} CCA components requested; {available} available")

    cca = CCA(n_components=n_components, scale=False, max_iter=max_iter, tol=tol)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        cca.fit(eeg_pc, fmri_pc)
    n_warnings = sum(issubclass(w.category, ConvergenceWarning) for w in caught)
    fitted = PCACCA(eeg_scaler, fmri_scaler, eeg_pca, fmri_pca, cca, n_warnings)
    fitted.transform(eeg, fmri)  # finite-value check
    return fitted


def pooled_row(y: Array, prediction: Array) -> Dict[str, float]:
    residual = y - prediction
    sse = float(np.sum(residual**2))
    sst = float(np.sum((y - y.mean()) ** 2))
    return {
        "n_valid": int(len(y)),
        "sse": sse,
        "mse": float(np.mean(residual**2)),
        "r2": float(1.0 - sse / max(sst, 1e-12)),
    }


def tune_pipeline(
    data: FoldData,
    components: Sequence[int],
    alphas: Sequence[float],
    inner_folds: int,
    seed: int,
    pca_variance: float,
    cca_max_iter: int,
    cca_tol: float,
) -> Tuple[int, float, pd.DataFrame, str]:
    splits, split_method = inner_splits(data.y_train, data.male_train, inner_folds, seed)
    fold_rows: List[Dict[str, object]] = []
    pooled_rows: List[Dict[str, object]] = []

    for n_comp in components:
        predictions = {alpha: np.full(len(data.y_train), np.nan) for alpha in alphas}
        fold_r2 = {alpha: [] for alpha in alphas}
        valid_component = True

        for inner_fold, (tr, va) in enumerate(splits, start=1):
            try:
                transform = fit_pca_cca(
                    data.eeg_train[tr], data.fmri_train[tr], n_comp,
                    pca_variance, cca_max_iter, cca_tol,
                )
            except (ValueError, np.linalg.LinAlgError):
                valid_component = False
                break

            Z_tr = transform.transform(data.eeg_train[tr], data.fmri_train[tr])
            Z_va = transform.transform(data.eeg_train[va], data.fmri_train[va])
            X_tr = np.c_[data.male_train[tr], Z_tr]
            X_va = np.c_[data.male_train[va], Z_va]

            for alpha in alphas:
                model = Pipeline([
                    ("standardize", StandardScaler()),
                    ("ridge", Ridge(alpha=alpha)),
                ]).fit(X_tr, data.y_train[tr])
                pred = model.predict(X_va)
                predictions[alpha][va] = pred
                m = pooled_row(data.y_train[va], pred)
                fold_r2[alpha].append(m["r2"])
                fold_rows.append({
                    "n_cca_components": n_comp,
                    "alpha": alpha,
                    "inner_fold": inner_fold,
                    **m,
                    "mean_fold_r2": np.nan,
                    "selected": False,
                    "pca_eeg_components": int(transform.eeg_pca.n_components_),
                    "pca_fmri_components": int(transform.fmri_pca.n_components_),
                    "cca_convergence_warnings": transform.convergence_warnings,
                })

        if not valid_component:
            continue
        for alpha in alphas:
            if not np.isfinite(predictions[alpha]).all():
                raise RuntimeError("Incomplete pooled inner-CV predictions")
            m = pooled_row(data.y_train, predictions[alpha])
            pooled_rows.append({
                "n_cca_components": n_comp,
                "alpha": alpha,
                "inner_fold": 0,
                **m,
                "mean_fold_r2": float(np.mean(fold_r2[alpha])),
                "selected": False,
                "pca_eeg_components": np.nan,
                "pca_fmri_components": np.nan,
                "cca_convergence_warnings": np.nan,
            })

    if not pooled_rows:
        raise RuntimeError("No CCA component count was estimable in all inner folds")
    pooled = pd.DataFrame(pooled_rows).sort_values(
        ["mse", "n_cca_components", "alpha"], ascending=[True, True, False]
    )
    best_components = int(pooled.iloc[0]["n_cca_components"])
    best_alpha = float(pooled.iloc[0]["alpha"])
    table = pd.DataFrame(fold_rows + pooled_rows)
    selected = (
        (table["inner_fold"] == 0)
        & (table["n_cca_components"] == best_components)
        & np.isclose(table["alpha"], best_alpha)
    )
    table.loc[selected, "selected"] = True
    return best_components, best_alpha, table, split_method


def fit_outer(data: FoldData, n_components: int, alpha: float, args) -> Tuple[Array, PCACCA]:
    transform = fit_pca_cca(
        data.eeg_train, data.fmri_train, n_components,
        args.pca_variance, args.cca_max_iter, args.cca_tol,
    )
    X_train = np.c_[data.male_train, transform.transform(data.eeg_train, data.fmri_train)]
    X_test = np.c_[data.male_test, transform.transform(data.eeg_test, data.fmri_test)]
    model = Pipeline([
        ("standardize", StandardScaler()),
        ("ridge", Ridge(alpha=alpha)),
    ]).fit(X_train, data.y_train)
    return model.predict(X_test), transform


def load_main_predictions(path: Path) -> Dict[str, pd.DataFrame]:
    df = pd.read_csv(path, dtype={"subject_id": str})
    df["subject_id"] = df["subject_id"].map(normalize_subject_id)
    df["factor_set"] = df["factor_set"].astype(str)
    df = df[(df["condition"] == "EO") & (df["rank"] == 12)]
    out = {}
    for model in ("M0", "Mf", "Me", "Mfused", "Mconcat"):
        factor_set = "none" if model == "M0" else "all"
        part = df[(df["model"] == model) & (df["factor_set"] == factor_set)].copy()
        if part.empty:
            raise ValueError(f"Missing {model}/{factor_set} predictions in {path}")
        out[model] = part.drop_duplicates("subject_id", keep="last")
    return out


def performance(df: pd.DataFrame, model: str) -> Dict[str, object]:
    row = {"condition": "EO", "rank": 12, "model": model,
           "factor_set": "none" if model == "M0" else "all",
           "n_subjects": int(len(df))}
    row.update(metrics(df["observed"].to_numpy(float), df["predicted"].to_numpy(float)))
    return row


def compare(a: pd.DataFrame, b: pd.DataFrame, model_a: str, model_b: str,
            n_boot: int, seed: int) -> Dict[str, object]:
    left = a[["subject_id", "observed", "predicted"]].rename(
        columns={"observed": "observed_a", "predicted": "pred_a"})
    right = b[["subject_id", "observed", "predicted"]].rename(
        columns={"observed": "observed_b", "predicted": "pred_b"})
    paired = left.merge(right, on="subject_id", validate="one_to_one")
    if len(paired) != len(left) or len(paired) != len(right):
        raise ValueError(f"Subject mismatch between {model_a} and {model_b}")
    if not np.allclose(paired["observed_a"], paired["observed_b"], rtol=1e-7, atol=1e-5):
        raise ValueError(f"Observed outcomes differ between {model_a} and {model_b}")
    input_table = pd.DataFrame({
        "observed": paired["observed_a"],
        "pred_a": paired["pred_a"],
        "pred_b": paired["pred_b"],
    })
    out = bootstrap_difference(input_table, n_boot=n_boot, seed=seed)
    out.update({"condition": "EO", "rank": 12, "factor_set": "all",
                "model_a": model_a, "model_b": model_b})
    return out


def write_json(path: Path, value: Dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="PCA-regularized CCA prediction analysis baseline")
    parser.add_argument("--prediction_dir", default="prediction_results/EO/R12")
    parser.add_argument("--data_npz", default="lemon_eo_model_inputs.npz")
    parser.add_argument("--phenotype_csv", default="lemon_subject_metadata.csv")
    parser.add_argument("--main_predictions", default="prediction_results/EO/R12/all_predictions.csv")
    parser.add_argument("--outdir", default="cca_benchmark_results")
    parser.add_argument("--ridge_alphas", default="0.01,0.1,1,10,100,1000")
    parser.add_argument("--cca_components", default="1,2,3,5,7,10")
    parser.add_argument("--pca_variance", type=float, default=0.95)
    parser.add_argument("--inner_folds", type=int, default=5)
    parser.add_argument("--cca_max_iter", type=int, default=5000)
    parser.add_argument("--cca_tol", type=float, default=1e-6)
    parser.add_argument("--n_boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not 0 < args.pca_variance < 1:
        raise ValueError("pca_variance must be between 0 and 1")
    prediction_dir = Path(args.prediction_dir).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    alphas = parse_floats(args.ridge_alphas)
    components = parse_ints(args.cca_components)

    phenotype = load_phenotypes(args.phenotype_csv, outcome="age_mid", sex_column="male")
    phenotype = phenotype.set_index("subject_id")
    omega = np.asarray(load_npz(args.data_npz)["omega"], float).reshape(-1)

    predictions, inner_tables, settings = [], [], []
    for fold in range(1, 6):
        fold_dir = outdir / f"fold_{fold:02d}"
        pred_path = fold_dir / "predictions.csv"
        inner_path = fold_dir / "inner_cv.csv"
        settings_path = fold_dir / "settings.json"
        if args.resume and pred_path.exists() and not args.overwrite:
            print(f"Reusing completed CCA baseline fold {fold}")
            predictions.append(pd.read_csv(pred_path, dtype={"subject_id": str}))
            inner_tables.append(pd.read_csv(inner_path))
            settings.append(json.loads(settings_path.read_text()))
            continue

        data = load_fold(prediction_dir, fold, omega, phenotype)
        inner_seed = args.seed + 100000 * fold + 1204  # same as all-factor Mconcat
        n_comp, alpha, inner, split_method = tune_pipeline(
            data, components, alphas, args.inner_folds, inner_seed,
            args.pca_variance, args.cca_max_iter, args.cca_tol,
        )
        pred, transform = fit_outer(data, n_comp, alpha, args)
        fold_dir.mkdir(parents=True, exist_ok=True)
        fold_predictions = pd.DataFrame({
            "condition": "EO", "rank": 12, "outer_fold": fold,
            "subject_id": data.test_ids, "observed": data.y_test,
            "predicted": pred, "model": "Mcca", "factor_set": "all",
            "n_factors": 12, "n_brain_features": 2 * n_comp,
            "ridge_alpha": alpha, "cca_components": n_comp,
            "pca_variance": args.pca_variance,
        })
        inner.insert(0, "condition", "EO")
        inner.insert(1, "rank", 12)
        inner.insert(2, "outer_fold", fold)
        inner["model"] = "Mcca"
        inner["factor_set"] = "all"
        inner["split_method"] = split_method
        setting = {
            "condition": "EO", "rank": 12, "outer_fold": fold,
            "n_train": len(data.train_ids), "n_test": len(data.test_ids),
            "selected_cca_components": n_comp, "selected_ridge_alpha": alpha,
            "pca_variance": args.pca_variance,
            "outer_fit_eeg_pca_components": int(transform.eeg_pca.n_components_),
            "outer_fit_fmri_pca_components": int(transform.fmri_pca.n_components_),
            "outer_fit_cca_convergence_warnings": transform.convergence_warnings,
            "inner_split_method": split_method, "inner_seed": inner_seed,
        }
        fold_predictions.to_csv(pred_path, index=False, float_format="%.8g")
        inner.to_csv(inner_path, index=False, float_format="%.6g")
        write_json(settings_path, setting)
        predictions.append(fold_predictions)
        inner_tables.append(inner)
        settings.append(setting)
        print(f"Completed CCA fold {fold}: components={n_comp}, alpha={alpha:g}")

    cca = pd.concat(predictions, ignore_index=True).drop_duplicates("subject_id", keep="last")
    if len(cca) != cca["subject_id"].nunique():
        raise RuntimeError("CCA held-out predictions are not one per subject")
    cca.to_csv(outdir / "cca_all_predictions.csv", index=False, float_format="%.8g")
    pd.concat(inner_tables, ignore_index=True).to_csv(
        outdir / "cca_inner_cv_results.csv", index=False, float_format="%.6g")
    pd.DataFrame(settings).sort_values("outer_fold").to_csv(
        outdir / "cca_outer_fold_settings.csv", index=False, float_format="%.6g")

    main_predictions = load_main_predictions(Path(args.main_predictions).expanduser().resolve())
    all_models = {**main_predictions, "Mcca": cca}
    pd.DataFrame([performance(all_models[m], m)
                  for m in ("M0", "Mf", "Me", "Mfused", "Mconcat", "Mcca")]).to_csv(
        outdir / "cca_performance_comparison.csv", index=False, float_format="%.4g")
    pd.DataFrame([
        compare(cca, all_models[m], "Mcca", m, args.n_boot, args.seed + 7000 + i)
        for i, m in enumerate(("Mfused", "Mconcat", "Mf", "Me", "M0"), start=1)
    ]).to_csv(outdir / "cca_paired_comparisons.csv", index=False, float_format="%.4g")

    write_json(outdir / "cca_analysis_configuration.json", {
        "analysis": "PCA-regularized CCA supplementary baseline",
        "condition": "EO", "rank": 12, "factor_set": "all",
        "pca_variance": args.pca_variance,
        "cca_components_grid": components,
        "ridge_alpha_grid": alphas,
        "inner_folds": args.inner_folds,
        "selection_metric": "pooled inner-CV mean squared error",
        "n_boot": args.n_boot, "seed": args.seed,
        "CCA_features": "concatenated EEG-view and fMRI-view canonical scores",
        "leakage_control": "standardization, PCA, CCA, and ridge refit within every inner training split and outer training fold",
    })
    print(f"Wrote CCA baseline outputs to {outdir}")


if __name__ == "__main__":
    main()
