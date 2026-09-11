#!/usr/bin/env python3
"""Run leakage-controlled age prediction for one EEG recording condition and rank.

Within each outer fold, this  performs training-only modality scaling,
training-only fused/EEG-only/fMRI-only dictionary fitting, training-only factor
ordering, fixed-dictionary scoring of held-out subjects, training-only feature
standardization, inner-fold ridge tuning, and held-out prediction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from analysis_utils import (
    align_condition,
    fit_training_model,
    import_fitter,
    load_npz,
    load_phenotypes,
    normalize_subject_id,
    order_factors,
    parse_factor_sets,
    prediction_models,
    scaled_subset,
    score_fixed_dictionary,
    training_scales,
    tune_ridge,
    validate_joint_npz,
)


def parse_float_list(text: str) -> List[float]:
    return [float(piece.strip()) for piece in text.split(",") if piece.strip()]


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        json.dump(payload, stream, indent=2, allow_nan=True)


def save_model_artifact(
    path: Path,
    ordered: Dict[str, Any],
    train_ids: np.ndarray,
    test_ids: np.ndarray,
    eeg_scale: float,
    fmri_scale: float,
    best,
) -> None:
    payload: Dict[str, Any] = {
        "Phi_hat": ordered["Phi"],
        "lambda_eeg_train": ordered["train_eeg"],
        "lambda_eeg_test": ordered["test_eeg"],
        "factor_energy": ordered["energy"],
        "factor_order_zero_based": ordered["order"],
        "train_subject_ids": train_ids,
        "test_subject_ids": test_ids,
        "eeg_training_scale": np.array(eeg_scale),
        "fmri_training_scale": np.array(fmri_scale),
        "J_final": np.array(best.objective),
        "selected_start_method": np.array(best.start_method),
        "selected_start_seed": np.array(best.start_seed),
        "iterations": np.array(best.iterations),
        "last_relative_change": np.array(best.last_relative_change),
        "converged": np.array(best.converged),
    }
    if ordered["train_fmri"] is not None:
        payload["lambda_fmri_train"] = ordered["train_fmri"]
        payload["lambda_fmri_test"] = ordered["test_fmri"]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Leakage-controlled MPI--LEMON age prediction")
    parser.add_argument("--condition", default="EO", choices=["EO", "EC"])
    parser.add_argument("--data_npz", default="")
    parser.add_argument("--phenotype_csv", default="lemon_subject_metadata.csv")
    parser.add_argument("--fold_csv", default="prediction_outer_folds.csv")
    parser.add_argument("--fitter", default="shared_spatial_model.py")
    parser.add_argument("--outdir", default="prediction_results")
    parser.add_argument("--rank", default=12, type=int)
    parser.add_argument("--factor_sets", default="all")
    parser.add_argument("--ridge_alphas", default="0.01,0.1,1,10,100,1000")
    parser.add_argument("--inner_folds", type=int, default=5)
    parser.add_argument("--n_starts", type=int, default=3)
    parser.add_argument("--alpha_lambda", type=float, default=1.0)
    parser.add_argument("--alpha0", type=float, default=1e-6)
    parser.add_argument("--step_phi", type=float, default=1e-2)
    parser.add_argument("--max_iter", type=int, default=30)
    parser.add_argument("--tol", type=float, default=1e-7)
    parser.add_argument("--fit_n_jobs", type=int, default=4)
    parser.add_argument("--lam_solver", choices=["bvls", "lsq"], default="bvls")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--only_fold", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.data_npz:
        args.data_npz = f"lemon_{args.condition.lower()}_model_inputs.npz"

    output = Path(args.outdir).expanduser().resolve() / args.condition / f"R{args.rank}"
    output.mkdir(parents=True, exist_ok=True)
    factor_sets = parse_factor_sets(args.factor_sets, args.rank)
    alphas = parse_float_list(args.ridge_alphas)

    data = load_npz(args.data_npz)
    info = validate_joint_npz(data)
    phenotype = load_phenotypes(args.phenotype_csv, outcome="age_mid", sex_column="male")
    folds = pd.read_csv(
        Path(args.fold_csv).expanduser().resolve(), dtype={"subject_id": str}
    )
    folds["subject_id"] = folds["subject_id"].map(normalize_subject_id)
    available = set(info["subject_ids"].tolist())
    folds = folds.loc[folds["subject_id"].isin(available)].copy()
    folds = folds.sort_values("subject_id").reset_index(drop=True)
    aligned = align_condition(data, phenotype, folds["subject_id"].tolist())

    subject_ids = aligned["subject_ids"]
    npz_indices = aligned["npz_indices"]
    y = aligned["outcome"]
    male = aligned["male"]
    fold_by_subject = dict(zip(folds["subject_id"], folds["outer_fold"].astype(int)))
    outer_fold = np.asarray([fold_by_subject[sid] for sid in subject_ids], dtype=int)

    fitter = import_fitter(args.fitter)
    all_predictions: List[pd.DataFrame] = []
    all_inner: List[pd.DataFrame] = []
    all_diagnostics: List[pd.DataFrame] = []

    for fold in sorted(np.unique(outer_fold)):
        if args.only_fold and fold != args.only_fold:
            continue
        fold_dir = output / f"fold_{fold:02d}"
        prediction_path = fold_dir / "predictions.csv"
        if args.resume and prediction_path.exists() and not args.overwrite:
            print(f"Skipping completed {args.condition} R={args.rank} fold {fold}")
            all_predictions.append(pd.read_csv(prediction_path, dtype={"subject_id": str}))
            inner_path = fold_dir / "inner_cv.csv"
            diagnostics_path = fold_dir / "fit_diagnostics.csv"
            if inner_path.exists():
                all_inner.append(pd.read_csv(inner_path))
            if diagnostics_path.exists():
                all_diagnostics.append(pd.read_csv(diagnostics_path))
            continue

        test_position = np.where(outer_fold == fold)[0]
        train_position = np.where(outer_fold != fold)[0]
        train_npz = npz_indices[train_position]
        test_npz = npz_indices[test_position]

        C_eeg = np.asarray(data["C_eeg"], float)
        C_fmri = np.asarray(data["C_fmri"], float)
        W_eeg = np.asarray(data["W_eeg"], float)
        W_fmri = np.asarray(data["W_fmri"], float)
        eeg_scale, fmri_scale = training_scales(
            C_eeg[train_npz], C_fmri[train_npz], W_eeg, W_fmri
        )
        train_data = scaled_subset(data, train_npz, eeg_scale, fmri_scale)
        test_data = scaled_subset(data, test_npz, eeg_scale, fmri_scale)

        fold_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            fold_dir / "fold_definition.json",
            {
                "condition": args.condition,
                "rank": args.rank,
                "outer_fold": int(fold),
                "n_train": int(len(train_position)),
                "n_test": int(len(test_position)),
                "train_subject_ids": subject_ids[train_position].tolist(),
                "test_subject_ids": subject_ids[test_position].tolist(),
                "eeg_training_scale": eeg_scale,
                "fmri_training_scale": fmri_scale,
                "factor_sets": [label for label, _ in factor_sets],
            },
        )

        ordered: Dict[str, Dict[str, Any]] = {}
        diagnostic_rows: List[Dict[str, Any]] = []
        for model_number, mode in enumerate(("fused", "eeg_only", "fmri_only"), start=1):
            model_dir = fold_dir / mode
            best, starts = fit_training_model(
                fitter=fitter,
                data=train_data,
                rank=args.rank,
                mode=mode,
                seed=args.seed + 1000 * fold + 10 * model_number,
                n_starts=args.n_starts,
                alpha_lambda=args.alpha_lambda,
                alpha0=args.alpha0,
                step_phi=args.step_phi,
                max_iter=args.max_iter,
                tolerance=args.tol,
                n_jobs=args.fit_n_jobs,
                solver=args.lam_solver,
                workdir=model_dir / "starts",
            )
            Phi = np.asarray(best.fit["Phi_hat"], float)
            train_eeg, train_fmri = score_fixed_dictionary(
                fitter,
                train_data,
                Phi,
                args.rank,
                mode,
                args.alpha_lambda,
                args.alpha0,
                args.fit_n_jobs,
                args.lam_solver,
            )
            test_eeg, test_fmri = score_fixed_dictionary(
                fitter,
                test_data,
                Phi,
                args.rank,
                mode,
                args.alpha_lambda,
                args.alpha0,
                args.fit_n_jobs,
                args.lam_solver,
            )
            ordered[mode] = order_factors(
                Phi, train_eeg, train_fmri, test_eeg, test_fmri, mode
            )
            save_model_artifact(
                model_dir / "training_fit_and_fixed_scores.npz",
                ordered[mode],
                subject_ids[train_position],
                subject_ids[test_position],
                eeg_scale,
                fmri_scale,
                best,
            )
            for start_index, result in enumerate(starts, start=1):
                diagnostic_rows.append(
                    {
                        "condition": args.condition,
                        "rank": args.rank,
                        "outer_fold": fold,
                        "fit_mode": mode,
                        "start_index": start_index,
                        "start_method": result.start_method,
                        "start_seed": result.start_seed,
                        "objective": result.objective,
                        "iterations": result.iterations,
                        "last_relative_change": result.last_relative_change,
                        "converged": result.converged,
                        "selected": bool(result is best),
                    }
                )

        y_train = y[train_position]
        y_test = y[test_position]
        male_train = male[train_position]
        male_test = male[test_position]
        omega = np.asarray(data["omega"], float)

        prediction_rows: List[Dict[str, Any]] = []
        inner_tables: List[pd.DataFrame] = []

        # M0 is independent of the retained factor count and is fit once per fold.
        X0_train = male_train
        X0_test = male_test
        alpha0, inner0, split0 = tune_ridge(
            X0_train,
            y_train,
            male_train,
            alphas,
            args.inner_folds,
            args.seed + 10000 + fold,
        )
        model0 = Pipeline(
            [("standardize", StandardScaler()), ("ridge", Ridge(alpha=alpha0))]
        )
        model0.fit(X0_train, y_train)
        pred0 = model0.predict(X0_test)
        inner0["condition"] = args.condition
        inner0["rank"] = args.rank
        inner0["outer_fold"] = fold
        inner0["model"] = "M0"
        inner0["factor_set"] = "none"
        inner0["split_method"] = split0
        inner_tables.append(inner0)
        for row_index, position in enumerate(test_position):
            prediction_rows.append(
                {
                    "condition": args.condition,
                    "rank": args.rank,
                    "outer_fold": fold,
                    "subject_id": subject_ids[position],
                    "observed": float(y_test[row_index]),
                    "predicted": float(pred0[row_index]),
                    "model": "M0",
                    "factor_set": "none",
                    "n_factors": 0,
                    "n_brain_features": 0,
                    "ridge_alpha": alpha0,
                }
            )

        for factor_label, k in factor_sets:
            models = prediction_models(ordered, omega, k, male_train, male_test)
            for model_name in ("Mf", "Me", "Mfused", "Mconcat"):
                X_train, X_test, n_brain = models[model_name]
                selected_alpha, inner, split_method = tune_ridge(
                    X_train,
                    y_train,
                    male_train,
                    alphas,
                    args.inner_folds,
                    args.seed + 100000 * fold + 100 * k + {"Mf": 1, "Me": 2, "Mfused": 3, "Mconcat": 4}[model_name],
                )
                model = Pipeline(
                    [("standardize", StandardScaler()), ("ridge", Ridge(alpha=selected_alpha))]
                )
                model.fit(X_train, y_train)
                prediction = model.predict(X_test)

                inner["condition"] = args.condition
                inner["rank"] = args.rank
                inner["outer_fold"] = fold
                inner["model"] = model_name
                inner["factor_set"] = factor_label
                inner["n_factors"] = k
                inner["split_method"] = split_method
                inner_tables.append(inner)

                for row_index, position in enumerate(test_position):
                    prediction_rows.append(
                        {
                            "condition": args.condition,
                            "rank": args.rank,
                            "outer_fold": fold,
                            "subject_id": subject_ids[position],
                            "observed": float(y_test[row_index]),
                            "predicted": float(prediction[row_index]),
                            "model": model_name,
                            "factor_set": factor_label,
                            "n_factors": k,
                            "n_brain_features": n_brain,
                            "ridge_alpha": selected_alpha,
                        }
                    )

        fold_predictions = pd.DataFrame(prediction_rows)
        fold_inner = pd.concat(inner_tables, ignore_index=True)
        fold_diagnostics = pd.DataFrame(diagnostic_rows)
        fold_predictions.to_csv(prediction_path, index=False)
        fold_inner.to_csv(fold_dir / "inner_cv.csv", index=False)
        fold_diagnostics.to_csv(fold_dir / "fit_diagnostics.csv", index=False)
        all_predictions.append(fold_predictions)
        all_inner.append(fold_inner)
        all_diagnostics.append(fold_diagnostics)
        print(
            f"Completed {args.condition} R={args.rank} fold {fold}: "
            f"n_train={len(train_position)}, n_test={len(test_position)}"
        )

    if not all_predictions:
        raise RuntimeError("No outer folds were completed")
    pd.concat(all_predictions, ignore_index=True).drop_duplicates(
        ["condition", "rank", "outer_fold", "subject_id", "model", "factor_set"],
        keep="last",
    ).to_csv(output / "all_predictions.csv", index=False)
    if all_inner:
        pd.concat(all_inner, ignore_index=True).drop_duplicates(
            ["condition", "rank", "outer_fold", "model", "factor_set", "alpha", "inner_fold"],
            keep="last",
        ).to_csv(output / "inner_cv_results.csv", index=False)
    if all_diagnostics:
        pd.concat(all_diagnostics, ignore_index=True).drop_duplicates(
            ["condition", "rank", "outer_fold", "fit_mode", "start_index"],
            keep="last",
        ).to_csv(output / "fit_diagnostics.csv", index=False)

    write_json(
        output / "analysis_configuration.json",
        {
            "condition": args.condition,
            "rank": args.rank,
            "outcome": "age_mid",
            "covariate": "male",
            "factor_sets": [{"label": label, "n_factors": k} for label, k in factor_sets],
            "ridge_alphas": alphas,
            "outer_folds": int(len(np.unique(outer_fold))),
            "inner_folds": args.inner_folds,
            "n_starts": args.n_starts,
            "alpha_lambda": args.alpha_lambda,
            "alpha0": args.alpha0,
            "step_phi": args.step_phi,
            "max_iter": args.max_iter,
            "tolerance": args.tol,
            "K": info["K"],
            "F": info["F"],
            "n_subjects": int(len(subject_ids)),
            "feature_definition": "fMRI strength; EEG theta/alpha/beta/gamma band masses and spectral center of mass",
            "models": {
                "M0": "male only",
                "Mf": "sex plus fMRI strengths from the shared fit",
                "Me": "sex plus EEG summaries from the shared fit",
                "Mfused": "male plus shared-dictionary fMRI and EEG features",
                "Mconcat": "sex plus features from independently learned fMRI-only and EEG-only representations",
            },
        },
    )
    print(f"Wrote {output / 'all_predictions.csv'}")


if __name__ == "__main__":
    main()
