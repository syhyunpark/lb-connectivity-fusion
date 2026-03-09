#!/usr/bin/env python3
"""
run_synergy_analysis.py

Nested-CV synergy test with ridge, with inner alpha selection per outer fold.

Models (all adjusted for covariates):
  M0     : covars only
  Mf     : covars + fused fMRI-strength block
  Me     : covars + fused EEG spectral-summary block
  Mfe    : covars + fused (fMRI + EEG) blocks 
  Mconcat: covars + (FMRIonly block) + (EEGonly block) 
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


def _is_baseline_block(col: str) -> bool:
    return col.startswith("FMRIonly__") or col.startswith("EEGonly__")


RE_FMRI_FUSED = re.compile(r"^(Lambda_fmri|fMRI_strength|FMRI_strength)_r(\d+)$")
RE_EEG_FUSED = re.compile(
    r"^EEG_(theta|alpha|beta|gamma)(_AUC)?_r(\d+)$|^EEG_(muHz|specCOM)_r(\d+)$"
)

RE_FMRI_ONLY = re.compile(r"^FMRIonly__(FMRI_score|fMRI_strength|FMRI_strength)_r(\d+)$")
RE_EEG_ONLY = re.compile(
    r"^EEGonly__EEG_(theta|alpha|beta|gamma)(_AUC)?_r(\d+)$|^EEGonly__EEG_(muHz|specCOM)_r(\d+)$"
)


def r2(y, yhat):
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    ssr = np.sum((y - yhat) ** 2)
    sst = np.sum((y - y.mean()) ** 2)
    return 1.0 - ssr / (sst + 1e-12)


def kfold_indices(n, k, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, k)
    for j in range(k):
        te = folds[j]
        tr = np.concatenate([folds[t] for t in range(k) if t != j])
        yield tr, te


def standardize_train_apply(Xtr, Xte):
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0, ddof=0)
    sd[sd == 0] = 1.0
    return (Xtr - mu) / sd, (Xte - mu) / sd


def ridge_fit_predict(Xtr, ytr, Xte, alpha):
    ymu = float(np.mean(ytr))
    yc = ytr - ymu
    XtX = Xtr.T @ Xtr
    p = XtX.shape[0]
    beta = np.linalg.solve(XtX + float(alpha) * np.eye(p), Xtr.T @ yc)
    return Xte @ beta + ymu


def build_covariates(df: pd.DataFrame, covars: List[str]) -> np.ndarray:
    parts = []
    for c in covars:
        if c not in df.columns:
            raise ValueError("Missing covariate column")
        s = df[c]
        sn = pd.to_numeric(s, errors="coerce")
        if sn.notna().mean() >= 0.9:
            parts.append(sn.to_numpy(float)[:, None])
        else:
            cat = s.astype("category")
            d = pd.get_dummies(cat, prefix=c, drop_first=True, dummy_na=False)
            if d.shape[1] > 0:
                parts.append(d.to_numpy(float))
    if not parts:
        return np.zeros((len(df), 0), float)
    return np.concatenate(parts, axis=1)


def build_X(df: pd.DataFrame, covars: List[str], feat_cols: List[str]) -> np.ndarray:
    Xc = build_covariates(df, covars)
    Xf = []
    for c in feat_cols:
        Xf.append(pd.to_numeric(df[c], errors="coerce").to_numpy(float)[:, None])
    Xf = np.concatenate(Xf, axis=1) if Xf else np.zeros((len(df), 0), float)
    return np.concatenate([Xc, Xf], axis=1)


def choose_alpha_holdout(Xtr, ytr, alpha_grid, val_frac=0.2, seed=0):
    rng = np.random.default_rng(seed)
    n = len(ytr)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = max(1, int(np.floor(val_frac * n)))
    val = idx[:n_val]
    tr = idx[n_val:]
    if tr.size < 10:
        return float(alpha_grid[len(alpha_grid) // 2])

    X_in_tr, X_in_val = Xtr[tr], Xtr[val]
    y_in_tr, y_in_val = ytr[tr], ytr[val]
    X_in_tr_s, X_in_val_s = standardize_train_apply(X_in_tr, X_in_val)

    best_a = float(alpha_grid[0])
    best = -np.inf
    for a in alpha_grid:
        yhat = ridge_fit_predict(X_in_tr_s, y_in_tr, X_in_val_s, float(a))
        sc = r2(y_in_val, yhat)
        if sc > best:
            best = sc
            best_a = float(a)
    return best_a


def choose_alpha_inner_kfold(Xtr, ytr, alpha_grid, k=3, seed=0):
    n = len(ytr)
    if n < 60:
        return choose_alpha_holdout(Xtr, ytr, alpha_grid, val_frac=0.2, seed=seed)

    scores = {float(a): [] for a in alpha_grid}
    for fold_id, (tr, va) in enumerate(kfold_indices(n, k, seed=seed)):
        X_in_tr, X_in_va = Xtr[tr], Xtr[va]
        y_in_tr, y_in_va = ytr[tr], ytr[va]
        X_in_tr_s, X_in_va_s = standardize_train_apply(X_in_tr, X_in_va)
        for a in alpha_grid:
            yhat = ridge_fit_predict(X_in_tr_s, y_in_tr, X_in_va_s, float(a))
            scores[float(a)].append(r2(y_in_va, yhat))
    mean_scores = {a: float(np.mean(v)) for a, v in scores.items()}
    return float(max(mean_scores, key=lambda a: mean_scores[a]))


def cv_r2_nested_ridge(
    X,
    y,
    k=5,
    seed=0,
    alpha_grid=(0.1, 1, 10, 100),
    inner_method="holdout",
    inner_frac=0.2,
    inner_k=3,
):
    r2s = []
    for fold_id, (tr, te) in enumerate(kfold_indices(len(y), k, seed=seed)):
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]

        if inner_method == "kfold":
            a_star = choose_alpha_inner_kfold(
                Xtr, ytr, alpha_grid, k=inner_k, seed=seed + 1000 + fold_id
            )
        else:
            a_star = choose_alpha_holdout(
                Xtr, ytr, alpha_grid, val_frac=inner_frac, seed=seed + 1000 + fold_id
            )

        Xtr_s, Xte_s = standardize_train_apply(Xtr, Xte)
        yhat = ridge_fit_predict(Xtr_s, ytr, Xte_s, a_star)
        r2s.append(r2(yte, yhat))
    return float(np.mean(r2s)), float(np.std(r2s))


def _keep_topk_networks(cols: List[str], topk: int) -> List[str]:
    if topk <= 0:
        return cols
    kept = []
    for c in cols:
        m = re.search(r"_r(\d+)$", c)
        if m and int(m.group(1)) <= topk:
            kept.append(c)
    return kept


def main():
    ap = argparse.ArgumentParser("Nested-CV synergy analysis with ridge")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--outcomes", required=True, help="comma-separated")
    ap.add_argument("--covars", required=True, help="comma-separated")
    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha_grid", default="0.1,1,10,100")
    ap.add_argument("--inner_method", choices=["holdout", "kfold"], default="holdout")
    ap.add_argument("--inner_frac", type=float, default=0.2)
    ap.add_argument("--inner_k", type=int, default=3)
    ap.add_argument("--topk_networks", type=int, default=0)
    args = ap.parse_args()

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    outcomes = [x.strip() for x in args.outcomes.split(",") if x.strip()]
    covars = [x.strip() for x in args.covars.split(",") if x.strip()]
    alpha_grid = tuple(float(x) for x in args.alpha_grid.split(",") if x.strip())

    fmri_fused_cols: List[str] = []
    eeg_fused_cols: List[str] = []
    for c in df.columns:
        if _is_baseline_block(c):
            continue
        if RE_FMRI_FUSED.match(c):
            fmri_fused_cols.append(c)
        if RE_EEG_FUSED.match(c):
            eeg_fused_cols.append(c)

    fmri_fused_cols = sorted(set(fmri_fused_cols))
    eeg_fused_cols = sorted(set(eeg_fused_cols))

    fmri_only_cols: List[str] = sorted({c for c in df.columns if RE_FMRI_ONLY.match(c)})
    eeg_only_cols: List[str] = sorted({c for c in df.columns if RE_EEG_ONLY.match(c)})

    if args.topk_networks > 0:
        fmri_fused_cols = _keep_topk_networks(fmri_fused_cols, args.topk_networks)
        eeg_fused_cols = _keep_topk_networks(eeg_fused_cols, args.topk_networks)
        fmri_only_cols = _keep_topk_networks(fmri_only_cols, args.topk_networks)
        eeg_only_cols = _keep_topk_networks(eeg_only_cols, args.topk_networks)

    has_concat = len(fmri_only_cols) > 0 and len(eeg_only_cols) > 0

    print(f"[INFO] topk_networks={args.topk_networks}")
    print(f"       fused:   fMRI={len(fmri_fused_cols)} EEG={len(eeg_fused_cols)}")
    print(f"       concat:  FMRIonly={len(fmri_only_cols)} EEGonly={len(eeg_only_cols)}")

    if not fmri_fused_cols or not eeg_fused_cols:
        raise RuntimeError("Missing fused feature blocks")

    if not has_concat:
        print("[WARN] No FMRIonly__/EEGonly__ baseline blocks found; skipping Mconcat")

    rows = []
    for ycol in outcomes:
        if ycol not in df.columns:
            print(f"[WARN] missing outcome {ycol}; skip")
            continue

        cols_needed = [ycol] + covars + fmri_fused_cols + eeg_fused_cols
        if has_concat:
            cols_needed += fmri_only_cols + eeg_only_cols

        d = df[cols_needed].copy()
        d[ycol] = pd.to_numeric(d[ycol], errors="coerce")
        d = d.dropna(subset=[ycol])

        X0 = build_X(d, covars, [])
        Xf = build_X(d, covars, fmri_fused_cols)
        Xe = build_X(d, covars, eeg_fused_cols)
        Xfe = build_X(d, covars, fmri_fused_cols + eeg_fused_cols)
        y = d[ycol].to_numpy(float)

        X_list = [X0, Xf, Xe, Xfe]
        if has_concat:
            Xconcat = build_X(d, covars, fmri_only_cols + eeg_only_cols)
            X_list.append(Xconcat)

        ok = np.isfinite(y)
        for X in X_list:
            ok &= np.all(np.isfinite(X), axis=1)

        y = y[ok]
        X0 = X0[ok]
        Xf = Xf[ok]
        Xe = Xe[ok]
        Xfe = Xfe[ok]
        if has_concat:
            Xconcat = Xconcat[ok]

        n = len(y)
        if n < 60:
            print(f"[WARN] outcome={ycol}: n={n} after complete-case; skip")
            continue

        r0_m, r0_s = cv_r2_nested_ridge(
            X0, y, k=args.kfold, seed=args.seed, alpha_grid=alpha_grid,
            inner_method=args.inner_method, inner_frac=args.inner_frac, inner_k=args.inner_k
        )
        rf_m, rf_s = cv_r2_nested_ridge(
            Xf, y, k=args.kfold, seed=args.seed, alpha_grid=alpha_grid,
            inner_method=args.inner_method, inner_frac=args.inner_frac, inner_k=args.inner_k
        )
        re_m, re_s = cv_r2_nested_ridge(
            Xe, y, k=args.kfold, seed=args.seed, alpha_grid=alpha_grid,
            inner_method=args.inner_method, inner_frac=args.inner_frac, inner_k=args.inner_k
        )
        rfe_m, rfe_s = cv_r2_nested_ridge(
            Xfe, y, k=args.kfold, seed=args.seed, alpha_grid=alpha_grid,
            inner_method=args.inner_method, inner_frac=args.inner_frac, inner_k=args.inner_k
        )

        if has_concat:
            rcat_m, rcat_s = cv_r2_nested_ridge(
                Xconcat, y, k=args.kfold, seed=args.seed, alpha_grid=alpha_grid,
                inner_method=args.inner_method, inner_frac=args.inner_frac, inner_k=args.inner_k
            )
            delta_fc = rfe_m - rcat_m
            n_concat_feats = len(fmri_only_cols) + len(eeg_only_cols)
        else:
            rcat_m, rcat_s = np.nan, np.nan
            delta_fc = np.nan
            n_concat_feats = 0

        rows.append({
            "outcome": ycol,
            "n": n,
            "covars": ",".join(covars),
            "kfold": args.kfold,
            "inner_method": args.inner_method,
            "topk_networks": args.topk_networks,
            "cvR2_covars_mean": r0_m,
            "cvR2_covars_sd": r0_s,
            "cvR2_fmri_mean": rf_m,
            "cvR2_fmri_sd": rf_s,
            "cvR2_eeg_mean": re_m,
            "cvR2_eeg_sd": re_s,
            "cvR2_fused_mean": rfe_m,
            "cvR2_fused_sd": rfe_s,
            "cvR2_concat_mean": rcat_m,
            "cvR2_concat_sd": rcat_s,
            "delta_fused_minus_concat": delta_fc,
            "delta_EEG_given_fMRI": (rfe_m - rf_m),
            "delta_fMRI_given_EEG": (rfe_m - re_m),
            "delta_over_covars": (rfe_m - r0_m),
            "n_fmri_feats": len(fmri_fused_cols),
            "n_eeg_feats": len(eeg_fused_cols),
            "n_fmri_only_feats": len(fmri_only_cols),
            "n_eeg_only_feats": len(eeg_only_cols),
            "n_concat_feats": n_concat_feats,
        })

    out = pd.DataFrame(rows)
    out_csv = outdir / "synergy_cv_summary.csv"
    out.to_csv(out_csv, index=False)
    print("Wrote:", out_csv)


if __name__ == "__main__":
    main()