#!/usr/bin/env python3
"""
run_behavior_associations.py

Run feature-on-exposure regressions with robust SE and FDR correction.
"""

import argparse
import re
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import statsmodels.api as sm


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def parse_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def zscore(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    mu = x.mean()
    sd = x.std(ddof=0)
    if not np.isfinite(sd) or sd <= 0:
        return pd.Series(np.nan, index=x.index)
    return (x - mu) / sd


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return q
    pv = p[ok]
    m = pv.size
    order = np.argsort(pv)
    ranked = pv[order]
    q_ranked = ranked * m / (np.arange(1, m + 1))
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0.0, 1.0)
    q_ok = np.empty_like(pv)
    q_ok[order] = q_ranked
    q[ok] = q_ok
    return q


def feature_set_name(col: str) -> str:
    if col.startswith("FMRIonly__"):
        return "FMRI-only"
    if col.startswith("EEGonly__"):
        return "EEG-only"
    if col.startswith("Fused__"):
        return "Fused"
    return "Fused"


def _strip_fused_prefix(c: str) -> str:
    return c[len("Fused__"):] if c.startswith("Fused__") else c


_RE_FMRI_STRENGTH = re.compile(r"^(Lambda_fmri|fMRI_strength)_r\d+$", re.IGNORECASE)
_RE_EEG_BAND = re.compile(r"^EEG_(theta|alpha|beta|gamma)(_AUC)?_r\d+$", re.IGNORECASE)
_RE_EEG_COM = re.compile(r"^EEG_(muHz|specCOM)_r\d+$", re.IGNORECASE)

_RE_BASE_FMRI = re.compile(r"^FMRIonly__(FMRI_score|fMRI_strength)_r\d+$", re.IGNORECASE)
_RE_BASE_EEG_BAND = re.compile(r"^EEGonly__EEG_(theta|alpha|beta|gamma)(_AUC)?_r\d+$", re.IGNORECASE)
_RE_BASE_EEG_COM = re.compile(r"^EEGonly__EEG_(muHz|specCOM)_r\d+$", re.IGNORECASE)


def select_brain_features(df: pd.DataFrame) -> List[str]:
    out: List[str] = []

    for c in df.columns:
        cl = c.lower()
        if "relerr" in cl or "baseline_relerr" in cl or "rankr_baseline" in cl:
            continue

        if c.startswith("FMRIonly__"):
            if "EEG_" in c:
                continue
            if _RE_BASE_FMRI.match(c):
                out.append(c)
            continue

        if c.startswith("EEGonly__"):
            if "FMRI" in c or "fMRI" in c:
                continue
            if _RE_BASE_EEG_BAND.match(c) or _RE_BASE_EEG_COM.match(c):
                out.append(c)
            continue

        c2 = _strip_fused_prefix(c)
        if _RE_FMRI_STRENGTH.match(c2) or _RE_EEG_BAND.match(c2) or _RE_EEG_COM.match(c2):
            out.append(c)
            continue

    out = sorted(set(out), key=lambda x: (feature_set_name(x), x))
    return out


def _as_numeric_or_categorical(df: pd.DataFrame, col: str) -> pd.DataFrame:
    s = df[col]
    xnum = pd.to_numeric(s, errors="coerce")
    frac_ok = np.isfinite(xnum).mean()

    if frac_ok >= 0.8:
        return pd.DataFrame({col: xnum}, index=df.index)

    s2 = s.astype("string").replace({"nan": pd.NA, "NaN": pd.NA, "None": pd.NA, "": pd.NA})
    d = pd.get_dummies(s2, prefix=col, drop_first=True)
    d = d.apply(pd.to_numeric, errors="coerce")
    return d


def build_design_matrix(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    blocks = []
    for c in cols:
        Xi = _as_numeric_or_categorical(df, c)
        if Xi.shape[1] > 0:
            blocks.append(Xi)
    if not blocks:
        return pd.DataFrame(index=df.index)
    X = pd.concat(blocks, axis=1)
    X = X.apply(pd.to_numeric, errors="coerce")
    return X


def fit_ols_safe(y: pd.Series, X: pd.DataFrame, robust: bool, cov_type: str):
    y_np = y.to_numpy(dtype=float)
    X_np = X.to_numpy(dtype=float)
    if robust:
        try:
            return sm.OLS(y_np, X_np).fit(cov_type=cov_type), True
        except Exception:
            return sm.OLS(y_np, X_np).fit(), False
    return sm.OLS(y_np, X_np).fit(), False


def run_feature_on_exposure(
    df: pd.DataFrame,
    exposure: str,
    feature: str,
    covars: List[str],
    robust: bool,
    cov_type: str,
    z_exposure: bool,
) -> Dict:
    y = zscore(df[feature])

    Xcols = [exposure] + covars
    X = build_design_matrix(df, Xcols)

    if exposure in X.columns:
        exp_col = exposure
    else:
        exp_cols = [c for c in X.columns if c.startswith(exposure + "_")]
        if not exp_cols:
            return {
                "exposure": exposure,
                "feature": feature,
                "feature_set": feature_set_name(feature),
                "n": 0,
                "beta": np.nan,
                "se": np.nan,
                "t": np.nan,
                "p": np.nan,
                "robust": robust,
                "cov_type": cov_type,
                "robust_used": robust,
                "exposure_col": None,
            }
        exp_col = exp_cols[0]

    if z_exposure and exp_col in X.columns:
        X[exp_col] = zscore(X[exp_col])

    X2 = sm.add_constant(X, has_constant="add")
    X2 = X2.apply(pd.to_numeric, errors="coerce")

    use = y.notna() & X2.notna().all(axis=1)
    n = int(use.sum())
    if n < (X2.shape[1] + 5):
        return {
            "exposure": exposure,
            "feature": feature,
            "feature_set": feature_set_name(feature),
            "n": n,
            "beta": np.nan,
            "se": np.nan,
            "t": np.nan,
            "p": np.nan,
            "robust": robust,
            "cov_type": cov_type,
            "robust_used": robust,
            "exposure_col": exp_col,
        }

    y2 = y.loc[use].astype(float)
    X2u = X2.loc[use]

    model, robust_used = fit_ols_safe(y2, X2u, robust=robust, cov_type=cov_type)

    cols = list(X2u.columns)
    j = cols.index(exp_col)

    return {
        "exposure": exposure,
        "feature": feature,
        "feature_set": feature_set_name(feature),
        "n": n,
        "beta": float(model.params[j]),
        "se": float(model.bse[j]),
        "t": float(model.tvalues[j]),
        "p": float(model.pvalues[j]),
        "robust": robust,
        "cov_type": cov_type,
        "robust_used": bool(robust_used),
        "exposure_col": exp_col,
    }


def main():
    ap = argparse.ArgumentParser("Run univariate associations with robust SE and FDR")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--outcomes", required=True, help="Comma-separated exposures")
    ap.add_argument("--covars", required=True, help="Comma-separated covariates")
    ap.add_argument("--alpha", type=float, default=0.05)

    ap.add_argument("--direction", choices=["feature_on_exposure"], default="feature_on_exposure")
    ap.add_argument("--standardize_exposure", action="store_true")

    ap.add_argument("--no_robust", action="store_true")
    ap.add_argument("--robust_cov", default="HC1", choices=["HC0", "HC1", "HC2", "HC3"])
    ap.add_argument("--feature_regex", default="", help="Optional regex to restrict features")

    args = ap.parse_args()

    outdir = Path(args.outdir).expanduser().resolve()
    ensure_dir(outdir)

    df = pd.read_csv(args.csv)
    exposures = parse_list(args.outcomes)
    covars = parse_list(args.covars)

    robust = not args.no_robust
    cov_type = args.robust_cov

    missing = [c for c in exposures + covars if c not in df.columns]
    if missing:
        raise ValueError("Missing columns in CSV")

    features = select_brain_features(df)
    if args.feature_regex:
        pat = re.compile(args.feature_regex)
        features = [f for f in features if pat.search(f)]

    if not features:
        raise RuntimeError("No brain features selected")

    counts = pd.Series([feature_set_name(f) for f in features]).value_counts()
    print("[INFO] Selected features by set:\n", counts.to_string())

    rows = []
    for exposure in exposures:
        for feat in features:
            rows.append(
                run_feature_on_exposure(
                    df=df,
                    exposure=exposure,
                    feature=feat,
                    covars=covars,
                    robust=robust,
                    cov_type=cov_type,
                    z_exposure=args.standardize_exposure,
                )
            )

    res = pd.DataFrame(rows)

    res["q_exposure"] = np.nan
    for exposure in exposures:
        idx = res["exposure"] == exposure
        res.loc[idx, "q_exposure"] = bh_fdr(res.loc[idx, "p"].values)

    res["q_global"] = bh_fdr(res["p"].values)

    res["sig_exposure_fdr"] = res["q_exposure"] <= args.alpha
    res["sig_global_fdr"] = res["q_global"] <= args.alpha

    res_path = outdir / "assoc_feature_on_exposure_results.csv"
    res.to_csv(res_path, index=False)

    summ = (
        res.groupby(["exposure", "feature_set"])
        .agg(
            n_tests=("p", "size"),
            n_sig_exposure_fdr=("sig_exposure_fdr", "sum"),
            n_sig_global_fdr=("sig_global_fdr", "sum"),
            n_used=("n", "median"),
            n_robust_fallback=("robust_used", lambda x: int((~x).sum())),
        )
        .reset_index()
    )
    summ_path = outdir / "assoc_feature_on_exposure_summary_counts.csv"
    summ.to_csv(summ_path, index=False)

    top = (
        res.sort_values(["q_exposure", "p"])
        .groupby("exposure")
        .head(25)
    )
    top_path = outdir / "assoc_feature_on_exposure_top25_per_exposure.csv"
    top.to_csv(top_path, index=False)

    print("Wrote:", res_path)
    print("Wrote:", summ_path)
    print("Wrote:", top_path)


if __name__ == "__main__":
    main()