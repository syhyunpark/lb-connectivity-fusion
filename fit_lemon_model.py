#!/usr/bin/env python3
"""Fit the primary shared-spatial-factor model to a released LEMON input file.

The dataset `lemon_eo_fitted_representation.npz` is the fitted object used in the manuscript. This script refits the same model from the released analysis-ready connectivity input data .
 
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np

import shared_spatial_model as model
from analysis_utils import (
    fit_training_model,
    infer_subject_ids,
    load_npz,
    order_factors,
    scaled_subset,
    training_scales,
    validate_joint_npz,
)


def band_indices(omega: np.ndarray) -> Dict[str, np.ndarray]:
    omega = np.asarray(omega, float).reshape(-1)
    return {
        'theta': np.where((omega >= 4) & (omega < 8))[0],
        'alpha': np.where((omega >= 8) & (omega < 13))[0],
        'beta': np.where((omega >= 13) & (omega < 30))[0],
        'gamma': np.where(omega >= 30)[0],
    }


def feature_blocks(lam_eeg: np.ndarray, lam_fmri: np.ndarray, omega: np.ndarray):
    bands = band_indices(omega)
    features = {
        'feature_fmri': np.asarray(lam_fmri, float),
        'feature_eeg_total': np.sum(lam_eeg, axis=1),
    }
    for name, idx in bands.items():
        if idx.size == 0:
            raise ValueError(f'No frequency bins found for {name}')
        features[f'feature_{name}'] = np.sum(lam_eeg[:, idx, :], axis=1)
    denom = np.sum(lam_eeg, axis=1)
    features['feature_centroid'] = np.divide(
        np.sum(lam_eeg * omega[None, :, None], axis=1),
        denom,
        out=np.full_like(denom, np.nan, dtype=float),
        where=denom > 0,
    )
    return features


def main() -> None:
    ap = argparse.ArgumentParser(description='Refit the primary LEMON shared spatial representation.')
    ap.add_argument('--data', default='lemon_eo_model_inputs.npz')
    ap.add_argument('--out', default='refit_lemon_eo_R12.npz')
    ap.add_argument('--rank', type=int, default=12)
    ap.add_argument('--n-starts', type=int, default=1)
    ap.add_argument('--max-iter', type=int, default=50)
    ap.add_argument('--tol', type=float, default=1e-7)
    ap.add_argument('--alpha-lambda', type=float, default=1.0)
    ap.add_argument('--alpha0', type=float, default=1e-6)
    ap.add_argument('--step-phi', type=float, default=1e-2)
    ap.add_argument('--n-jobs', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--solver', choices=['bvls', 'lsq'], default='bvls')
    ap.add_argument('--workdir', default='refit_work')
    args = ap.parse_args()

    data = load_npz(args.data)
    info = validate_joint_npz(data)
    all_idx = np.arange(info['n'])
    eeg_scale, fmri_scale = training_scales(
        np.asarray(data['C_eeg'], float), np.asarray(data['C_fmri'], float),
        np.asarray(data['W_eeg'], float), np.asarray(data['W_fmri'], float)
    )
    scaled = scaled_subset(data, all_idx, eeg_scale, fmri_scale)

    best, starts = fit_training_model(
        fitter=model, data=scaled, rank=args.rank, mode='fused', seed=args.seed,
        n_starts=args.n_starts, alpha_lambda=args.alpha_lambda, alpha0=args.alpha0,
        step_phi=args.step_phi, max_iter=args.max_iter, tolerance=args.tol,
        n_jobs=args.n_jobs, solver=args.solver, workdir=args.workdir,
    )
    Phi = np.asarray(best.fit['Phi_hat'], float)
    lam_eeg = np.asarray(best.fit['lambda_hat'], float)
    lam_fmri = np.asarray(best.fit['lambda_fmri_hat'], float)
    ordered = order_factors(Phi, lam_eeg, lam_fmri, lam_eeg, lam_fmri, 'fused')
    Phi = ordered['Phi']
    lam_eeg = ordered['train_eeg']
    lam_fmri = ordered['train_fmri']
    omega = np.asarray(data['omega'], float).reshape(-1)
    features = feature_blocks(lam_eeg, lam_fmri, omega)

    payload = {
        'rank': np.array(args.rank),
        'Phi_hat': Phi.astype(np.float32),
        'lambda_eeg_all': lam_eeg.astype(np.float32),
        'lambda_fmri_all': lam_fmri.astype(np.float32),
        'factor_energy': np.asarray(ordered['energy'], float),
        'subject_ids': infer_subject_ids(data),
        'eeg_scale': np.array(eeg_scale),
        'fmri_scale': np.array(fmri_scale),
        'objective': np.array(best.objective),
        'start_method': np.array(best.start_method),
        'start_seed': np.array(best.start_seed),
        'iterations': np.array(best.iterations),
        'last_relative_change': np.array(best.last_relative_change),
        'converged': np.array(best.converged),
        **{k: np.asarray(v, np.float32) for k, v in features.items()},
    }
    np.savez_compressed(args.out, **payload)
    print(f'Wrote {Path(args.out).resolve()}')
    print(f'Selected objective: {best.objective:.6g}')
    print('All start objectives:', [float(x.objective) for x in starts])


if __name__ == '__main__':
    main()
