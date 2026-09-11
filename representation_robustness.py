#!/usr/bin/env python3
"""Compact real-data representation-robustness utilities.

This  focuses on operations: 

1. construct nearby EEG-support and band-specific K=50 sensitivity inputs;
2. compare an alternative fitted representation with the released primary fit.

(The manuscript's K=75 and template MNE source-transfer analyses require
additional intermediate K=200/MNE objects.)
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import scipy.linalg as la
from scipy.optimize import linear_sum_assignment


def load_npz(path: str | Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def hard_mask(K: int, cutoff: int, include_diag: bool) -> np.ndarray:
    W = np.zeros((K, K), float)
    W[:cutoff, :cutoff] = 1.0
    if not include_diag:
        np.fill_diagonal(W, 0.0)
    return W


def taper_mask(K: int, full_through: int, zero_at: int, include_diag: bool) -> np.ndarray:
    rho = np.zeros(K, float)
    rho[:full_through] = 1.0
    for index in range(full_through, min(zero_at, K)):
        k = index + 1
        rho[index] = 0.5 * (1 + np.cos(np.pi * (k - full_through) / (zero_at - full_through)))
    W = np.sqrt(np.outer(rho, rho))
    if not include_diag:
        np.fill_diagonal(W, 0.0)
    return W


def include_diag(data: Dict[str, np.ndarray]) -> bool:
    if 'include_diag' in data:
        return bool(int(np.asarray(data['include_diag']).reshape(-1)[0]))
    return bool(int(np.asarray(data.get('include_diag_eeg', 1)).reshape(-1)[0]))


def save_variant(data: Dict[str, np.ndarray], out: Path, W_eeg: np.ndarray, C_eeg=None, omega=None, label=''):
    Ce = np.asarray(data['C_eeg'] if C_eeg is None else C_eeg, np.float32)
    om = np.asarray(data['omega'] if omega is None else omega, np.float32).reshape(-1)
    if Ce.shape[1] == np.asarray(data['C_eeg']).shape[1]:
        m_fmri = np.asarray(data['m_fmri'], np.int32)
        w_m = np.asarray(data['w_m'], np.float32)
    else:
        # A band-specific input has one EEG frequency summary. The fMRI strength
        # remains a separate modality-specific parameter, so use one unit weight.
        m_fmri = np.array([0], np.int32)
        w_m = np.array([1.0], np.float32)
    keep = {
        'C_eeg': Ce,
        'C_fmri': np.asarray(data['C_fmri'], np.float32),
        'W_eeg': np.asarray(W_eeg, np.float32),
        'W_fmri': np.asarray(data['W_fmri'], np.float32),
        'omega': om,
        'm_fmri': m_fmri,
        'w_m': w_m,
        'include_diag': np.array(int(include_diag(data)), np.int32),
        'sigma_eeg': np.array(1.0), 'sigma_fmri': np.array(1.0),
        'subject_ids': np.asarray(data['subject_ids']),
        'setting': np.array(label),
    }
    np.savez_compressed(out, **keep)


def make_sensitivity_inputs(input_path: str, outdir: str) -> None:
    data = load_npz(input_path)
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    K = int(np.asarray(data['C_fmri']).shape[-1])
    diag = include_diag(data)
    save_variant(data, out/'eeg_hard10.npz', hard_mask(K, 10, diag), label='hard10')
    save_variant(data, out/'eeg_hard15.npz', hard_mask(K, 15, diag), label='hard15')
    save_variant(data, out/'eeg_taper10_20.npz', taper_mask(K, 10, 20, diag), label='taper10_20')

    omega = np.asarray(data['omega'], float).reshape(-1)
    bands = {
        'theta': np.where((omega >= 4) & (omega < 8))[0],
        'alpha': np.where((omega >= 8) & (omega < 13))[0],
        'beta': np.where((omega >= 13) & (omega < 30))[0],
        'gamma': np.where(omega >= 30)[0],
    }
    C = np.asarray(data['C_eeg'], float)
    for name, idx in bands.items():
        if idx.size == 0: continue
        Cb = np.mean(C[:, idx, :, :], axis=1, keepdims=True)
        save_variant(data, out/f'band_{name}.npz', np.asarray(data['W_eeg'], float),
                     C_eeg=Cb, omega=np.array([float(np.mean(omega[idx]))]), label=f'band_{name}')
    print(f'Wrote K=50 sensitivity inputs to {out.resolve()}')


def orth(Phi: np.ndarray) -> np.ndarray:
    Q, _ = la.qr(np.asarray(Phi, float), mode='economic')
    return Q


def get_array(d: Dict[str, np.ndarray], *keys: str):
    for k in keys:
        if k in d: return np.asarray(d[k], float)
    return None


def compare_fits(reference: str, alternative: str, output: str) -> None:
    ref, alt = load_npz(reference), load_npz(alternative)
    A0 = get_array(ref, 'Phi_hat'); B0 = get_array(alt, 'Phi_hat')
    if A0 is None or B0 is None: raise KeyError('Both fits need Phi_hat')
    K = min(A0.shape[0], B0.shape[0])
    A, B = orth(A0[:K]), orth(B0[:K])
    s = np.clip(la.svdvals(A.T @ B), 0, 1)
    # Normalized subspace overlap: 1 means identical lower-dimensional subspace.
    subspace_similarity = float(np.sqrt(np.sum(s*s) / max(1, min(A.shape[1], B.shape[1]))))

    M = A0[:K].T @ B0[:K]
    rows, cols = linear_sum_assignment(-np.abs(M))
    order = np.argsort(rows); rows, cols = rows[order], cols[order]
    map_similarity = np.abs(M[rows, cols])

    result = {
        'reference': str(reference), 'alternative': str(alternative),
        'common_K': K, 'reference_rank': A0.shape[1], 'alternative_rank': B0.shape[1],
        'subspace_similarity': subspace_similarity,
        'mean_principal_angle_deg': float(np.mean(np.degrees(np.arccos(s)))),
        'max_principal_angle_deg': float(np.max(np.degrees(np.arccos(s)))),
        'median_matched_map_similarity': float(np.median(map_similarity)),
        'min_matched_map_similarity': float(np.min(map_similarity)),
    }

    # Feature correlations when both fitted files contain full-subject features.
    for stem in ['feature_fmri', 'feature_eeg_total', 'feature_centroid']:
        X = get_array(ref, stem); Y = get_array(alt, stem)
        if X is None or Y is None or X.shape[0] != Y.shape[0]: continue
        vals = []
        for r, c in zip(rows, cols):
            if r < X.shape[1] and c < Y.shape[1] and np.std(X[:, r]) > 0 and np.std(Y[:, c]) > 0:
                vals.append(abs(float(np.corrcoef(X[:, r], Y[:, c])[0, 1])))
        if vals: result[f'median_{stem}_correlation'] = float(np.median(vals))

    pd.DataFrame([result]).to_csv(output, index=False)
    print(pd.DataFrame([result]).to_string(index=False))
    print(f'Wrote {Path(output).resolve()}')


def main() -> None:
    ap = argparse.ArgumentParser(description='Real-data representation robustness utilities.')
    sub = ap.add_subparsers(dest='command', required=True)
    p = sub.add_parser('make-inputs', help='Create K=50 EEG-support and band-specific sensitivity inputs.')
    p.add_argument('--input', default='lemon_eo_model_inputs.npz')
    p.add_argument('--outdir', default='robustness_inputs')
    p = sub.add_parser('compare', help='Compare an alternative fit with the released primary fit.')
    p.add_argument('--reference', default='lemon_eo_fitted_representation.npz')
    p.add_argument('--alternative', required=True)
    p.add_argument('--output', default='representation_comparison.csv')
    args = ap.parse_args()
    if args.command == 'make-inputs': make_sensitivity_inputs(args.input, args.outdir)
    else: compare_fits(args.reference, args.alternative, args.output)


if __name__ == '__main__': main()
