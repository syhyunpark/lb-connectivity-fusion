#!/usr/bin/env python3
"""Validate the public data release and perform lightweight consistency checks."""
from pathlib import Path
import argparse, hashlib
import numpy as np
import pandas as pd


def h(path):
    x=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): x.update(b)
    return x.hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--directory', default='.')
    a=ap.parse_args(); d=Path(a.directory)
    required=['lemon_eo_model_inputs.npz','lemon_ec_model_inputs.npz',
              'lemon_eo_fitted_representation.npz','lemon_eo_subject_features.csv',
              'lemon_subject_metadata.csv','prediction_outer_folds.csv']
    for name in required:
        p=d/name
        if not p.exists(): raise FileNotFoundError(p)
        print(f'{name:42s} {p.stat().st_size/1024**2:7.3f} MiB  {h(p)}')
    with np.load(d/'lemon_eo_model_inputs.npz',allow_pickle=True) as eo, \
         np.load(d/'lemon_ec_model_inputs.npz',allow_pickle=True) as ec, \
         np.load(d/'lemon_eo_fitted_representation.npz',allow_pickle=True) as fit:
        assert eo['C_eeg'].shape[0]==189 and ec['C_eeg'].shape[0]==188
        assert fit['Phi_hat'].shape==(50,12)
        assert fit['lambda_eeg_all'].shape[:2]==(189,20)
        assert fit['lambda_fmri_all'].shape==(189,12)
        assert list(eo['subject_ids'].astype(str))==list(fit['subject_ids'].astype(str))
    f=pd.read_csv(d/'lemon_eo_subject_features.csv')
    m=pd.read_csv(d/'lemon_subject_metadata.csv')
    print(f'EO feature rows: {len(f)}; metadata rows: {len(m)}')
    print('Release validation passed.')

if __name__=='__main__': main()
