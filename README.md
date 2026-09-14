# Bandwidth-aware EEG–fMRI fusion in cortical eigenmode space

This repository accompanies the manuscript:

**Bandwidth-aware fusion of resting-state EEG and fMRI functional connectivity in cortical eigenmode space**

It contains the derived input datasets used for the empirical analysis, the primary fitted representation, and the code for the main prediction, simulation, robustness, and age-association analyses.

## Data

The original MRI and EEG recordings come from the publicly available MPI Leipzig Mind–Brain–Body (MPI–LEMON) dataset. This repository contains derived analysis objects.

Main data files:

- `lemon_eo_model_inputs.npz` — exact eyes-open model input (`n=189`)
- `lemon_ec_model_inputs.npz` — exact eyes-closed model input (`n=188`)
- `lemon_eo_fitted_representation.npz` — primary full-data eyes-open fit (`R=12`)
- `lemon_eo_subject_features.csv` — subject-level features from the primary fit
- `lemon_subject_metadata.csv` — age, sex, education, and data-quality variables used in the analyses
- `prediction_outer_folds.csv` — outer-fold assignments used for age prediction

See `DATA_DICTIONARY.md` for details.

## Model 

EEG and fMRI are first represented in the same template-based cortical Laplace–Beltrami (LB) coordinates, and treated as distinct modality-specific measurements. 

The model then learns common spatial factors while estimating separate subject-level EEG and fMRI strength parameters. EEG strengths are additionally resolved over frequency.
 

## Quick check

Install the Python dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Then verify the available data:

```bash
python3 validate_release.py
```
 

## Quick example

A small model-concordant synthetic example illustrates the core estimator
without requiring the MPI–LEMON data. EEG and fMRI connectivity matrices are
generated with a common low-dimensional spatial basis but separate
modality-specific subject strengths; EEG strengths additionally vary over
frequency and are observed only over lower-order LB modes.
 

### 1. Generate toy data

```bash
python3 simulate_toy_fusion_data.py \
  --out toy_fusion_data.npz \
  --n 80 \
  --K 50 \
  --F 20 \
  --R_true 5 \
  --snr 4 \
  --k_eeg_max 20 \
  --seed 0
```

### 2. Fit the shared-spatial model

```bash
python3 shared_spatial_model.py \
  --data toy_fusion_data.npz \
  --out toy_fit.npz \
  --R 5 \
  --fmri_mode separate \
  --alpha_lambda 0 \
  --max_iter 30 \
  --tol 1e-6 \
  --n_jobs 4 \
  --lam_solver bvls \
  --init_method fmri_eig
```

### 3. Evaluate recovery

```bash
python3 evaluate_toy_recovery.py \
  --data toy_fusion_data.npz \
  --fit toy_fit.npz \
  --outdir toy_recovery
```

The evaluation reports recovery of the generating spatial subspace,
modality-specific subject strengths, and  EEG/fMRI connectivity
matrices.


### Modality-specific signal-generation simulation

Run the simulation grid used to test recovery under distinct EEG and fMRI observation procedures:

```bash
python3 observation_model_grid.py \
  --manifest observation_scenarios.csv \
  --outdir observation_model_runs \
  --seeds 1001-1030 \
  --n_starts 3 \
  --resume
```

Summarize the results:

```bash
python3 observation_model_summary.py \
  --summary_csv observation_model_runs/observation_model_summary.csv \
  --manifest observation_scenarios.csv \
  --outdir observation_model_summary
```

### Adaptive rank-screening simulation

```bash
python3 rank_screening_grid.py \
  --outdir rank_screening_runs \
  --trueR_list 5,10 \
  --n_list 100,200 \
  --noise_models gaussian,student_t,freq_hetero \
  --seeds 1001-1020 \
  --resume
```

Summarize the results:

```bash
python3 rank_screening_summary.py \
  --eval_dir rank_screening_runs/eval \
  --outdir rank_screening_summary
```



## MPI–LEMON analysis: Refit the primary representation

```bash
python3 fit_lemon_model.py
```

This refits the `R=12` eyes-open model from the dataset `lemon_eo_model_inputs.npz`.

The exact fitted representation used in the manuscript is provided as `lemon_eo_fitted_representation.npz`. 


## Main analyses

### Cross-validated age prediction

Eyes-open analysis:

```bash
python3 prediction_analysis.py --condition EO
```

Eyes-closed sensitivity analysis:

```bash
python3 prediction_analysis.py --condition EC
```

Summarize prediction performance and paired bootstrap comparisons:

```bash
python3 prediction_summary.py \
  --prediction_csvs prediction_results/EO/R12/all_predictions.csv \
  --outdir prediction_summary \
  --n_boot 10000
```

PCA-regularized CCA benchmark:

```bash
python3 cca_prediction_benchmark.py
```



### Real-data representation robustness analysis

Create the K=50 EEG-support and band-specific sensitivity inputs:

```bash
python3 representation_robustness.py make-inputs
```

Compare an alternative fitted representation with the primary fit. 

For example, refit the model using a hard EEG spatial support through the first 15 LB modes rather than the primary first-20 support:
```bash
python3 fit_lemon_model.py \
  --data robustness_inputs/eeg_hard15.npz \
  --out robustness_hard15_fit.npz \
  --rank 12 \
  --n-starts 1 \
  --max-iter 50 \
  --tol 1e-7 \
  --workdir robustness_hard15_work
```

Compare the resulting representation with the primary fit:

```bash
python3 representation_robustness.py compare \
  --alternative robustness_hard15_fit.npz \
  --output representation_comparison_hard15.csv
``` 


### Age-association robustness analysis

```bash
python3 age_association_analysis.py
python3 gamma_centroid_sensitivity.py
```



## File organization
 
`shared_spatial_model.py` contains the core estimator. `analysis_utils.py` contains shared utilities used by the fold-wise prediction code.

 

## Citation

Please cite the manuscript and the original MPI–LEMON dataset when using these derived data or analysis code.
