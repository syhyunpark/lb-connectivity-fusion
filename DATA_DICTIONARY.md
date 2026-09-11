# Data dictionary

This repository contains derived MPI–LEMON analysis objects used in the manuscript.  

## 1. Model inputs

### `lemon_eo_model_inputs.npz`

Exact eyes-open model input.

- subjects: 189
- EEG frequencies: 20
- retained LB coordinates: 50
- primary EEG support: first 20 LB coordinates

Key arrays:

| Key | Shape | Description |
|---|---:|---|
| `C_eeg` | `189 × 20 × 50 × 50` | Empirical EEG amplitude-envelope covariance matrices in LB coordinates |
| `C_fmri` | `189 × 50 × 50` | Empirical resting-state fMRI covariance matrices in LB coordinates |
| `W_eeg` | `50 × 50` | EEG support / weighting matrix used in fitting |
| `W_fmri` | `50 × 50` | fMRI support / weighting matrix used in fitting |
| `omega` | `20` | EEG frequency coordinates |
| `freq_bands` | `20 × 2` | Frequency-bin boundaries |
| `subject_ids` | `189` | Subject order for `C_eeg` and `C_fmri` |
| `m_fmri`, `w_m` | — | Frequency-weighting metadata used by the fitter |
| `K`, `K_eeg`, `F` | scalar | Number of retained LB coordinates, primary EEG-supported coordinates, and EEG frequencies |

### `lemon_ec_model_inputs.npz`

Eyes-closed model input with the same structure as the eyes-open file.

- subjects: 188
- EEG frequencies: 20
- retained LB coordinates: 50

`C_eeg` has shape `188 × 20 × 50 × 50`; `C_fmri` has shape `188 × 50 × 50`.




The files also retain original scaling metadata such as `scale_mode`, `scale_eeg`, `scale_fmri`, `w_eeg_suggest`, and `w_fmri_suggest`.

## 2. Primary fitted representation

### `lemon_eo_fitted_representation.npz`

This is the primary full-data eyes-open fit used in the manuscript.

| Key | Shape | Description |
|---|---:|---|
| `Phi_hat` | `50 × 12` | Learned shared spatial factors in LB coordinates |
| `lambda_eeg_all` | `189 × 20 × 12` | Frequency-resolved EEG strengths for each subject and factor |
| `lambda_fmri_all` | `189 × 12` | fMRI strengths for each subject and factor |
| `factor_energy` | `12` | Factor-ordering summary used for reporting |
| `subject_ids` | `189` | Subject order |
| `feature_fmri` | `189 × 12` | fMRI factor-strength features |
| `feature_eeg_total` | `189 × 12` | Total EEG strength by factor |
| `feature_theta` | `189 × 12` | Theta-band EEG strength by factor |
| `feature_alpha` | `189 × 12` | Alpha-band EEG strength by factor |
| `feature_beta` | `189 × 12` | Beta-band EEG strength by factor |
| `feature_gamma` | `189 × 12` | Gamma-band EEG strength by factor |
| `feature_centroid` | `189 × 12` | Frequency-weighted spectral centroid by factor |

The file also contains fitting diagnostics such as modality scaling, objective value, initialization, iteration count, and convergence information.

## 3. Readable subject features

### `lemon_eo_subject_features.csv`

One row per eyes-open subject (`n=189`).

Columns contain the primary fitted features for factors 1–12:

- `fmri_r01` … `fmri_r12`
- `eeg_total_r01` … `eeg_total_r12`
- `theta_r01` … `theta_r12`
- `alpha_r01` … `alpha_r12`
- `beta_r01` … `beta_r12`
- `gamma_r01` … `gamma_r12`
- `centroid_r01` … `centroid_r12`

These values are exports of arrays already contained in `lemon_eo_fitted_representation.npz`.

## 4. Subject metadata and quality measures

### `lemon_subject_metadata.csv`

One row per eyes-open analysis subject (`n=189`).

The table includes variables used in the prediction and age-association analyses, including:

- `subject_id`
- `has_eo`, `has_ec`
- `age_mid`
- `male`
- education variables used in sensitivity analyses
- `mean_fd` and `log1p_mean_fd`
- `mean_std_dvars`
- `n_windows` and `log_n_windows`
- `channel_retention`
- `aperiodic_exponent_median`
- `hf_log_relative_p90`
 

## 5. Prediction folds

### `prediction_outer_folds.csv`

Outer-fold assignment used for the cross-validated prediction analysis.

Each eligible subject is assigned to exactly one of the five outer folds. The same fold definitions are used when comparing competing prediction models.

## 6. NPZ inventory

### `npz_data_dictionary.csv`

Lists every key, array shape, and dtype contained in the NPZ files. 
