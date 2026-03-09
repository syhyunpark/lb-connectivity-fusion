# lb-connectivity-fusion
Code for the paper:

**Bandwidth-aware fusion of resting-state EEG–fMRI connectivity in cortical eigenmode space**

This repository contains a minimal set of standalone scripts for:

- fitting the LB-domain EEG–fMRI connectivity fusion model,
- extracting fused subject-level features,
- generating network visualizations and diagnostics,
- and implementing the main simulation and MPI–LEMON analysis workflows.


## Main scripts

- `fit_map_fixedR_cov_fast.py` — fit the fused EEG–fMRI model (fixed rank or ARD)
- `inspect_fit.py` — extract fitted feature summaries and diagnostics
- `export_network_maps_to_surface.py` — export fitted spatial factors to cortical surface maps
- `make_network_montage_nilearn.py` — make surface montages of fitted networks
- `make_elbow_figure.py` — rank-selection elbow plots
- `run_synergy_analysis.py` — nested-CV prediction / multimodal synergy analysis
- `run_behavior_associations.py` — univariate association screening
- `make_age_assoc_outputs.py` — manuscript-ready association figures and tables
- `plot_scale_frequency_signature.py` — scale–frequency summary figure
- `roi_top_labels_schaefer_fsaverage5.py` — Schaefer ROI top-label extraction
- `roi_system_interpretability_from_BI.py` — Schaefer-7 system composition summaries

Additional utility / figure scripts:
- `simulate_toy_fusion_data.py` — generate a small synthetic dataset for a runnable example
- `plot_group_latent_sigma.py` — plot the group-typical latent spatio-spectral field 

## Requirements

- Python ≥ 3.9
- `numpy`, `scipy`, `pandas`, `matplotlib`, `nibabel`, `nilearn`, `joblib`, `pyyaml`

Install:
```bash
pip install -r requirements.txt
```

## Quick runnable example
A small synthetic example illustrating the end-to-end workflow is:

### 1. Generate a LB-domain EEG–fMRI dataset
```bash
python3 simulate_toy_fusion_data.py \
  --out toy_fusion_data.npz \
  --n 200 \
  --K 50 \
  --F 20 \
  --R_true 5 \
  --snr 4 \
  --k_eeg_max 20 \
  --seed 0
```

### 2. Fit the fusion model
```bash
python3 fit_map_fixedR_cov_fast.py \
  --data toy_fusion_data.npz \
  --out toy_fit.npz \
  --R 5 \
  --fmri_mode separate \
  --max_iter 30 \
  --tol 1e-5 \
  --sort_factors median
```

### 3. Inspect the fit
```bash
python3 inspect_fit.py \
  --data toy_fusion_data.npz \
  --fit toy_fit.npz \
  --outdir toy_inspect \
  --prefix test \
  --formats png,pdf \
  --sort_by_energy median
```

This should produce:
-	a fitted model file (toy_fit.npz)
-	feature summaries and diagnostics in toy_inspect/

### 4. Evaluate recovery against the ground truth
```bash
python3 evaluate_toy_recovery.py \
  --data toy_fusion_data.npz \
  --fit toy_fit.npz \
  --outdir toy_recovery \
  --prefix toy
```

This reports:
- subspace recovery of the fitted spatial factors,
- matched factor-wise recovery metrics,
- EEG/fMRI lambda recovery,
- and reconstructed covariance recovery.

The fitted model recovers the true shared factor subspace very accurately (principal angles all below about 2° in our example) and improves reconstruction over simple modality-wise baselines for both EEG and fMRI. 


### Reproducibility note

This repository provides the core model-fitting, feature-extraction, and figure-generation scripts used in the paper, together with a runnable synthetic example that illustrates the full workflow. Reproducing the MPI–LEMON application additionally requires access to the public dataset and user-prepared LB-domain EEG/fMRI connectivity derivatives.

## Paper workflows

### Simulation studies
The simulation results in the paper are based on repeated synthetic-data generation followed by fitting and evaluation across:
-	different true ranks; different EEG spatial cutoffs; and different signal-to-noise ratios.

A typical fixed-rank fit is:
```bash
python fit_map_fixedR_cov_fast.py \
  --data sim_data.npz \
  --out fit_result.npz \
  --R 5 \
  --max_iter 20 \
  --tol 1e-6 \
  --n_jobs 4 \
  --lam_solver bvls
```

A typical ARD-based over-parameterized fit for effective-rank estimation is:
```bash
python fit_map_fixedR_cov_fast.py \
  --data sim_data.npz \
  --out fit_result_ard.npz \
  --use_ard \
  --Rmax 20 \
  --max_iter 20 \
  --tol 1e-6 \
  --n_jobs 4 \
  --lam_solver bvls
```
ARD-based rank screening can be analyzed with:
- analyze_ard_results.py
- make_elbow_figure.py

### MPI–LEMON application

The paper’s real-data application uses publicly available MPI–LEMON resting-state EEG and fMRI data.

Given LB-domain EEG/fMRI connectivity objects, the main fit is:

```bash
python fit_map_fixedR_cov_fast.py \
  --data realdata_fusion_K50_EEGK20_EO.npz \
  --out fit_real_EO_sep_R12_default.npz \
  --R 12 \
  --fmri_mode separate \
  --max_iter 30 \
  --tol 1e-7 \
  --n_jobs 4 \
  --lam_solver bvls \
  --sort_factors median
```

Typical downstream steps include:
- inspect_fit.py
- export_network_maps_to_surface.py
- make_network_montage_nilearn.py
- run_synergy_analysis.py
- run_behavior_associations.py
- make_age_assoc_outputs.py

