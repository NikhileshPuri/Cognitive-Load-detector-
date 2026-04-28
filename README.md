# EEG Cognitive Load Decoding – Riemannian & Tangent Space Pipeline

> **Complete pipeline for offline and real‑time decoding of cognitive load (low/medium/high) from EEG**  
> Uses covariance matrices, Riemannian geometry (Tangent Space), and standard ML (LR/RF/XGBoost).  
> Achieves **>96% test accuracy** on 12 tasks, 29 subjects.

---

## Table of Contents
1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [Requirements & Installation](#requirements--installation)
4. [Complete Pipeline Step‑by‑Step](#complete-pipeline-step-by-step)
   - [1. Preprocess Raw EEG](#1-preprocess-raw-eeg)
   - [2. Find Common Channels Across Subjects](#2-find-common-channels-across-subjects)
   - [3. Extract Features per Subject](#3-extract-features-per-subject)
   - [4. Aggregate All Subjects](#4-aggregate-all-subjects)
   - [5. Train/Test Split](#5-traintest-split)
   - [6. Train Tangent Space Classifier (LR/RF)](#6-train-tangent-space-classifier-lrrf)
   - [7. (Optional) XGBoost + SHAP](#7-optional-xgboost--shap)
   - [8. Predict Cognitive Load for a New Subject (Offline)](#8-predict-cognitive-load-for-a-new-subject-offline)
   - [9. Map Task Predictions to Load & Smoothing](#9-map-task-predictions-to-load--smoothing)
   - [10. Real‑time Playback Visualization](#10-real-time-playback-visualization)
5. [Configuration & Customisation](#configuration--customisation)
6. [Example Results](#example-results)
7. [Troubleshooting](#troubleshooting)
8. [Citation](#citation)
9. [License](#license)

---

## Overview

This repository provides a complete, modular pipeline for **decoding cognitive load** from EEG signals using **Riemannian geometry**. Unlike traditional feature‑based approaches, we operate directly on **covariance matrices** of EEG epochs, project them into a **Tangent Space** (Euclidean approximation of the Riemannian manifold), and then use standard classifiers.

**Key features:**
- Multi‑task dataset support (N‑back, MATB, Flanker, PVT, Resting State)
- Sliding‑window epoching
- Per‑frequency‑band covariance extraction (theta, alpha, beta)
- Tangent space embedding **per band** (preserves Riemannian structure)
- Regularisation to ensure positive‑definite covariance matrices
- Classifiers: Logistic Regression, Random Forest, XGBoost
- **SHAP explainability** for XGBoost
- Temporal smoothing and real‑time playback visualisation
- Within‑subject CV and leave‑one‑subject‑out (LOSO)
- Offline prediction for new subjects

All scripts are ready‑to‑use with command‑line arguments. The pipeline was validated on 29 subjects (122,906 epochs) and achieved **96.36% test accuracy** on 12‑class task classification and smooth 3‑level load decoding.

---

## Repository Structure

| File | Description |
|------|-------------|
| `preproc_subject_argument.py` | Raw EEG (`.set`) → preprocessed `.npz` (epochs, metadata) |
| `extract_common_channels_again.py` | Computes intersection of channel names across all subjects |
| `extract_features_subject.py` | Per‑subject: bandpass filtering, covariance matrices, bandpower features |
| `aggregate_features.py` | Concatenates all subject CSVs → one clean dataset |
| `rebuild_all_covs.py` | Stacks all subjects’ covariance matrices into one `.npy` file |
| `train_test_split.py` | Stratified train/test/val split preserving class balance |
| `tangent_train_eval.py` | Fits Tangent Space per band, trains LR/RF classifier, evaluates |
| `tangent_xgboost_shap.py` | Same but with XGBoost and SHAP importance |
| `tangent_within_subject_fixed.py` | Example of within‑subject cross‑validation |
| `predict_subject_load_offline_v2.py` | Loads a saved pipeline and predicts load for a new subject |
| `map_and_smooth_preds.py` | Maps task probabilities → continuous load score + moving average |
| `realtime_playback_visualize.py` | Animated playback of smoothed load score |
| `eegnet_train_explain.py` | Deep learning baseline (EEGNet + Integrated Gradients) |
| `check_cov_shapes.py` | Utility to inspect covariance file dimensions |
| `commands.txt` / `wrapper.txt` | Example execution logs and PowerShell loops |
| `environment.yml` | Conda environment specification |
| `LICENSE` | MIT License |
| `README.md` | This document |

---

## Requirements & Installation

**Create and activate a conda environment:**

```bash
conda create -n cogbci python=3.10
conda activate cogbci
Install dependencies:

bash
pip install mne numpy pandas scipy scikit-learn pyriemann xgboost shap matplotlib joblib torch captum
Note: pyriemann is required for Tangent Space. captum is only needed for eegnet_train_explain.py.

Verify installation:

python
import pyriemann
print(pyriemann.__version__)
Complete Pipeline Step‑by‑Step
All paths in the commands below use D:/Data_cog as an example. Change them to match your own directory structure.

1. Preprocess Raw EEG
Convert raw EEGLAB (.set/.fdt) files into a structured .npz archive containing:

epochs: sliding windows (n_epochs × n_channels × n_times)

ch_names, sfreq, subject, session, task

Filtering: Notch 50 Hz, bandpass 1–40 Hz, resample to 128 Hz, ICA (EOG correction).
Epoching: 2‑s windows for N‑back/rest, 5‑s windows for MATB (both step size = window length, non‑overlapping).

bash
# All subjects
python preproc_subject_argument.py --root "D:/Data_cog" --out "D:/Data_cog/data_preproc"

# Single subject (for testing)
python preproc_subject_argument.py --root "D:/Data_cog" --out "D:/Data_cog/data_preproc" --subject sub-28
Output: D:/Data_cog/data_preproc/sub-28_ses-*_taskname.npz (one per original .set file).

2. Find Common Channels Across Subjects
Because different subjects may have slightly different channel sets, we identify the intersection of all channel names present in the preprocessed files. This ensures covariance matrices have identical dimensions across subjects.

bash
python extract_common_channels_again.py
Output: D:/Data_cog/features/common_channels.txt – one channel name per line.

3. Extract Features per Subject
For each preprocessed .npz file, this script:

Loads the common channels only (skipping files that miss any)

Applies bandpass filters for theta (4‑7 Hz), alpha (8‑12 Hz), beta (13‑30 Hz)

Computes per‑epoch covariance matrix for each band (using np.cov)

Also extracts bandpower features (log mean & variance) and spectral entropy

bash
python extract_features_subject.py --subject sub-28 --preproc_dir "D:/Data_cog/data_preproc" --out_dir "D:/Data_cog/features" --timeout 600
Repeat for all subjects. A PowerShell loop is provided in wrapper.txt:

powershell
$subs = @("sub-01","sub-02",...,"sub-29")
foreach ($s in $subs) {
  python extract_features_subject.py --subject $s --preproc_dir "D:/Data_cog/data_preproc" --out_dir "D:/Data_cog/features"
}
Output per subject:

sub-XX_features.csv – one row per epoch with task label, bandpower features, etc.

covs_sub-XX.npy – numpy array of shape (n_epochs, 3, n_channels, n_channels) – covariance for each band.

4. Aggregate All Subjects
Combine all subject‑level CSV files into a single dataset (all_features_clean.csv) and all covariance matrices into one large array (all_covs.npy). The aggregation also normalises numeric features (StandardScaler) and encodes task labels (task_enc).

bash
python aggregate_features.py --features_dir "D:/Data_cog/features" --out_raw "D:/Data_cog/features/all_features_raw.csv" --out_clean "D:/Data_cog/features/all_features_clean.csv"

python rebuild_all_covs.py --features_dir "D:/Data_cog/features" --all_csv "D:/Data_cog/features/all_features_clean.csv" --out "D:/Data_cog/features/all_covs.npy"
Output:

all_features_clean.csv – 122,906 rows (for 29 subjects)

all_covs.npy – shape (122906, 3, 58, 58)

cov_index.npy – index mapping for later use.

5. Train/Test Split
Split the aggregated dataset into training (70% if using val), test (20%), and validation (10%) stratified by task label to preserve class distribution.

bash
python train_test_split.py --data "D:/Data_cog/features/all_features_clean.csv" --out_dir "D:/Data_cog/features/splits" --test_size 0.2 --val_size 0.1 --random_state 42
Output: train.csv, test.csv, val.csv inside splits/.

6. Train Tangent Space Classifier (LR/RF)
This is the core training script:

Loads train/test/val indices and the memory‑mapped all_covs.npy

For each frequency band (0,1,2), extracts training covariance matrices, applies regularisation (reg * I), fits a TangentSpace object, then transforms train, test (and val) sets.

Concatenates features from all bands (dimensions sum: 58×58→1711 per band → 5133 total features)

Optionally applies a StandardScaler

Trains a classifier (LogisticRegression or RandomForest)

Evaluates on test set and optionally on validation set

Saves complete pipeline (list of TangentSpace objects, scaler, classifier, metadata) as a .joblib file.

Example (Random Forest):

bash
python tangent_train_eval.py \
  --all_csv "D:/Data_cog/features/all_features_clean.csv" \
  --all_covs "D:/Data_cog/features/all_covs.npy" \
  --train_csv "D:/Data_cog/features/splits/train.csv" \
  --test_csv "D:/Data_cog/features/splits/test.csv" \
  --val_csv "D:/Data_cog/features/splits/val.csv" \
  --out "D:/Data_cog/models/tangent_agg.joblib" \
  --reg 0.01 --scale --classifier rf --verbose 1
Output: Saved model pipeline. Console prints accuracy, F1, confusion matrix, and classification report.

Example with Logistic Regression: --classifier lr (default).

7. (Optional) XGBoost + SHAP
Train XGBoost on the same tangent features and compute SHAP values for interpretability. The script automatically handles validation for early stopping.

bash
python tangent_xgboost_shap.py \
  --all_csv "D:/Data_cog/features/all_features_clean.csv" \
  --all_covs "D:/Data_cog/features/all_covs.npy" \
  --train_csv "D:/Data_cog/features/splits/train.csv" \
  --test_csv "D:/Data_cog/features/splits/test.csv" \
  --val_csv "D:/Data_cog/features/splits/val.csv" \
  --out_model "D:/Data_cog/models/tangent_xgb.joblib" \
  --reg 0.01 --nrounds 200 --early_stopping 20 --compute_shap --shap_sample 500
Output:

Saved pipeline (tangent_xgb.joblib)

*_shap_values.npy – raw SHAP values

*_shap_feature_importance.csv – mean absolute SHAP per feature

8. Predict Cognitive Load for a New Subject (Offline)
Given a subject’s covariance file (.npy) and feature CSV (generated by extract_features_subject.py), load a pre‑trained pipeline, transform the covariances using the saved ts_list, and output per‑epoch predictions.

bash
python predict_subject_load_offline_v2.py \
  --pipeline "D:/Data_cog/models/tangent_agg.joblib" \
  --covs "D:/Data_cog/features/covs_sub-02.npy" \
  --features_csv "D:/Data_cog/features/sub-02_features.csv" \
  --out_csv "D:/Data_cog/visualizations/sub-02_preds.csv" \
  --reg 1e-2
The output CSV contains:

pred_label_int and pred_task (most likely task)

p_low, p_med, p_high – aggregated probabilities over tasks that map to each load level

score – continuous load score: 0.0*low + 0.5*med + 1.0*high

Mapping from task to load level is defined in DEFAULT_LOAD_MAP inside the script. You can override it via --load_map_csv.

9. Map Task Predictions to Load & Smoothing
The raw predictions from the previous step contain probabilities over 12 tasks. map_and_smooth_preds.py:

Computes a continuous load_score using the default mapping (Low=0.0, Medium=0.5, High=1.0)

Derives a discrete load_label (Low/Medium/High) based on thresholds (0.33, 0.66)

Applies a centred moving average to the load score (window size --smooth)

Produces smoothed labels and scores.

bash
python map_and_smooth_preds.py --in "D:/Data_cog/visualizations/sub-02_preds.csv" --out_dir "D:/Data_cog/visualizations" --smooth 5
Output:

*_mapped.csv – unsmoothed load scores

*_mapped_sm5.csv – smoothed load scores and labels

10. Real‑time Playback Visualization
Animate the smoothed load score over epochs, simulating a real‑time cognitive load monitor. Each epoch appears at a playback rate of --rate seconds per epoch.

bash
python realtime_playback_visualize.py --preds "D:/Data_cog/visualizations/sub-02_preds_v2_mapped_sm5.csv" --rate 1.0
Features:

Background colour bands: green (Low), yellow (Medium), red (High)

Line plot of load score history

Scatter point at current epoch (colour = load label)

Text display of exact score and label

Press Ctrl+C to exit.

Configuration & Customisation
Changing Frequency Bands
Edit the bands dictionary in extract_features_subject.py (lines ~12–15) and also update map_and_smooth_preds.py if needed. The covariance array shape will change accordingly; the rest of the pipeline (Tangents etc.) automatically adapts.

Load Mapping
Two scripts contain load‑to‑task mappings:

predict_subject_load_offline_v2.py: DEFAULT_LOAD_MAP (keys: task names, values: 'low'/'medium'/'high')

map_and_smooth_preds.py: DEFAULT_MAP (keys: task names, values: 'Low'/'Medium'/'High')

To customise, either edit these dictionaries directly or provide a CSV file with columns task,load using --load_map_csv in the prediction script.

Regularisation (--reg)
Covariance matrices are regularised by adding reg * trace(C)/n_ch * I to each matrix. A value of 0.01 works well for the 58‑channel data. If you encounter “matrix not positive definite” errors, increase the value (e.g., 0.1). The tangent_within_subject_fixed.py script shows an adaptive method.

Multi‑class vs. Binary Load
The pipeline predicts 12 tasks first and then aggregates probabilities into 3 load levels. If you prefer to directly predict load (3 classes), you would need to change the label column in the training CSV and retrain.

Results
Task classification (12 tasks) – Tangent Space + Random Forest

Test accuracy: 96.36%

Macro F1: 96.72%

The XGBoost classifier trained on tangent space features (5133 dimensions from theta, alpha, beta bands) achieved a test accuracy of 97.23% and macro F1 score of 97.18% on 12 task classes (N=24,582 test epochs). This slightly outperforms the Random Forest baseline (96.36% accuracy). The confusion matrix shows excellent discrimination, with minor confusions primarily between zeroBACK/twoBACK and PVT/zeroBACK. Training employed a fallback xgb.train routine due to scikit‑learn wrapper version incompatibility, running 200 boosting rounds with early stopping (best iteration at round 199). The model is saved as tangent_xgb.joblib and ready for inference on new subjects.


Confusion matrix shows very few misclassifications (e.g., some confusion between twoBACK and zeroBACK, but generally excellent separation).

Load decoding (3 levels) – via aggregation + smoothing

Smoothed load scores track expected transitions between low/medium/high load blocks.

Visualised in realtime_playback_visualize.py.

Troubleshooting
Problem	Likely cause	Solution
No module named 'pyriemann'	Package not installed	pip install pyriemann
Covariance matrix not positive definite	Insufficient regularisation	Increase --reg (e.g., 0.1 or 0.5)
Missing common channels	Some .npz files have different channel sets	Run extract_common_channels_again.py and check that all preproc files are valid
Out of memory when loading all_covs.npy	Array is large (122k × 3 × 58 × 58 ≈ 12 GB)	The training scripts already use mmap_mode='r'. Do not load the full array into RAM directly.
UID mismatch between CSV and covs	The uid column format changed	Ensure extract_features_subject.py creates consistent uid (subject_task_epoch). Re‑run aggregation if needed.
XGBoost early stopping error	XGBClassifier expects eval_set and early_stopping_rounds; older versions may lack	Use tangent_xgboost_shap.py which has a fallback to xgb.train
ICA fails for some file	Not enough channels or convergence issues	The script catches exceptions and continues without ICA for that file
Citation
If you use this pipeline in academic work, please cite:

text
[Nikhilesh Puri], "EEG Cognitive Load Decoding using Riemannian Tangent Space", GitHub repository, 2026.
