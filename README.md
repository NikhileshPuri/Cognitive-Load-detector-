**EEG Cognitive Load Decoding – Riemannian & Tangent Space Pipeline**

This repository provides a complete pipeline for offline and real-time decoding of cognitive load from EEG data (low/medium/high load) using covariance matrices, Riemannian geometry (Tangent Space), and standard machine learning classifiers. It supports:

- Preprocessing of raw EEG (`.set` files) – filte0ing, ICA, sliding-window epoching
- Extraction of per‑band covariance matrices (theta, alpha, beta) and spectral features
- Aggregation across subjects and train/test splits
- Tangent space mapping (per frequency band) with regularization
- Classification using Logistic Regression, Random Forest, or XGBoost
- SHAP feature importance for XGBoost
- Within‑subject cross‑validation and leave‑one‑subject‑out (LOSO) evaluation
- Per‑subject prediction of cognitive load from new EEG data
- Temporal smoothing and real‑time playback visualization

  ## Requirements

Create a conda environment with:

```bash
conda create -n cogbci python=3.10
conda activate cogbci
pip install mne numpy pandas scikit-learn scipy pyriemann xgboost shap matplotlib joblib torch captum

The pipeline has been validated on a multi‑task dataset (N‑back, MATB, Flanker, PVT, Resting State) with 29 subjects and achieves >96% test accuracy.
Pipeline Steps (Detailed)
1. Preprocess Raw EEG
Convert raw .set/.fdt files (EEGLAB format) into .npz archives containing:

epochs: sliding windows (n_epochs × n_channels × n_times)

ch_names, sfreq, subject, session, task

bash
python preproc_subject_argument.py --root "D:/Data_cog" --out "D:/Data_cog/data_preproc"
# Optionally restrict to one subject:
python preproc_subject_argument.py --root "D:/Data_cog" --out "D:/Data_cog/data_preproc" --subject sub-28
2. Find Common Channels Across Subjects
Identify the intersection of EEG channels present in all subjects (required for consistent covariance matrices).

bash
python extract_common_channels_again.py
# Output: D:/Data_cog/features/common_channels.txt
3. Extract Features per Subject
For each preprocessed .npz file, compute:

Per‑epoch covariance matrices (for theta, alpha, beta bands after bandpass filtering)

Bandpower features (log mean & variance per band)

Spectral entropy across the three bands

bash
python extract_features_subject.py --subject sub-28 --preproc_dir "D:/Data_cog/data_preproc" --out_dir "D:/Data_cog/features"
Repeat for all subjects (see wrapper.txt for a PowerShell loop).

4. Aggregate All Subjects
Combine all subject‑level CSV files into one all_features_clean.csv (with label encoding) and all covariance matrices into all_covs.npy.

bash
python aggregate_features.py --features_dir "D:/Data_cog/features" --out_raw "D:/Data_cog/features/all_features_raw.csv" --out_clean "D:/Data_cog/features/all_features_clean.csv"
python rebuild_all_covs.py --features_dir "D:/Data_cog/features" --all_csv "D:/Data_cog/features/all_features_clean.csv" --out "D:/Data_cog/features/all_covs.npy"
5. Train/Test Split
Stratified split preserving class balance (based on task_enc column).

bash
python train_test_split.py --data "D:/Data_cog/features/all_features_clean.csv" --out_dir "D:/Data_cog/features/splits" --test_size 0.2 --val_size 0.1
6. Train Tangent Space Classifier (Logistic Regression / Random Forest)
Fit a TangentSpace per frequency band on the training set, transform train/test/val, then train a classifier.

bash
python tangent_train_eval.py \
  --all_csv "D:/Data_cog/features/all_features_clean.csv" \
  --all_covs "D:/Data_cog/features/all_covs.npy" \
  --train_csv "D:/Data_cog/features/splits/train.csv" \
  --test_csv "D:/Data_cog/features/splits/test.csv" \
  --val_csv "D:/Data_cog/features/splits/val.csv" \
  --out "D:/Data_cog/models/tangent_agg.joblib" \
  --reg 0.01 --scale --classifier rf
The saved pipeline includes: ts_list (one TangentSpace per band), scaler, classifier.

7. (Optional) XGBoost with SHAP Explainability
Train XGBoost on the same tangent features and compute SHAP values.

bash
python tangent_xgboost_shap.py \
  --all_csv "D:/Data_cog/features/all_features_clean.csv" \
  --all_covs "D:/Data_cog/features/all_covs.npy" \
  --train_csv "D:/Data_cog/features/splits/train.csv" \
  --test_csv "D:/Data_cog/features/splits/test.csv" \
  --val_csv "D:/Data_cog/features/splits/val.csv" \
  --out_model "D:/Data_cog/models/tangent_xgb.joblib" \
  --reg 0.01 --nrounds 200 --early_stopping 20 --compute_shap
8. Predict Cognitive Load for a New Subject (Offline)
Given a subject’s covariance file (.npy) and features CSV, load the pre‑trained pipeline, transform the covariances, and output per‑epoch predictions (task probabilities and load scores).

bash
python predict_subject_load_offline_v2.py \
  --pipeline "D:/Data_cog/models/tangent_agg.joblib" \
  --covs "D:/Data_cog/features/covs_sub-02.npy" \
  --features_csv "D:/Data_cog/features/sub-02_features.csv" \
  --out_csv "D:/Data_cog/visualizations/sub-02_preds.csv"
9. Map Task Predictions to Load & Temporal Smoothing
Transform the raw predictions (task probabilities) into a continuous load score (low=0.0, medium=0.5, high=1.0) and apply a moving average.

bash
python map_and_smooth_preds.py --in "D:/Data_cog/visualizations/sub-02_preds.csv" --out_dir "D:/Data_cog/visualizations" --smooth 5
Output: *_mapped_sm5.csv containing load_score_smooth and load_label_smooth.

10. Real‑time Playback Visualization
Animate the smoothed load score over epochs (like a real‑time monitor).

bash
python realtime_playback_visualize.py --preds "D:/Data_cog/visualizations/sub-02_preds_v2_mapped_sm5.csv" --rate 1.0
