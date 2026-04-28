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

The pipeline has been validated on a multi‑task dataset (N‑back, MATB, Flanker, PVT, Resting State) with 29 subjects and achieves >96% test accuracy.
