
# Deep Anomaly Detection with Autoencoders

**A Comparative Study of Vanilla and Denoising Autoencoders for Unsupervised Anomaly Detection on High-Dimensional Tabular Data**

> Final project Machine Learning course
> Authors: Hayrunnisa Açıkgöz (23LYAZ032) · Burak Can Tarhan (23LBIM022)
> Istanbul Ticaret University

---

## 1. Overview

This project investigates whether a deep autoencoder, trained without any anomaly labels, can reliably detect anomalous samples in high-dimensional tabular data. We treat anomaly detection as a one-class learning problem: the model is exposed only to normal samples during training and learns to reconstruct them with low error. At inference time, samples that cannot be reconstructed accurately are flagged as anomalies.

We compare two autoencoder variants a Vanilla Deep Autoencoder (AE) and a Denoising Autoencoder (DAE) and benchmark them against two classical unsupervised baselines: Isolation Forest and One-Class SVM.

The case study uses network traffic data (UNSW-NB15 v2 and CSE-CIC-IDS2018), but the methodology and findings generalize to any tabular anomaly detection problem.

---

## 2. Problem Statement

**Goal:** Given a sample x ∈ ℝᵈ from an unknown distribution, decide whether x belongs to the "normal" data distribution or is an anomaly.

**Challenge:** The training set contains only normal samples. The model has zero information about what an anomaly looks like.

**Approach:** Learn a low-dimensional manifold representing the normal data. Use the distance between a sample and its projection onto this manifold (reconstruction error) as the anomaly score.

---

## 3. Methodology

### 3.1 Vanilla Deep Autoencoder

A symmetric encoder–decoder network compresses each d-dimensional input into an 8-dimensional latent vector and reconstructs it back.

```
Encoder: d → 128 → 64 → 32 → 8   (ReLU + BatchNorm)
Decoder: 8 → 32 → 64 → 128 → d   (ReLU + BatchNorm, Sigmoid output)
```

The model is trained to minimize the mean-squared reconstruction error:

```
L(x) = ‖x − decoder(encoder(x))‖²
```

The anomaly score for a new sample x is the per-sample reconstruction error.

### 3.2 Denoising Autoencoder

Same architecture, modified training procedure. During training, each input is corrupted with additive Gaussian noise before being passed to the encoder:

```
x̃ = clamp(x + ε, 0, 1),  ε ~ 𝒩(0, σ²),  σ = 0.05
L(x) = ‖x − decoder(encoder(x̃))‖²
```

The model is asked to recover the **clean** input from a **noisy** version. This forces the encoder to learn features that are invariant to small perturbations, producing a more robust representation of the normal manifold.

### 3.3 Threshold Selection

Two strategies are compared:
- **95th percentile** set the threshold at the 95th percentile of reconstruction errors on the normal validation set (high precision, low recall).
- **Youden Index** maximize J = TPR − FPR on the ROC curve (balanced precision/recall trade-off).

Empirically, the Youden threshold provides far better F1 scores and is reported as the primary result.

### 3.4 Baselines

- **Isolation Forest** (sklearn, 100 trees, contamination=0.01) partitions feature space by random splits; anomalies are isolated faster.
- **One-Class SVM** (sklearn, RBF kernel, ν=0.01) learns a tight boundary around the normal class.

Both baselines are trained on the same normal-only training set as the autoencoders.

---

## 4. Datasets

| | UNSW-NB15 v2 | CSE-CIC-IDS2018 |
|---|---|---|
| Total records | 1,986,745 | ~2.5M (sampled) |
| Anomaly ratio | 3.78% | ~77% |
| Categories | 9 anomaly classes + normal | 14 anomaly classes + normal |
| Raw features | 43 | 80 |

Both datasets are publicly available on Kaggle.

### 4.1 Preprocessing Pipeline

All preprocessing statistics are computed exclusively on the normal training subset to prevent data leakage:

1. Drop identifier and timestamp columns
2. Replace inf/NaN values with the median (computed on normal train data only)
3. Remove low-variance features (var ≤ 0.01)
4. Remove highly correlated features (|r| > 0.95) via upper-triangular pairwise inspection
5. Fit MinMaxScaler on normal train data only, then transform all splits

Final feature counts: UNSW-NB15 v2 → 31, CSE-CIC-IDS2018 → 41.

---

## 5. Training

- Framework: PyTorch
- Optimizer: Adam (lr=1e-3)
- Batch size: 1024
- Epochs: 50
- LR scheduler: ReduceLROnPlateau (factor=0.5, patience=5)
- Hardware: NVIDIA T4 GPU (Kaggle)

The data split is stratified 80/20. Only the **normal subset** of the training split is used to fit the autoencoder. The full test split (normal + anomalies) is held out for evaluation.

---

## 6. Results

### 6.1 Aggregate Metrics

| Model | Dataset | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| **Vanilla AE** | UNSW-NB15 v2 | **0.9874** | 0.7490 | 0.466 | 0.963 | 0.628 |
| DAE | UNSW-NB15 v2 | 0.9714 | 0.6155 | 0.238 | 0.958 | 0.381 |
| Isolation Forest | UNSW-NB15 v2 | 0.8510 | 0.1118 | | | |
| One-Class SVM | UNSW-NB15 v2 | 0.7380 | 0.2020 | | | |
| Vanilla AE | CSE-CIC-IDS2018 | 0.8126 | 0.9314 | 0.894 | 0.962 | 0.927 |
| **DAE** | CSE-CIC-IDS2018 | **0.8757** | **0.9542** | 0.918 | 0.944 | 0.931 |

All metrics use the Youden threshold.

### 6.2 Key Observations

1. **Both autoencoder variants outperform classical baselines** by a wide margin (ROC-AUC +0.14 over Isolation Forest, +0.25 over One-Class SVM on UNSW-NB15 v2).
2. **No single architecture dominates.** Vanilla AE wins on UNSW-NB15 v2, DAE wins on CSE-CIC-IDS2018. The choice depends on dataset characteristics.
3. **DAE specifically improves low-volume anomaly classes.** On CSE-CIC-IDS2018, recall on Brute Force-XSS rises from 0.694 → 0.980, SQL Injection from 0.750 → 0.812. The denoising objective helps the model retain sensitivity to rare patterns.
4. **Threshold selection is critical.** On CSE-CIC-IDS2018, the 95th percentile threshold yields a recall of only 0.39, while the Youden threshold raises recall to 0.96 with acceptable precision trade-off.

### 6.3 Per-Category Performance

Per-class recall plots and full breakdowns are available in `figures/`.

---

## 7. Findings

- **Reconstruction error is a strong unsupervised anomaly signal** for high-dimensional tabular data, even without any anomaly labels during training.
- **Denoising regularization is a low-cost intervention** that can substantially improve detection of low-volume anomaly classes.
- **Cross-domain transfer fails** when feature spaces differ a model trained on UNSW-NB15 cannot be directly applied to CSE-CIC-IDS2018 (ROC-AUC collapses to 0.24). Domain-specific training remains necessary.
- **Threshold calibration** via Youden Index converts a high-precision-but-low-recall detector into a balanced one without retraining.

---

## 8. Limitations & Future Work

- The current models operate on aggregated flow features, not raw packet streams. Sequence models (LSTM, Transformer) could capture temporal dependencies.
- Variational Autoencoders (VAE) would provide probabilistic anomaly scores with calibrated uncertainty.
- Adaptive thresholding (e.g., online recalibration on streaming data) is not addressed.
- Inference latency was not measured relevant for real-time deployment.
- Class imbalance handling at the threshold level only; cost-sensitive learning could further improve rare-class recall.

---

## 9. Project Structure

```
project/
├── README.md
├── notebooks/
│  └── unsw_pipeline.ipynb    # Preprocessing + training + evaluation
│  └── cse_pipeline.ipynb
├── model/
│  ├── unsw_ae_weights.pt    # Trained autoencoder weights
│  ├── scaler_unsw.pkl      # Fitted MinMaxScaler
│  └── keep_cols_unsw.pkl    # Selected feature names
├── demo/
│  ├── backend.py        # FastAPI inference server
│  └── index.html        # Web demo UI
├── figures/
│  ├── roc_comparison_unsw.png
│  ├── roc_comparison_cse.png
│  ├── category_comparison_unsw.png
│  ├── category_comparison_cse.png
│  ├── error_distribution_unsw.png
│  ├── error_distribution_cse.png
│  ├── training_curve_unsw.png
│  └── training_curve_cse.png
└── requirements.txt
```

---

## 10. How to Reproduce

### 10.1 Train the Models

Open `notebooks/unsw_pipeline.ipynb` on Kaggle (or any environment with GPU access) and run all cells. The notebook handles:
- Loading the raw dataset
- Preprocessing (variance filter, correlation filter, MinMax scaling)
- Training both Vanilla AE and DAE for 50 epochs
- Computing reconstruction errors on the test set
- Evaluating with ROC-AUC, PR-AUC, F1, and per-category recall
- Saving model weights and preprocessing artifacts to `model/`

Repeat with `notebooks/cse_pipeline.ipynb` for the second dataset.

### 10.2 Run the Live Demo

```bash
cd demo/
pip install -r ../requirements.txt
python backend.py
```

Open `http://localhost:8000` in a browser. Paste a raw feature vector or click one of the preset example buttons. The model returns:
- A binary verdict (Safe / Malicious)
- A confidence score
- The exact reconstruction error

### 10.3 Dependencies

```
torch
scikit-learn
pandas
numpy
joblib
fastapi
uvicorn
matplotlib
```

---

## 11. References

- An, J., & Cho, S. (2015). *Variational Autoencoder based Anomaly Detection using Reconstruction Probability*. SNU Data Mining Center.
- Vincent, P., et al. (2008). *Extracting and Composing Robust Features with Denoising Autoencoders*. ICML.
- Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). *Isolation Forest*. ICDM.
- Schölkopf, B., et al. (2001). *Estimating the Support of a High-Dimensional Distribution*. Neural Computation.
- Youden, W. J. (1950). *Index for Rating Diagnostic Tests*. Cancer.
- Moustafa, N., & Slay, J. (2015). *UNSW-NB15: A Comprehensive Data Set for Network Intrusion Detection Systems*. MilCIS.
- Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). *Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization*. ICISSP.

---

## 12. License

This project is submitted as coursework for academic purposes.
