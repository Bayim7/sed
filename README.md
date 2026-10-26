# Multi-Label Sound Event Detection (SED) in Domestic Environments

**Machine Learning Pattern Classification (MLPC) 2026 – Capstone Project**
**Team: Validated Encoders**

This repository contains the complete pipeline and final deliverables for a semester-long Sound Event Detection (SED) challenge. The objective was to build a robust machine learning system capable of identifying 15 overlapping domestic sound classes (e.g., running water, microwave, light switch, footsteps) in continuous audio recordings, handling extreme class imbalances and noisy crowdsourced annotations.

## 📌 Project Overview

The project was structured across five continuous phases simulating a real-world AI startup lifecycle (Kepler Intelligent Audio Labs):

1. **Data Collection:** Recording diverse domestic acoustic scenes.
2. **Data Annotation:** Bounding-box temporal labeling via Label Studio.
3. **Exploratory Data Analysis (EDA):** Feature extraction (MFCCs, Mel-spectrograms), statistical analysis, and dimensionality reduction (t-SNE).
4. **Classification:** Aggregating overlapping annotations and training baseline models.
5. **Challenge Phase:** Developing a continuous multi-label SED pipeline, applying temporal post-processing, and evaluating against a hidden test set.

## 💡 My Core Contributions

While this was a collaborative effort encompassing end-to-end machine learning engineering, my specific focus and primary contributions included:

- **Evaluation Strategy & Metric Design:** Championed the use of Segment-based Macro-F1 to combat a severe 47x class imbalance, preventing models from artificially inflating accuracy by ignoring rare transient classes.
- **Theoretical Performance Modeling:** Conducted deep error analysis to establish a hard theoretical performance ceiling (0.55–0.61 Macro-F1) by isolating structural bottlenecks: inherent label noise (single-annotator files), rare class scarcity, and representation limits (low-pass filtering of transients via 1-second feature aggregation).
- **Technical Writing & Reporting:** Authored the comprehensive LaTeX technical reports and designed the executive slide decks, synthesizing complex hyperparameter sweeps, ablation studies, and qualitative spectrogram analyses into clear, professional deliverables.

## ⚙️ System Architecture & Pipeline

The final challenge submission replaces greedy, axis-aligned Decision Trees with a highly tuned Histogram-based Gradient Boosting (HistGB) ensemble.

- **Feature Representation:** Audio is processed into 1-second windows with a 0.5s hop size. Each window is represented by a 960-dimensional vector aggregating the mean, standard deviation, min, and max of base acoustic features (MFCCs, Spectral Flux, Rolloff, etc.).
- **Multi-Label Classification:** Using a Binary Relevance wrapper, 15 independent HistGB models are trained (one per class).
- **Threshold Calibration:** Because rare-class posteriors sit far below the default 0.5 decision boundary, per-class thresholds were rigorously tuned on the validation set, unlocking massive gains for transient events.
- **Temporal Refinement:** A sliding Median Filter ($w=3$) was evaluated over the prediction timeline to vote out isolated false positives and bridge temporal gaps in continuous sounds.

## 📊 Key Results & Insights

- **+64% Performance Lift:** The tuned HistGB pipeline achieved a Test Macro-F1 of **0.545**, massively outperforming the baseline Decision Tree (0.316) and completely clearing the empirical prior floor (0.053).
- **The Transient Bottleneck:** Discovered that while gradient boosting easily detects sustained broadband sounds (vacuum, running water), 1-second temporal aggregation inherently washes out brief transients (light switch clicks, door shutting). This proved that the input representation, rather than model capacity, was the true limiting factor.
- **Deep Learning Extensions:** Explored raw-waveform and log-mel spectrogram architectures (CNNs, CRNNs, and a frozen Audio Spectrogram Transformer) to successfully bypass the aggregation bottleneck, pushing performance to ~**0.605** Macro-F1.

## 🛠 Technologies & Tools

- **Languages:** Python
- **Machine Learning:** scikit-learn, XGBoost, HistGradientBoosting
- **Audio Processing:** Librosa, FFmpeg
- **Evaluation & Visualization:** Matplotlib, Seaborn, Pandas, t-SNE
- **Documentation:** LaTeX, Overleaf

> **Note:** The raw audio dataset and proprietary hidden test labels are restricted by course licensing and are not included in this repository. The provided `.npz` feature files and metadata can be used to reproduce the classification pipeline.