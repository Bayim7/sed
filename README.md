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
- **Threshold