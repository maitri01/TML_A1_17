# TML25 Assignment 1 – Membership Inference Attack

## 📌 Additional Notes

Both a **Jupyter Notebook (`assignment1_solution.ipynb`)** and a **Python script (`assignment1_solution.py`)** are included in this repository.  
They implement the same pipeline, so you can choose either based on your workflow or system constraints.

---

## Overview

This repository contains our solution to Assignment 1 of the **Trustworthy Machine Learning** course (SS 2025), focused on implementing a **Membership Inference Attack (MIA)** on a ResNet-18 model. Our goal was to infer the membership status of data points in a private dataset by generating continuous confidence scores using various model-based and statistical features.

---

## 👥 Team #17

Maitri Shah - 7075780         
Yashashri Balwaik - 7075733

---

## Files

- `assignment1_solution.ipynb`  
  → Jupyter notebook containing the full implementation:
  - Loading the model and datasets
  - Feature extraction (logits, deep features, entropy, etc.)
  - PCA for dimensionality reduction
  - XGBoost training and prediction
  - Submission generation and evaluation
  - Validation

- `test.csv`  
  → Final submission file with membership confidence scores for the private dataset.

- `Report`      
  → A detailed **report** outlining our approach, feature engineering, design decisions, evaluation results, and alternative strategies is included in the repository as `TML_A1_17 Report.pdf`. It summarizes the full methodology and supports the code implementation for grading and review purposes.

---

## Approach

### Step 1: Feature Extraction

We used a modified ResNet-18 to extract:
- **Logits** and **deep features** (from the second last layer)
- **Engineered statistics**, such as:
  - Confidence, cross-entropy loss, entropy, margin
  - Rank of the true class, RMIA score
  - Logit differences, norms, top-k entropy
- **Cosine similarity** between the sample's deep feature and average deep features of known members and non-members (from the public dataset)

### Step 2: Dimensionality Reduction

To reduce deep features and eliminate noise:
- Applied **PCA (64 components)** to the 512-dimensional deep features
- Avoided overfitting and made the features more informative

### Step 3: Classification

- Combined **engineered features**, **PCA-transformed deep features**, and **logits**
- Trained an **XGBoost classifier** on publicly available data where membership is known
- Used the trained model to predict **membership confidence scores** on the private data

---


## ✅ Results

- **TPR@FPR=0.05**: 0.1293  
- **AUC**: 0.6645  
- Internal validation AUC: ~0.709


