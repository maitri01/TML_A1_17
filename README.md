# TML25 Assignment 1 – Membership Inference Attack

## 📌 Additional Notes

Both a **Jupyter Notebook (`assignment1_solution.ipynb`)** and a **Python script (`assignment1_solution.py`)** are included in this repository.  
They implement the same pipeline, so you can choose either based on your workflow or system constraints.

---

## 📌 Overview

This repository contains our solution to Assignment 1 of the **Trustworthy Machine Learning** course (SS 2025), focused on implementing a **Membership Inference Attack (MIA)** on a ResNet-18 model. Our goal was to infer the membership status of data points in a private dataset by generating continuous confidence scores using various model-based and statistical features.

---

## 🧪 Files

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

## 🧠 Methodology

We extract both **engineered features** (confidence, entropy, RMIA score, etc.) and **deep features** from the penultimate layer of ResNet-18. We also compute **cosine similarity** to member and non-member mean embeddings. PCA is applied to reduce dimensionality. These features are combined and used to train an **XGBoost** classifier on the public dataset. The classifier is then applied to the private dataset to generate membership scores.

---

## ✅ Results

- **TPR@FPR=0.05**: 0.1293  
- **AUC**: 0.6645  
- Internal validation AUC: ~0.709


