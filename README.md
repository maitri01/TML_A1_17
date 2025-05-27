# TML25 Assignment 1 – Membership Inference Attack

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
  - Validation experiments and visualizations

- `test.csv`  
  → Final submission file with membership confidence scores for the private dataset.

- `README.md`  
  → This file. Summarizes the approach, how to run the code, and the results.

---

## 🧠 Methodology

We extract both **engineered features** (confidence, entropy, RMIA score, etc.) and **deep features** from the penultimate layer of ResNet-18. We also compute **cosine similarity** to member and non-member mean embeddings. PCA is applied to reduce dimensionality. These features are combined and used to train an **XGBoost** classifier on the public dataset. The classifier is then applied to the private dataset to generate membership scores.

---

## ✅ Results

- **TPR@FPR=0.05**: 0.1293  
- **AUC**: 0.6645  
- Internal validation AUC: ~0.709

---

## 🧪 How to Run

1. Clone the repository.
2. Ensure you have the required packages installed:
