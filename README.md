# Credit Risk Prediction from Scratch (German Credit Dataset)

## 📌 Project Overview
This project implements a **credit risk prediction model from scratch in Python** using the **German Credit Dataset** from the UCI Machine Learning Repository.  
The objective is to predict whether a loan applicant represents a **good** or **bad** credit risk using a **regularized logistic regression model implemented without high-level ML libraries**.

The project focuses on:
- Core machine learning fundamentals
- Mathematical intuition behind classification
- Gradient-based optimization
- Regularization to control overfitting
- Business-aware evaluation metrics

---

## 🎯 Problem Statement
Banks and financial institutions must assess credit risk accurately to reduce defaults while maintaining approval rates.  
Given applicant attributes such as credit history, employment, loan purpose, and personal details, the goal is to:

> **Predict whether an applicant is creditworthy (good) or high-risk (bad).**

This is a **binary classification problem** with real-world cost asymmetry.

---

## 📊 Dataset Description
- **Dataset:** German Credit Data
- **Source:** UCI Machine Learning Repository  
- **Samples:** 1,000
- **Features:** 20 (categorical + numerical)
- **Target Variable:**
  - `1` → Good credit risk  
  - `0` → Bad credit risk  

The dataset reflects realistic financial decision constraints and bias trade-offs.

---

## ⚙️ Methodology & Approach

### 1. Data Preprocessing
- Categorical features encoded numerically
- Numerical features normalized to improve gradient descent stability
- Target variable binarized
- Train-test split for unbiased evaluation

---

### 2. Model Architecture (From Scratch)

A **regularized logistic regression classifier** is implemented entirely from scratch.

#### Linear Model
\[
z = Xw + b
\]

#### Sigmoid Activation
\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

#### Prediction
\[
\hat{y} = \sigma(Xw + b)
\]

---

### 3. Loss Function with L2 Regularization

To reduce overfitting and stabilize learned coefficients, **L2 (Ridge) regularization** is added.

#### Binary Cross-Entropy Loss
\[
L_{BCE} = -\frac{1}{n} \sum \left[y \log(\hat{y}) + (1-y)\log(1-\hat{y})\right]
\]

#### L2 Regularization Term
\[
L_{L2} = \frac{\lambda}{2n} \sum w^2
\]

#### Final Objective Function
\[
L = L_{BCE} + L_{L2}
\]

where:
- \( \lambda \) controls regularization strength
- Penalizes large weights
- Encourages smoother, more generalizable solutions

---

### 4. Gradient Descent Optimization

#### Gradient Updates
\[
\frac{\partial L}{\partial w} =
\frac{1}{n} X^T(\hat{y} - y) + \frac{\lambda}{n} w
\]

\[
\frac{\partial L}{\partial b} =
\frac{1}{n} \sum (\hat{y} - y)
\]

Weights are iteratively updated using gradient descent until convergence.

---

## 🏋️ Training & Convergence
- Loss is tracked across iterations
- Regularization prevents coefficient explosion
- Convergence is smooth and stable
- L2 term improves generalization on unseen data

Loss curves demonstrate steady optimization without oscillations.

---

## 📈 Model Evaluation

The model is evaluated using **business-relevant metrics**:

- **Accuracy** – Overall prediction correctness
- **ROC-AUC** – Ranking quality of credit risk
- **Loss Curve** – Optimization stability
- **ROC Curve** – Threshold-independent performance

> **ROC-AUC is prioritized**, as false approvals of risky borrowers carry higher financial cost.

---

## 🧠 Business Insights
- L2 regularization improves model robustness in noisy financial data
- Penalizing large weights reduces over-reliance on any single feature
- Linear models remain strong baselines in regulated domains
- Proper regularization helps balance risk sensitivity and approval rate

---

## 🛠️ Tech Stack
- Python
- NumPy
- Pandas
- Matplotlib
- Jupyter Notebook

**All learning logic implemented from scratch (no sklearn models)**

