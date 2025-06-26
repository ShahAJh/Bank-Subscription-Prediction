# 📊 Bank Subscription Prediction

A Machine Learning project to predict whether a customer will subscribe to a bank term deposit based on historical telemarketing data.

---

## 📌 Overview

Banks often run telemarketing campaigns to promote term deposits. This project applies supervised ML models to predict if a client will subscribe (`yes` or `no`) to a term deposit offer. It compares **Logistic Regression**, **K-Nearest Neighbors (KNN)**, **Support Vector Machine (SVM)**, and **Random Forest**, with class imbalance handling and threshold tuning.

---

## 📂 Dataset

- **Source:** [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
- **Records:** 45,211 rows × 17 columns
- **Target Variable:** `y` (subscription status: `yes` or `no`)
- **Features:**
  - **Demographic:** `age`, `job`, `marital`, `education`
  - **Financial:** `balance`, `housing`, `loan`
  - **Campaign-related:** `duration`, `campaign`, `previous`, `pdays`

---

## 🛠️ Methodology

### 🔹 1. Data Preprocessing
- Removed `unknown` values (e.g., job, education)
- Dropped non-informative features (`contact`, `poutcome`)
- Label Encoding for categorical variables
- Standardization of numerical features

### 🔹 2. Handling Class Imbalance
- Original imbalance ratio: **7.6:1**
- Applied **Tomek Links** to remove borderline majority instances
- After cleaning: **7.4:1** ratio

| Phase            | Class 0 (No) | Class 1 (Yes) | Imbalance Ratio |
|------------------|--------------|----------------|------------------|
| Before Tomek     | 38,172       | 5,021          | 7.6:1            |
| After Tomek      | 25,878       | 3,515          | 7.4:1            |

### 🔹 3. Model Training
- Algorithms: Logistic Regression, KNN, SVM, Random Forest
- Cross-validation: 5-fold Stratified
- Evaluation Metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC
- Applied threshold tuning for Random Forest (threshold = 0.25)

---

## 📊 Results

| Model                       | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|-----------------------------|----------|-----------|--------|----------|----------|
| Logistic Regression         | 0.891    | 0.576     | 0.252  | 0.351    | 0.870    |
| SVM                         | 0.897    | 0.596     | 0.356  | 0.446    | 0.864    |
| KNN                         | 0.896    | 0.598     | 0.331  | 0.426    | 0.833    |
| **Random Forest**           | **0.901**| **0.617** | 0.385  | 0.474    | **0.919**|
| **Random Forest (0.25 T)**  | 0.876    | 0.480     | **0.782** | **0.595** | 0.919    |

> ✅ **Random Forest** with adjusted threshold yielded the best recall, improving model sensitivity.

---

## 📈 Visualizations

- **Graph 1:** Bar Chart – Model Performance Comparison
- **Graph 2:** Feature Importance (Random Forest)
- **Graph 3:** ROC Curve – All models
- **Graph 4:** Precision-Recall Curve
- **Graph 5:** Confusion Matrices

---

## 🔍 Key Takeaways

- **Random Forest** performed best across most metrics.
- **Threshold adjustment** significantly boosted recall – ideal for minimizing false negatives.
- **Important features:** `duration`, `pdays`, `previous`, and `campaign`.
- **Class imbalance** still impacts performance — future work should include oversampling.

---

## 🚀 Future Work

- Apply **SMOTE** or **ADASYN** for synthetic sampling
- Perform **hyperparameter tuning** (GridSearchCV, Optuna)
- Try **boosting models** (XGBoost, LightGBM)
- Integrate into a **web dashboard** or marketing automation system

---

## 💻 How to Run

### 🔧 Setup

```bash
git clone https://github.com/ShahAJh/bank-subscription-prediction.git
