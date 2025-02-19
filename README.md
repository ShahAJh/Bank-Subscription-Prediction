# Bank Subscription Prediction 📊  
Machine Learning project to predict whether a customer will subscribe to a bank term deposit based on historical marketing data.

## 📌 Project Overview
Banks conduct telemarketing campaigns to sell term deposits. This project applies **Machine Learning** techniques to predict whether a customer will subscribe (`yes` or `no`). The project compares different models, including **Logistic Regression, K-Nearest Neighbors, Support Vector Machine, and Random Forest**.

## 📂 Dataset
- **Source:** [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
- **Size:** 45,211 rows × 17 columns
- **Target Variable:** `y` (Subscription: `yes` or `no`)
- **Features:**
  - **Demographic Data:** Age, job, marital status, education
  - **Financial Information:** Balance, housing loan, personal loan
  - **Campaign Data:** Contact method, number of previous contacts, duration of last call

## 🛠️ Methodology
1. **Data Preprocessing:**
   - Removed missing values (`unknown` values in job & education).
   - Dropped non-essential features (`contact`, `poutcome`).
   - Encoded categorical variables using Label Encoding.
   - Standardized numerical features using StandardScaler.
   
2. **Handling Class Imbalance:**
   - Applied **Tomek Links** to remove redundant majority class instances.

3. **Model Training & Evaluation:**
   - **Models Compared:** Logistic Regression, KNN, SVM, Random Forest
   - **Cross-Validation:** Stratified K-Fold (5-folds)
   - **Performance Metrics:** Accuracy, Precision, Recall, F1-score, ROC-AUC
   - **Threshold Tuning:** Adjusted probability threshold to **0.25** for better recall.

## 📊 Results
| Model                | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|----------------------|----------|-----------|--------|-----------|----------|
| Logistic Regression | 89.18%   | 59.93%    | 20.91% | 31.00%    | 86.23%   |
| K-Nearest Neighbors | 89.00%   | 55.15%    | 29.05% | 38.03%    | 79.71%   |
| Support Vector Machine | 89.33%   | 62.92%    | 19.99% | 30.33%    | 83.27%   |
| **Random Forest (Best)** | **89.82%**  | **63.64%**   | **29.13%** | **39.95%** | **91.88%** |

✅ **Random Forest performed the best**, but recall is still low.  
🚀 **Future Work:** SMOTE oversampling, hyperparameter tuning, testing XGBoost.

## 🔧 Installation & Usage
1. Clone this repository:
   ```bash
   git clone https://github.com/ShahAJh/bank-subscription-prediction.git
