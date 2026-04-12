# 📊 Alternative Credit Scoring for Financial Inclusion in Cambodia

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)
![Academic Project](https://img.shields.io/badge/Type-Academic%20Project-lightgrey)

## 🚀 Overview
This project builds an **alternative credit scoring model** for **unbanked individuals in Cambodia** using machine learning and alternative financial indicators instead of relying only on traditional credit history.

The goal is to support **financial inclusion** by predicting **loan default risk** more fairly for people working in the informal sector.

---

## 🎯 Key Highlights
- Designed for the **Cambodian financial context**
- Addresses the **lack of formal credit history**
- Uses engineered features such as **Debt-to-Income Ratio**, **Stability Score**, and **Interest Burden**
- Compares **Logistic Regression**, **Decision Tree**, and **Random Forest**
- Best result achieved with **Random Forest**

---

## 📂 Dataset
- **Source:** Kaggle Credit Risk Dataset
- **Original size:** 32,581 rows
- **Target variable:** `loan_status`
  - `0` = Non-default
  - `1` = Default
---

## 🧹 Data Preparation
The dataset was cleaned and filtered to improve realism and consistency.

### Main preprocessing steps
- Removed rows with missing `person_emp_length`
- Filled missing `loan_int_rate` values using the median
- Removed unrealistic borrower ages
- Removed unrealistic employment history records
- Retained meaningful outliers that may represent real high-risk borrowers

### Final cleaned dataset
- **Rows after cleaning:** 31,650
- **Missing values remaining:** None

---

## 🧠 Feature Engineering
To better reflect Cambodian borrower conditions, the project introduced localized and engineered features:

- **Debt-to-Income Ratio (DTI)**  
  Measures how much income is consumed by debt

- **Stability Score**  
  Combines employment length and credit history length

- **Interest Burden**  
  Estimates how much interest pressure affects repayment ability

- **Land Security Mapping**  
  Mapped housing ownership into Cambodian-style categories:
  - Hard Title
  - Soft Title
  - No Title

---

## 🤖 Models Used
Three machine learning models were trained and evaluated:

1. **Logistic Regression**
2. **Decision Tree**
3. **Random Forest**

---

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|------|---------:|----------:|-------:|---------:|
| Logistic Regression | 82.26% | 71.18% | 35.41% | 47.30% |
| Decision Tree | 87.82% | 72.61% | 73.58% | 73.09% |
| **Random Forest** | **92.21%** | **93.46%** | 70.27% | **80.22%** |

📌 **Best Model: Random Forest**  
It achieved the highest overall accuracy, precision, and F1-score, making it the recommended model for this project.

---

## 🔍 Key Insights
- Borrowers with **previous default history** have much higher risk
- **Housing stability** is strongly linked to repayment behavior
- **Loan purpose** influences default probability
- The dataset is **moderately imbalanced**

---

## ⚙️ Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Jupyter Notebook

---

## 📁 Project Structure
```text
credit-scoring-project/
│
├── data/              # Raw and cleaned dataset
├── notebooks/         # EDA, preprocessing, and model training notebooks
├── models/            # Saved model files (.pkl)
├── src/               # Helper scripts / reusable code
├── app/               # Optional Gradio interface
├── requirements.txt
└── README.md
