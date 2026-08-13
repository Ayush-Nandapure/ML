# 🏦 Credit Risk & Loan Default Predictor

## 📌 Project Objective
This project builds a machine learning pipeline to predict loan defaults. In the banking sector, the cost of missing a defaulter (False Negative) is significantly higher than rejecting a good applicant (False Positive). Therefore, this pipeline abandons standard accuracy metrics in favor of optimizing **Recall** and minimizing actual **Financial Loss** through threshold calibration.

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-Learn, XGBoost, Matplotlib, Seaborn
* **Techniques:** Feature Engineering, Class Weight Balancing, Pipeline & ColumnTransformer, Cost-Benefit Analysis

## 🏗️ Pipeline Architecture
1. **Data Cleaning:** Handled missing employment lengths and interest rates. Filtered out impossible outliers (e.g., employment length > 60 years).
2. **Feature Engineering:** Engineered logical ratios like `Age_to_Credit_History` to identify inconsistent applications.
3. **Preprocessing:** Utilized `ColumnTransformer` to apply `KNNImputer` and `PowerTransformer` (Yeo-Johnson) to highly skewed continuous variables, and `OneHotEncoder` to nominal categories.
4. **Modeling:** Trained Logistic Regression, Random Forest, and XGBoost classifiers. Addressed the heavy class imbalance (defaults made up ~20% of data) using `class_weight='balanced'` and `scale_pos_weight`.

## 📊 Key Results & Business Impact
* **XGBoost** emerged as the top-performing model, achieving a ROC-AUC score of **~0.94**.
* **Financial Threshold Calibration:** Standard ML models use a 0.50 probability cutoff. By mapping False Positives and False Negatives to estimated dollar amounts ($500 vs $5,000), a custom cost-function curve determined that setting the decision threshold to **0.31** minimized total financial loss for the institution.
* **Feature Importance:** Loan-to-Income ratio, Loan Grade, and Total Income were identified as the strongest predictors of default risk.