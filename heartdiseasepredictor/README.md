# ❤️ Heart Disease Diagnostic Predictor

## 📌 Project Objective
This project applies machine learning to the UCI Heart Disease dataset to predict the presence of heart disease in patients. The focus of this pipeline is rigorous data hygiene, handling complex mixed-type features, and tuning classification thresholds mathematically to maximize diagnostic sensitivity.

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-Learn, YData Profiling, Matplotlib
* **Techniques:** IQR Outlier Capping, Advanced ColumnTransformer Pipelines, Principal Component Analysis (PCA), Youden's J Statistic

## 🏗️ Preprocessing Architecture
The dataset contained missing values encoded as zeros and extreme outliers. A highly customized `ColumnTransformer` was built to handle distinct feature groups:
* **Group 1 (Cholesterol, Oldpeak, Max HR):** Imputed via `KNNImputer` (distance-weighted) and scaled.
* **Group 2 (Categorical variables like ECG & Chest Pain):** Modal imputation and `OneHotEncoder`.
* **Group 3 (Resting Blood Pressure):** Outliers capped mathematically using the Interquartile Range (IQR). Imputed via KNN, normalized using `PowerTransformer`, and scaled.
* **Group 4 & 5 (Slope & Demographics):** Mode imputation combined with `OrdinalEncoder`. 

## 🤖 Modeling & Dimensionality Reduction
* Applied **PCA** to capture 95% of the variance, mitigating multicollinearity.
* Trained a **Logistic Regression** model using an `L1` penalty (Lasso) via the `liblinear` solver to enforce feature selection.
* Optimized hyperparameters ($C$ value, penalty) using 5-fold `GridSearchCV` scoring on `roc_auc`.

## 📈 Threshold Optimization
In medical diagnostics, missing a sick patient is dangerous. Instead of accepting the default classification threshold, the **ROC Curve** was extracted, and **Youden's J Statistic** ($J = TPR - FPR$) was calculated to mathematically identify the optimal probability threshold, vastly improving the model's diagnostic reliability.