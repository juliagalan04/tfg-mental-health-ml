# Mental Health Prediction using Lifestyle Data

This project explores the use of machine learning techniques to predict mental health indicators from lifestyle data.

The main goal is to understand which variables related to daily habits can be used to estimate indicators such as stress level and work-life balance.

---

## Datasets

The project evaluates three different datasets:

- **Student Lifestyle Dataset**
- **Sleep Health and Lifestyle Dataset**
- **Lifestyle and Wellbeing Dataset (Kaggle)**

Each dataset was analyzed to assess its suitability for a realistic machine learning problem.

---

## Key Findings from Dataset Analysis

- The **Student Lifestyle Dataset** presented *data leakage*, leading to artificially perfect results.
- The **Sleep Health Dataset** contained a target variable derived directly from other features, making the problem trivial.
- The **Lifestyle and Wellbeing Dataset** was selected as the most suitable, as it represents a more realistic and complex problem.

---

## Approach

- Data preprocessing:
  - Standardization of numerical features
  - One-hot encoding of categorical variables

- Models evaluated:
  - Logistic Regression
  - Support Vector Machine (SVM)
  - Random Forest
  - Gradient Boosting
  - K-Nearest Neighbors (KNN)

- Evaluation metrics:
  - Accuracy
  - Macro F1-score
  - Cross-validation (Stratified K-Fold)

---

## Experiments

Several experiments were conducted:

1. **Work-Life Balance Prediction**
   - Using discretized scores (balanced with quantiles)
   - High performance achieved (~0.95–0.99 Macro F1)

2. **Impact of Stress Variable**
   - Comparison of models **with and without `DAILY_STRESS`**
   - Demonstrated strong dependence on this variable

3. **Stress Prediction**
   - Using `DAILY_STRESS` as the target
   - Significantly lower performance (~0.5 Macro F1)
   - Indicates higher complexity and variability of stress

---

## Key Insights

- Work-life balance is easier to predict from lifestyle data than stress.
- Stress appears to depend on more complex or unobserved factors.
- Careful feature selection is critical to avoid data leakage.
- Balanced class distributions improve model reliability.

---

## Project Structure

src/
├── 01_student_lifestyle_classification.py
├── 02_sleep_health_trivial_target.py
├── 03_wellbeing_balance_with_stress.py
├── 04_wellbeing_balance_without_stress.py
└── 05_daily_stress_prediction.py
Each script corresponds to a different dataset or experimental setup.

---

## Author

Julia Galán
