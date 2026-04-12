import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

SEED = 42
DATA_PATH = "../data/Wellbeing_and_lifestyle_data_Kaggle.csv"
OUTPUT_DIR = "../outputs"

# Create output folder
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------
# 1. Load data
# --------------------------------------------------
df = pd.read_csv(DATA_PATH)

# Drop useless column
df = df.drop(columns=["Timestamp"])


# --------------------------------------------------
# 2. Create target
# --------------------------------------------------
def categorize_score(x):
    if x < 550:
        return "Low"
    elif x < 650:
        return "Moderate"
    else:
        return "High"

df["Balance_Category"] = df["WORK_LIFE_BALANCE_SCORE"].apply(categorize_score)


# --------------------------------------------------
# 3. Show and save class distribution
# --------------------------------------------------
class_counts = df["Balance_Category"].value_counts()
class_props = df["Balance_Category"].value_counts(normalize=True)

print("\n=== CLASS COUNTS ===")
print(class_counts)

print("\n=== CLASS PROPORTIONS ===")
print(class_props)

# Save class counts to CSV
class_counts.to_csv(os.path.join(OUTPUT_DIR, "class_counts.csv"), header=["Count"])
class_props.to_csv(os.path.join(OUTPUT_DIR, "class_proportions.csv"), header=["Proportion"])

# Plot class distribution
plt.figure(figsize=(8, 5))
class_counts.plot(kind="bar")
plt.title("Class Distribution")
plt.xlabel("Balance Category")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "class_distribution.png"), dpi=300, bbox_inches="tight")
plt.close()


# --------------------------------------------------
# 4. Features and target
# --------------------------------------------------
X = df.drop(columns=["WORK_LIFE_BALANCE_SCORE", "Balance_Category"])
y = df["Balance_Category"]


# --------------------------------------------------
# 5. Column types
# --------------------------------------------------
num_cols = [col for col in X.columns if X[col].dtype != "object"]
cat_cols = [col for col in X.columns if X[col].dtype == "object"]


# --------------------------------------------------
# 6. Preprocessing
# --------------------------------------------------
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])


# --------------------------------------------------
# 7. Train-test split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=SEED,
    stratify=y
)


# --------------------------------------------------
# 8. Models to test
# --------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=SEED
    ),
    "Logistic Regression Balanced": LogisticRegression(
        max_iter=2000,
        random_state=SEED,
        class_weight="balanced"
    ),
    "SVM": SVC(random_state=SEED),
    "Random Forest": RandomForestClassifier(random_state=SEED),
    "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
    "KNN": KNeighborsClassifier()
}

results = []
fitted_models = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


# --------------------------------------------------
# 9. Train, evaluate, and cross-validate all models
# --------------------------------------------------
for name, model in models.items():

    pipe = Pipeline([
        ("prep", preprocessor),
        ("clf", model)
    ])

    # Train on training split
    pipe.fit(X_train, y_train)

    # Predict on test split
    y_pred = pipe.predict(X_test)

    # Test metrics
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    # Cross-validation on full dataset
    cv_scores = cross_val_score(
        pipe,
        X,
        y,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1
    )

    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Macro_F1": macro_f1,
        "CV_Macro_F1_Mean": cv_mean,
        "CV_Macro_F1_STD": cv_std
    })

    fitted_models[name] = pipe


# --------------------------------------------------
# 10. Compare results
# --------------------------------------------------
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="Macro_F1", ascending=False).reset_index(drop=True)

print("\n=== MODEL COMPARISON ===")
print(results_df)

# Save results table
results_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)

# Plot model comparison
plt.figure(figsize=(10, 6))
plt.bar(results_df["Model"], results_df["Macro_F1"])
plt.title("Model Comparison by Macro F1-score")
plt.xlabel("Model")
plt.ylabel("Macro F1-score")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "model_comparison_macro_f1.png"), dpi=300, bbox_inches="tight")
plt.close()


# --------------------------------------------------
# 11. Best model
# --------------------------------------------------
best_model_name = results_df.loc[0, "Model"]
best_model = fitted_models[best_model_name]

print(f"\nBest model: {best_model_name}")


# --------------------------------------------------
# 12. Classification report
# --------------------------------------------------
y_pred_best = best_model.predict(X_test)

print("\n=== CLASSIFICATION REPORT ===")
report = classification_report(y_test, y_pred_best)
print(report)

# Save classification report
with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
    f.write(report)


# --------------------------------------------------
# 13. Confusion matrix
# --------------------------------------------------
labels = ["Low", "Moderate", "High"]
cm = confusion_matrix(y_test, y_pred_best, labels=labels)

print("\n=== CONFUSION MATRIX ===")
print(pd.DataFrame(cm, index=labels, columns=labels))

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot()
plt.title(f"Confusion Matrix - {best_model_name}")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix_best_model.png"), dpi=300, bbox_inches="tight")
plt.close()


# --------------------------------------------------
# 14. Feature importance for best model
# --------------------------------------------------
prep = best_model.named_steps["prep"]
clf = best_model.named_steps["clf"]

feature_names = prep.get_feature_names_out()

print("\n=== TOP IMPORTANT FEATURES ===")

if hasattr(clf, "coef_"):

    importance = np.mean(np.abs(clf.coef_), axis=0)

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False)

    print(importance_df.head(15))
    importance_df.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False)

    top_n = 15
    top_features = importance_df.head(top_n).sort_values(by="Importance")
    plt.figure(figsize=(10, 6))
    plt.barh(top_features["Feature"], top_features["Importance"])
    plt.title(f"Top {top_n} Important Features - {best_model_name}")
    plt.xlabel("Mean Absolute Coefficient")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "top_features_best_model.png"), dpi=300, bbox_inches="tight")
    plt.close()

elif hasattr(clf, "feature_importances_"):

    importance = clf.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False)

    print(importance_df.head(15))
    importance_df.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False)

    top_n = 15
    top_features = importance_df.head(top_n).sort_values(by="Importance")
    plt.figure(figsize=(10, 6))
    plt.barh(top_features["Feature"], top_features["Importance"])
    plt.title(f"Top {top_n} Important Features - {best_model_name}")
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "top_features_best_model.png"), dpi=300, bbox_inches="tight")
    plt.close()

else:
    print("This model does not provide direct feature importance.")

print(f"\nAll figures and result files have been saved in: {OUTPUT_DIR}")