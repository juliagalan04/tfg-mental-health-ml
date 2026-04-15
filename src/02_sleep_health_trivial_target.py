import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

SEED = 42
DATA_PATH = "../data/Sleep_health_and_lifestyle_dataset.csv"

xgb_available = True
cat_available = True

try:
    from xgboost import XGBClassifier
except Exception:
    xgb_available = False

try:
    from catboost import CatBoostClassifier
except Exception:
    cat_available = False


def categorize_stress(x):
    if x <= 3:
        return "Low"
    elif x <= 6:
        return "Moderate"
    else:
        return "High"


def load_data(path):
    df = pd.read_csv(path)
    return df


def prepare_data(df):
    if "Person ID" in df.columns:
        df = df.drop(columns=["Person ID"])
    df["Stress_Category"] = df["Stress Level"].apply(categorize_stress)
    X = df.drop(columns=["Stress Level", "Stress_Category"])
    y = df["Stress_Category"]
    return X, y


def get_column_types():
    num_cols = [
        "Age",
        "Sleep Duration",
        "Quality of Sleep",
        "Physical Activity Level",
        "Heart Rate",
        "Daily Steps"
    ]
    cat_cols = [
        "Gender",
        "Occupation",
        "BMI Category",
        "Blood Pressure",
        "Sleep Disorder"
    ]
    return num_cols, cat_cols


def build_preprocessor(num_cols, cat_cols):
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])
    return preprocessor


def evaluate_sklearn_models(X_train, X_test, y_train, y_test, preprocessor):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=SEED),
        "SVM": SVC(random_state=SEED),
        "Random Forest": RandomForestClassifier(random_state=SEED),
        "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
        "KNN": KNeighborsClassifier()
    }
    results = []
    fitted_models = {}
    for name, model in models.items():
        pipe = Pipeline([
            ("prep", preprocessor),
            ("clf", model)
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        results.append({
            "Model": name,
            "Accuracy": acc,
            "Macro_F1": f1
        })
        fitted_models[name] = pipe
    return results, fitted_models


def prepare_encoded_data_for_boosting(X_train, X_test, num_cols, cat_cols):
    encoder = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])
    X_train_enc = encoder.fit_transform(X_train)
    X_test_enc = encoder.transform(X_test)
    return X_train_enc, X_test_enc


def evaluate_xgboost(X_train_enc, X_test_enc, y_train, y_test):
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_test_enc = label_encoder.transform(y_test)
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softmax",
        num_class=len(label_encoder.classes_),
        random_state=SEED,
        eval_metric="mlogloss"
    )
    model.fit(X_train_enc, y_train_enc)
    y_pred_enc = model.predict(X_test_enc)
    y_pred = label_encoder.inverse_transform(y_pred_enc)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    result = {
        "Model": "XGBoost",
        "Accuracy": acc,
        "Macro_F1": f1
    }
    return result, model, label_encoder


def evaluate_catboost(X_train_enc, X_test_enc, y_train, y_test):
    model = CatBoostClassifier(
        iterations=200,
        depth=4,
        learning_rate=0.05,
        loss_function="MultiClass",
        random_seed=SEED,
        verbose=0
    )
    model.fit(X_train_enc, y_train)
    y_pred = model.predict(X_test_enc)
    y_pred = np.array(y_pred).reshape(-1)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    result = {
        "Model": "CatBoost",
        "Accuracy": acc,
        "Macro_F1": f1
    }
    return result, model


def main():
    df = load_data(DATA_PATH)
    X, y = prepare_data(df)
    num_cols, cat_cols = get_column_types()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=SEED,
        stratify=y
    )
    preprocessor = build_preprocessor(num_cols, cat_cols)
    sklearn_results, sklearn_models = evaluate_sklearn_models(
        X_train, X_test, y_train, y_test, preprocessor
    )
    all_results = sklearn_results.copy()
    xgb_model = None
    xgb_label_encoder = None
    cat_model = None
    X_train_enc, X_test_enc = prepare_encoded_data_for_boosting(
        X_train, X_test, num_cols, cat_cols
    )
    if xgb_available:
        try:
            xgb_result, xgb_model, xgb_label_encoder = evaluate_xgboost(
                X_train_enc, X_test_enc, y_train, y_test
            )
            all_results.append(xgb_result)
        except Exception as e:
            print("\nXGBoost could not be evaluated:")
            print(e)
    else:
        print("\nXGBoost is not available in this environment.")
        
    if cat_available:
        try:
            cat_result, cat_model = evaluate_catboost(
                X_train_enc, X_test_enc, y_train, y_test
            )
            all_results.append(cat_result)
        except Exception as e:
            print("\nCatBoost could not be evaluated:")
            print(e)
    else:
        print("\nCatBoost is not available in this environment.")

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values(by="Macro_F1", ascending=False).reset_index(drop=True)

    print("\n=== MODEL COMPARISON ===")
    print(results_df)

    best_model_name = results_df.loc[0, "Model"]
    print(f"\nBest model: {best_model_name}")

    print("\n=== CLASSIFICATION REPORT OF BEST MODEL ===")

    if best_model_name in sklearn_models:
        best_model = sklearn_models[best_model_name]
        y_pred_best = best_model.predict(X_test)
        print(classification_report(y_test, y_pred_best))

    elif best_model_name == "XGBoost" and xgb_model is not None:
        y_pred_enc = xgb_model.predict(X_test_enc)
        y_pred_best = xgb_label_encoder.inverse_transform(y_pred_enc)
        print(classification_report(y_test, y_pred_best))

    elif best_model_name == "CatBoost" and cat_model is not None:
        y_pred_best = cat_model.predict(X_test_enc)
        y_pred_best = np.array(y_pred_best).reshape(-1)
        print(classification_report(y_test, y_pred_best))


if __name__ == "__main__":
    main()