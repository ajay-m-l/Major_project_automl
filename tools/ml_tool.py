"""
AutoML tools — train models, evaluate, and give business insights.
Stores last ML result in memory for follow-up business questions.
"""

import pandas as pd
import numpy as np
import logging
from langchain.tools import tool

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, mean_absolute_error

logger = logging.getLogger(__name__)

_dataset: pd.DataFrame = None
_memory = None   # set by executor so ML can push results back to memory


def set_dataset(df: pd.DataFrame):
    global _dataset
    _dataset = df
    logger.info(f"ML tool: dataset registered {df.shape}")


def set_memory(mem):
    global _memory
    _memory = mem


def _require_dataset():
    if _dataset is None:
        raise ValueError("No dataset loaded.")
    return _dataset


def _detect_task_type(df, target_col):
    if df[target_col].dtype in ["object", "category"]:
        return "classification"
    return "classification" if df[target_col].nunique() <= 20 else "regression"


def _prepare_data(df, target_col):
    df = df.dropna(subset=[target_col]).copy()
    y = df[target_col].copy()
    X = df.drop(columns=[target_col])

    for col in X.select_dtypes(include=["object", "category"]).columns:
        try:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        except Exception:
            X.drop(columns=[col], inplace=True)

    le = None
    if y.dtype in ["object", "category"]:
        le = LabelEncoder()
        y = le.fit_transform(y.astype(str))

    X = X.fillna(X.mean(numeric_only=True))
    X = X.select_dtypes(include="number")
    return X, y, le


@tool
def auto_train_models(query: str = "") -> str:
    """
    Auto-detect ML task, train models (Logistic Regression, Random Forest,
    Linear Regression), return accuracy/RMSE/R² metrics and feature importance.
    Optionally specify target column: 'target: column_name'
    """
    try:
        df = _require_dataset()

        # Parse target column
        target_col = None
        if "target:" in query.lower():
            raw = query.lower().split("target:")[1].strip().split()[0].strip().rstrip(".,;")
            for col in df.columns:
                if col.lower() == raw:
                    target_col = col
                    break
        if not target_col or target_col not in df.columns:
            target_col = df.columns[-1]

        task_type = _detect_task_type(df, target_col)
        X, y, le = _prepare_data(df, target_col)

        if X.shape[1] == 0:
            return "No usable feature columns found."

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        results = [
            f"AutoML Results",
            f"  Task Type      : {task_type.upper()}",
            f"  Target Column  : {target_col}",
            f"  Features used  : {X.shape[1]} columns — {', '.join(str(col) for col in X.columns.tolist())}",
            f"  Training rows  : {len(X_train)}",
            f"  Test rows      : {len(X_test)}",
            "",
        ]

        best_metric = -float("inf")
        best_model_name = ""
        top_features = []

        if task_type == "classification":
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            }
            results.append("Classification Metrics:")
            for name, model in models.items():
                try:
                    model.fit(X_train_s, y_train)
                    preds = model.predict(X_test_s)
                    acc = accuracy_score(y_test, preds)
                    cv = cross_val_score(model, X_train_s, y_train, cv=3, scoring="accuracy")
                    results.append(f"\n  {name}")
                    results.append(f"    Accuracy    : {acc:.4f}  ({acc:.1%})")
                    results.append(f"    CV Accuracy : {cv.mean():.4f} ± {cv.std():.4f}")
                    if acc > best_metric:
                        best_metric = acc
                        best_model_name = name
                    if hasattr(model, "feature_importances_"):
                        imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
                        top_features = [str(col) for col in imp.head(5).index.tolist()]
                        results.append(f"    Top features: {', '.join(top_features[:3])}")
                        for feat, val in imp.head(5).items():
                            results.append(f"      {feat}: {val:.4f}")
                except Exception as e:
                    results.append(f"  {name} failed: {e}")

            results.append(f"\nBest Model: {best_model_name}  (Accuracy: {best_metric:.4f} = {best_metric:.1%})")

            # Interpretation
            if best_metric >= 0.95:
                results.append("Interpretation: Excellent accuracy — model is reliable.")
            elif best_metric >= 0.80:
                results.append("Interpretation: Good accuracy — suitable for most use cases.")
            elif best_metric >= 0.65:
                results.append("Interpretation: Fair accuracy — needs improvement.")
            else:
                results.append("Interpretation: Poor accuracy — review data quality.")

            # Store in memory
            if _memory:
                _memory.store_ml_result({
                    "task": "classification",
                    "best_model": best_model_name,
                    "best_accuracy": best_metric,
                    "top_features": top_features,
                    "target": target_col,
                })

        else:  # regression
            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
            }
            results.append("Regression Metrics:")
            for name, model in models.items():
                try:
                    model.fit(X_train_s, y_train)
                    preds = model.predict(X_test_s)
                    rmse = np.sqrt(mean_squared_error(y_test, preds))
                    mae = mean_absolute_error(y_test, preds)
                    r2 = r2_score(y_test, preds)
                    results.append(f"\n  {name}")
                    results.append(f"    RMSE  : {rmse:.4f}")
                    results.append(f"    MAE   : {mae:.4f}")
                    results.append(f"    R²    : {r2:.4f}  (explains {max(0,r2):.1%} of variance)")
                    if r2 > best_metric:
                        best_metric = r2
                        best_model_name = name
                    if hasattr(model, "feature_importances_"):
                        imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
                        top_features = [str(col) for col in imp.head(5).index.tolist()]
                        results.append(f"    Top features: {', '.join(top_features[:3])}")
                        for feat, val in imp.head(5).items():
                            results.append(f"      {feat}: {val:.4f}")
                except Exception as e:
                    results.append(f"  {name} failed: {e}")

            results.append(f"\nBest Model: {best_model_name}  (R²: {best_metric:.4f})")

            if best_metric >= 0.85:
                results.append("Interpretation: Strong predictor — suitable for business forecasting.")
            elif best_metric >= 0.60:
                results.append("Interpretation: Moderate predictor — useful directionally.")
            else:
                results.append("Interpretation: Weak predictor — consider adding more relevant features.")

            if _memory:
                _memory.store_ml_result({
                    "task": "regression",
                    "best_model": best_model_name,
                    "best_r2": best_metric,
                    "top_features": top_features,
                    "target": target_col,
                })

        return "\n".join(results)

    except Exception as e:
        logger.error(f"auto_train_models error: {e}", exc_info=True)
        return f"Error during AutoML training: {e}"


@tool
def detect_task_type(query: str = "") -> str:
    """Detect whether the dataset requires classification or regression."""
    try:
        df = _require_dataset()
        target_col = df.columns[-1]
        task = _detect_task_type(df, target_col)
        unique = df[target_col].nunique()
        dtype = str(df[target_col].dtype)
        reason = (
            "String/category column → classification"
            if df[target_col].dtype in ["object", "category"]
            else f"{unique} unique numeric values → {task}"
        )
        return (
            f"Task Detection\n"
            f"  Target column  : '{target_col}'\n"
            f"  Data type      : {dtype}\n"
            f"  Unique values  : {unique}\n"
            f"  Detected task  : {task.upper()}\n"
            f"  Reasoning      : {reason}\n"
            f"\nThis means your dataset is set up for {task}.\n"
            f"{'You are trying to predict a category/class label.' if task=='classification' else 'You are trying to predict a continuous numeric value.'}"
        )
    except Exception as e:
        return f"Error: {e}"
