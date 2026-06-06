"""
Cleaning tools — handle missing values, duplicates, and basic data quality issues.
Results update the global dataset in session.
"""

import pandas as pd
import logging
from langchain.tools import tool
from typing import Callable

logger = logging.getLogger(__name__)

# Global dataset reference and update callback
_dataset: pd.DataFrame = None
_update_callback: Callable = None


def set_dataset(df: pd.DataFrame, update_callback: Callable = None):
    """
    Register the active dataset and an optional callback to propagate updates.

    Args:
        df: The DataFrame to use.
        update_callback: Called with cleaned DataFrame when data is modified.
    """
    global _dataset, _update_callback
    _dataset = df
    _update_callback = update_callback
    logger.info(f"Cleaning tool: dataset registered with shape {df.shape}")


def get_dataset() -> pd.DataFrame:
    """Return the current (possibly cleaned) dataset."""
    return _dataset


def _require_dataset() -> pd.DataFrame:
    if _dataset is None:
        raise ValueError("No dataset loaded.")
    return _dataset


def _save(df: pd.DataFrame):
    """Persist changes back through callback."""
    global _dataset
    _dataset = df
    if _update_callback:
        _update_callback(df)


@tool
def remove_duplicates(query: str = "") -> str:
    """
    Remove duplicate rows from the dataset and report how many were removed.
    """
    try:
        df = _require_dataset()
        before = len(df)
        cleaned = df.drop_duplicates()
        removed = before - len(cleaned)
        _save(cleaned)
        logger.info(f"Removed {removed} duplicate rows.")
        return f"✅ Removed {removed} duplicate rows. Dataset now has {len(cleaned)} rows."

    except Exception as e:
        logger.error(f"remove_duplicates error: {e}")
        return f"Error removing duplicates: {e}"


@tool
def fill_missing_mean(query: str = "") -> str:
    """
    Fill missing values in numeric columns with their column mean.
    """
    try:
        df = _require_dataset()
        numeric_cols = df.select_dtypes(include="number").columns
        total_missing_before = df[numeric_cols].isnull().sum().sum()

        df_cleaned = df.copy()
        for col in numeric_cols:
            if df_cleaned[col].isnull().any():
                mean_val = df_cleaned[col].mean()
                df_cleaned[col] = df_cleaned[col].fillna(mean_val)

        _save(df_cleaned)
        total_missing_after = df_cleaned[numeric_cols].isnull().sum().sum()
        filled = total_missing_before - total_missing_after
        logger.info(f"Filled {filled} missing values with column means.")
        return f"✅ Filled {filled} missing values (numeric columns) with column means."

    except Exception as e:
        logger.error(f"fill_missing_mean error: {e}")
        return f"Error filling missing values: {e}"


@tool
def fill_missing_median(query: str = "") -> str:
    """
    Fill missing values in numeric columns with their column median.
    """
    try:
        df = _require_dataset()
        numeric_cols = df.select_dtypes(include="number").columns
        total_missing_before = df[numeric_cols].isnull().sum().sum()

        df_cleaned = df.copy()
        for col in numeric_cols:
            if df_cleaned[col].isnull().any():
                df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())

        _save(df_cleaned)
        filled = total_missing_before - df_cleaned[numeric_cols].isnull().sum().sum()
        logger.info(f"Filled {filled} missing values with column medians.")
        return f"✅ Filled {filled} missing values (numeric columns) with column medians."

    except Exception as e:
        logger.error(f"fill_missing_median error: {e}")
        return f"Error filling missing values: {e}"


@tool
def drop_missing_rows(query: str = "") -> str:
    """
    Drop any rows that contain at least one missing value.
    """
    try:
        df = _require_dataset()
        before = len(df)
        cleaned = df.dropna()
        dropped = before - len(cleaned)
        _save(cleaned)
        logger.info(f"Dropped {dropped} rows with missing values.")
        return f"✅ Dropped {dropped} rows with missing values. Dataset now has {len(cleaned)} rows."

    except Exception as e:
        logger.error(f"drop_missing_rows error: {e}")
        return f"Error dropping rows: {e}"


@tool
def fill_missing_mode(query: str = "") -> str:
    """
    Fill missing values in categorical/object columns with their column mode (most frequent value).
    """
    try:
        df = _require_dataset()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        df_cleaned = df.copy()
        filled_cols = []
        for col in cat_cols:
            if df_cleaned[col].isnull().any():
                mode_val = df_cleaned[col].mode()
                if not mode_val.empty:
                    df_cleaned[col] = df_cleaned[col].fillna(mode_val[0])
                    filled_cols.append(col)
        _save(df_cleaned)
        logger.info(f"Filled categorical columns with mode: {filled_cols}")
        return f"✅ Filled missing values in categorical columns [{', '.join(filled_cols)}] with mode."

    except Exception as e:
        logger.error(f"fill_missing_mode error: {e}")
        return f"Error filling categorical missing values: {e}"
    
@tool
def remove_outliers(query: str = "") -> str:
    """
    Remove outliers from numeric columns using Z-score method (threshold = 3).
    """
    try:
        df = _require_dataset()

        numeric_cols = df.select_dtypes(include="number").columns
        df_cleaned = df.copy()

        before_rows = len(df_cleaned)

        for col in numeric_cols:
            mean = df_cleaned[col].mean()
            std = df_cleaned[col].std()

            if std == 0:
                continue

            z_scores = (df_cleaned[col] - mean) / std
            df_cleaned = df_cleaned[(z_scores.abs() < 3)]

        after_rows = len(df_cleaned)
        removed = before_rows - after_rows

        _save(df_cleaned)

        logger.info(f"Removed {removed} outliers.")

        return f"Removed {removed} outlier rows. Dataset now has {after_rows} rows."

    except Exception as e:
        logger.error(f"remove_outliers error: {e}")
        return f"Error removing outliers: {e}"


@tool
def cleaning_report(query: str = "") -> str:
    """
    Generate a data quality report showing missing values, duplicates, and data types.
    """
    try:
        df = _require_dataset()
        lines = [
            "🧹 Data Quality Report",
            f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns",
            f"  Duplicate rows: {df.duplicated().sum()}",
            f"  Total missing cells: {df.isnull().sum().sum()}",
            "",
            "Per-column missing values:",
        ]
        for col in df.columns:
            missing = df[col].isnull().sum()
            pct = (missing / len(df) * 100).round(1)
            status = "⚠️" if missing > 0 else "✅"
            lines.append(f"  {status} {col}: {missing} missing ({pct}%)")
        return "\n".join(lines)

    except Exception as e:
        logger.error(f"cleaning_report error: {e}")
        return f"Error generating cleaning report: {e}"
