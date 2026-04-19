"""
Analysis tools — pandas-based dataset summary and statistics.
These are called by the analysis_agent via LangChain tool interface.
"""

import pandas as pd
import logging
from langchain.tools import tool
from typing import Any

logger = logging.getLogger(__name__)

# Global dataset reference — set by app.py via set_dataset()
_dataset: pd.DataFrame = None


def set_dataset(df: pd.DataFrame):
    """Register the active dataset for tools to use."""
    global _dataset
    _dataset = df
    logger.info(f"Analysis tool: dataset registered with shape {df.shape}")


def _require_dataset() -> pd.DataFrame:
    if _dataset is None:
        raise ValueError("No dataset loaded. Please upload a CSV or load the Iris dataset.")
    return _dataset


@tool
def dataset_summary(query: str = "") -> str:
    """
    Return a comprehensive summary of the dataset including shape,
    column names, data types, and missing value counts.
    """
    try:
        df = _require_dataset()
        lines = [
            f"📊 Dataset Summary",
            f"  Rows: {df.shape[0]}",
            f"  Columns: {df.shape[1]}",
            f"  Column names: {', '.join(str(col) for col in df.columns.tolist())}",
            "",
            "Column Details:",
        ]
        for col in df.columns:
            missing = df[col].isnull().sum()
            pct = (missing / len(df) * 100).round(1)
            lines.append(f"  - {col}: {df[col].dtype}  |  missing: {missing} ({pct}%)")

        dupes = df.duplicated().sum()
        lines.append(f"\nDuplicate rows: {dupes}")
        return "\n".join(lines)

    except Exception as e:
        logger.error(f"dataset_summary error: {e}")
        return f"Error generating summary: {e}"


@tool
def describe_statistics(query: str = "") -> str:
    """
    Return descriptive statistics (mean, std, min, max, quartiles)
    for all numeric columns in the dataset.
    """
    try:
        df = _require_dataset()
        numeric = df.select_dtypes(include="number")
        if numeric.empty:
            return "No numeric columns found in the dataset."

        desc = numeric.describe().round(4)
        lines = ["📈 Descriptive Statistics (numeric columns):\n"]
        lines.append(desc.to_string())

        # Categorical summary
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if cat_cols:
            lines.append("\n\n📝 Categorical Columns:")
            for col in cat_cols:
                top = df[col].value_counts().head(5)
                lines.append(f"\n  {col} (unique: {df[col].nunique()}):")
                for val, cnt in top.items():
                    lines.append(f"    {val}: {cnt}")

        return "\n".join(lines)

    except Exception as e:
        logger.error(f"describe_statistics error: {e}")
        return f"Error computing statistics: {e}"


@tool
def correlation_summary(query: str = "") -> str:
    """
    Return the top correlated feature pairs from the dataset.
    """
    try:
        df = _require_dataset()
        numeric = df.select_dtypes(include="number")
        if numeric.shape[1] < 2:
            return "Need at least 2 numeric columns for correlation analysis."

        corr = numeric.corr().abs()
        # Extract upper triangle pairs
        pairs = []
        cols = corr.columns.tolist()
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                pairs.append((cols[i], cols[j], corr.iloc[i, j]))

        pairs.sort(key=lambda x: x[2], reverse=True)
        lines = ["🔗 Top Correlated Feature Pairs:\n"]
        for a, b, v in pairs[:10]:
            lines.append(f"  {a} ↔ {b}: {v:.4f}")
        return "\n".join(lines)

    except Exception as e:
        logger.error(f"correlation_summary error: {e}")
        return f"Error computing correlations: {e}"
