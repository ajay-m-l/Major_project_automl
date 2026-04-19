"""
Schema utilities — extracts lightweight metadata from a DataFrame.
This is what gets sent to the LLM (never the raw data).
"""

import pandas as pd
from typing import Dict, Any


def extract_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Extract a minimal schema from a DataFrame — safe to send to an LLM.

    Returns:
        dict with keys: rows, columns, column_names, dtypes, missing_values, sample_values
    """
    column_names = [str(col) for col in df.columns.tolist()]
    schema = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": column_names,
        "dtypes": {str(col): str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": {str(col): int(missing) for col, missing in df.isnull().sum().items()},
        "numeric_columns": [str(col) for col in df.select_dtypes(include="number").columns.tolist()],
        "categorical_columns": [str(col) for col in df.select_dtypes(include=["object", "category"]).columns.tolist()],
    }
    return schema


def schema_to_text(schema: Dict[str, Any]) -> str:
    """
    Convert schema dict to a compact text summary for LLM prompts.
    """
    lines = [
        f"Dataset: {schema['rows']} rows × {schema['columns']} columns",
        f"Columns: {', '.join(str(col) for col in schema['column_names'])}",
        f"Numeric columns: {', '.join(str(col) for col in schema['numeric_columns']) or 'None'}",
        f"Categorical columns: {', '.join(str(col) for col in schema['categorical_columns']) or 'None'}",
        "Data types:",
    ]
    for col, dtype in schema["dtypes"].items():
        missing = schema["missing_values"].get(col, 0)
        lines.append(f"  - {col}: {dtype} (missing: {missing})")

    return "\n".join(lines)


def get_dataset_intelligence(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Full intelligence panel data — displayed in Streamlit sidebar.
    """
    numeric_df = df.select_dtypes(include="number")
    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.to_dict(),
        "missing": df.isnull().sum().to_dict(),
        "missing_pct": (df.isnull().mean() * 100).round(2).to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "describe": df.describe(include="all").to_dict() if not df.empty else {},
        "numeric_columns": numeric_df.columns.tolist(),
        "categorical_columns": df.select_dtypes(include=["object", "category"]).columns.tolist(),
    }
