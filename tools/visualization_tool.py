"""
Visualization tools — generate matplotlib/seaborn figures.
Agents call these tools; figures are returned and rendered in Streamlit.
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server use
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from langchain.tools import tool

logger = logging.getLogger(__name__)

# Global dataset reference
_dataset: pd.DataFrame = None
# Storage for the last generated figure (for Streamlit to pick up)
_last_figure = None


def set_dataset(df: pd.DataFrame):
    """Register the active dataset."""
    global _dataset
    _dataset = df
    logger.info(f"Visualization tool: dataset registered with shape {df.shape}")


def get_last_figure():
    """Retrieve the most recently generated figure."""
    return _last_figure


def _require_dataset() -> pd.DataFrame:
    if _dataset is None:
        raise ValueError("No dataset loaded.")
    return _dataset


@tool
def correlation_heatmap(query: str = "") -> str:
    """
    Generate a correlation heatmap for numeric columns in the dataset.
    The heatmap is stored and rendered in Streamlit automatically.
    """
    global _last_figure
    try:
        df = _require_dataset()
        numeric = df.select_dtypes(include="number")
        if numeric.shape[1] < 2:
            return "Need at least 2 numeric columns to generate a heatmap."

        fig, ax = plt.subplots(figsize=(10, 7))
        corr_matrix = numeric.corr()
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            square=True,
            ax=ax,
            linewidths=0.5,
        )
        ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
        plt.tight_layout()
        _last_figure = fig
        logger.info("Correlation heatmap generated.")
        return "✅ Correlation heatmap generated and ready to display."

    except Exception as e:
        logger.error(f"correlation_heatmap error: {e}")
        return f"Error generating heatmap: {e}"


@tool
def feature_distributions(query: str = "") -> str:
    """
    Generate distribution plots (histograms with KDE) for all numeric columns.
    """
    global _last_figure
    try:
        df = _require_dataset()
        numeric = df.select_dtypes(include="number")
        if numeric.empty:
            return "No numeric columns found."

        n_cols = min(numeric.shape[1], 4)
        n_rows = (numeric.shape[1] + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

        for i, col in enumerate(numeric.columns):
            sns.histplot(numeric[col].dropna(), kde=True, ax=axes[i], color="steelblue")
            axes[i].set_title(f"Distribution: {col}", fontsize=11)
            axes[i].set_xlabel(col)
            axes[i].set_ylabel("Count")

        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle("Feature Distributions", fontsize=14, fontweight="bold", y=1.01)
        plt.tight_layout()
        _last_figure = fig
        logger.info("Feature distributions generated.")
        return "✅ Feature distribution plots generated and ready to display."

    except Exception as e:
        logger.error(f"feature_distributions error: {e}")
        return f"Error generating distributions: {e}"


@tool
def pairplot_visualization(query: str = "") -> str:
    """
    Generate a seaborn pairplot for numeric columns (max 5 for performance).
    """
    global _last_figure
    try:
        df = _require_dataset()
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if len(numeric_cols) < 2:
            return "Need at least 2 numeric columns for a pairplot."

        # Limit to 5 columns for performance
        cols_to_plot = numeric_cols[:5]
        plot_df = df[cols_to_plot].dropna()

        # Check if there's a categorical column for hue
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        hue_col = cat_cols[0] if cat_cols else None
        if hue_col:
            plot_df = pd.concat([plot_df, df[hue_col]], axis=1).dropna()

        pair_fig = sns.pairplot(plot_df, hue=hue_col, diag_kind="kde", plot_kws={"alpha": 0.5})
        pair_fig.fig.suptitle("Pairplot of Numeric Features", y=1.02, fontsize=13)
        _last_figure = pair_fig.fig
        logger.info("Pairplot generated.")
        return "✅ Pairplot generated and ready to display."

    except Exception as e:
        logger.error(f"pairplot error: {e}")
        return f"Error generating pairplot: {e}"


@tool
def boxplot_visualization(query: str = "") -> str:
    """
    Generate boxplots for all numeric columns to detect outliers.
    """
    global _last_figure
    try:
        df = _require_dataset()
        numeric = df.select_dtypes(include="number")
        if numeric.empty:
            return "No numeric columns found."

        fig, ax = plt.subplots(figsize=(max(8, numeric.shape[1] * 1.5), 6))
        numeric.boxplot(ax=ax, vert=True, patch_artist=True)
        ax.set_title("Boxplots — Outlier Detection", fontsize=13, fontweight="bold")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        plt.tight_layout()
        _last_figure = fig
        logger.info("Boxplot generated.")
        return "✅ Boxplot visualization generated and ready to display."

    except Exception as e:
        logger.error(f"boxplot error: {e}")
        return f"Error generating boxplot: {e}"


@tool
def scatter_plot(query: str = "") -> str:
    """
    Generate a scatter plot for the first two numeric columns in the dataset.
    If a categorical column exists it is used as the hue (colour by class).
    """
    global _last_figure
    try:
        df = _require_dataset()
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if len(numeric_cols) < 2:
            return "Need at least 2 numeric columns for a scatter plot."

        x_col, y_col = numeric_cols[0], numeric_cols[1]
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        hue_col = cat_cols[0] if cat_cols else None

        fig, ax = plt.subplots(figsize=(8, 6))
        if hue_col:
            for label, grp in df.groupby(hue_col):
                ax.scatter(grp[x_col], grp[y_col], label=str(label), alpha=0.7)
            ax.legend(title=hue_col)
        else:
            ax.scatter(df[x_col], df[y_col], alpha=0.7, color="steelblue")

        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"Scatter Plot — {x_col} vs {y_col}", fontsize=13, fontweight="bold")
        plt.tight_layout()
        _last_figure = fig
        logger.info("Scatter plot generated.")
        return f"✅ Scatter plot generated: {x_col} vs {y_col}."

    except Exception as e:
        logger.error(f"scatter_plot error: {e}")
        return f"Error generating scatter plot: {e}"


@tool
def histogram_plot(query: str = "") -> str:
    """
    Generate histograms for every numeric column (no KDE overlay).
    Useful for quickly seeing the raw frequency distribution of each feature.
    """
    global _last_figure
    try:
        df = _require_dataset()
        numeric = df.select_dtypes(include="number")
        if numeric.empty:
            return "No numeric columns found."

        n_cols = min(numeric.shape[1], 4)
        n_rows = (numeric.shape[1] + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

        for i, col in enumerate(numeric.columns):
            axes[i].hist(numeric[col].dropna(), bins=20, color="cornflowerblue", edgecolor="white")
            axes[i].set_title(f"{col}", fontsize=11)
            axes[i].set_xlabel(col)
            axes[i].set_ylabel("Frequency")

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle("Histograms", fontsize=14, fontweight="bold", y=1.01)
        plt.tight_layout()
        _last_figure = fig
        logger.info("Histogram plot generated.")
        return "✅ Histogram plots generated and ready to display."

    except Exception as e:
        logger.error(f"histogram_plot error: {e}")
        return f"Error generating histograms: {e}"


@tool
def violin_plot(query: str = "") -> str:
    """
    Generate violin plots for numeric columns, split by the first categorical
    column if one exists. Violin plots show both distribution shape and summary stats.
    """
    global _last_figure
    try:
        df = _require_dataset()
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if not numeric_cols:
            return "No numeric columns found."

        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        hue_col = cat_cols[0] if cat_cols else None

        cols_to_plot = numeric_cols[:4]   # limit for readability
        n = len(cols_to_plot)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 6))
        axes = [axes] if n == 1 else list(axes)

        for ax, col in zip(axes, cols_to_plot):
            if hue_col:
                groups = [grp[col].dropna().values for _, grp in df.groupby(hue_col)]
                labels = [str(l) for l in df[hue_col].unique()]
                parts = ax.violinplot(groups, showmedians=True)
                ax.set_xticks(range(1, len(labels) + 1))
                ax.set_xticklabels(labels, rotation=30, ha="right")
            else:
                ax.violinplot([df[col].dropna().values], showmedians=True)
                ax.set_xticks([1])
                ax.set_xticklabels([col])
            ax.set_title(col, fontsize=11)
            ax.set_ylabel("Value")

        plt.suptitle("Violin Plots", fontsize=14, fontweight="bold", y=1.01)
        plt.tight_layout()
        _last_figure = fig
        logger.info("Violin plot generated.")
        return "✅ Violin plots generated and ready to display."

    except Exception as e:
        logger.error(f"violin_plot error: {e}")
        return f"Error generating violin plots: {e}"
