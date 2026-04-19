"""
React Agents — SimpleAgent pattern (works with phi3 and small local LLMs).

Why SimpleAgent instead of LangChain ReAct:
  - ReAct requires strict Thought/Action/Observation format that phi3 cannot follow
  - SimpleAgent asks the LLM ONE question: "pick a tool name from this list"
  - Rule-based validation then cross-checks and overrides when needed
  - FallbackAgentExecutor runs tools directly when Ollama is offline

Decision flow inside SimpleAgent.invoke():
  Step 1 — LLM picks a tool name from the list (one word answer)
  Step 2 — Rule engine checks query keywords for obvious chart/tool matches
  Step 3 — Final decision: if both agree → use LLM, if conflict → prefer rule,
            if only one answered → use that, if neither → use first tool
"""

import logging
import time
from typing import Any, Dict, List

from langchain.tools import BaseTool
from utils.llm_wrapper import get_llm

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Fallback executor — zero LLM calls, pure keyword matching
# ─────────────────────────────────────────────────────────────────────────────

class FallbackAgentExecutor:
    """
    Runs tools directly by matching query keywords to tool names.
    Used when Ollama is offline. No LLM involved at all.
    """

    KEYWORD_TOOL_MAP = {
        "summary":      ["dataset_summary"],
        "statistic":    ["describe_statistics"],
        "describe":     ["describe_statistics"],
        "correlat":     ["correlation_heatmap"],
        "heatmap":      ["correlation_heatmap"],
        "histogram":    ["histogram_plot"],
        "hist":         ["histogram_plot"],
        "scatter":      ["scatter_plot"],
        "violin":       ["violin_plot"],
        "distribut":    ["feature_distributions"],
        "boxplot":      ["boxplot_visualization"],
        "box plot":     ["boxplot_visualization"],
        "pairplot":     ["pairplot_visualization"],
        "pair plot":    ["pairplot_visualization"],
        "missing":      ["cleaning_report", "fill_missing_mean"],
        "null":         ["fill_missing_mean"],
        "fill":         ["fill_missing_mean"],
        "median":       ["fill_missing_median"],
        "mode":         ["fill_missing_mode"],
        "duplicate":    ["remove_duplicates"],
        "clean":        ["cleaning_report"],
        "quality":      ["cleaning_report"],
        "train":        ["auto_train_models"],
        "model":        ["auto_train_models"],
        "automl":       ["auto_train_models"],
        "classif":      ["auto_train_models"],
        "regress":      ["auto_train_models"],
        "task":         ["detect_task_type"],
    }

    def __init__(self, tools: List[BaseTool], agent_name: str):
        self.tools = tools
        self.agent_name = agent_name
        self._tool_map = {t.name: t for t in tools}

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        query = inputs.get("input", "")
        q = query.lower()
        logger.warning(f"[{self.agent_name}] Fallback mode — keyword matching")

        tools_to_run = []
        for keyword, tool_names in self.KEYWORD_TOOL_MAP.items():
            if keyword in q:
                for tn in tool_names:
                    if tn in self._tool_map and tn not in tools_to_run:
                        tools_to_run.append(tn)

        if not tools_to_run:
            tools_to_run = [self.tools[0].name]

        results = []
        for tname in tools_to_run:
            tool = self._tool_map.get(tname)
            if not tool:
                continue
            try:
                results.append(tool.invoke(query))
                logger.info(f"Fallback ran: {tname}")
            except Exception as e:
                results.append(f"{tname} error: {e}")
                logger.error(f"Fallback tool {tname} failed: {e}")

        return {"output": "\n\n".join(results)}


# ─────────────────────────────────────────────────────────────────────────────
# SimpleAgent — LLM picks tool + rule validation
# ─────────────────────────────────────────────────────────────────────────────

class SimpleAgent:
    """
    Asks the LLM to pick one tool name, then cross-checks with keyword rules.
    Works with phi3 because the prompt asks for a single word, not a formatted loop.
    """

    # keyword → expected tool name (for override / tie-breaking)
    RULE_TOOL_MAP = {
        "histogram":    "histogram_plot",
        "hist":         "histogram_plot",
        "scatter":      "scatter_plot",
        "violin":       "violin_plot",
        "boxplot":      "boxplot_visualization",
        "box plot":     "boxplot_visualization",
        "heatmap":      "correlation_heatmap",
        "correlat":     "correlation_heatmap",
        "pairplot":     "pairplot_visualization",
        "pair plot":    "pairplot_visualization",
        "distribut":    "feature_distributions",
        "missing":      "fill_missing_mean",
        "null":         "fill_missing_mean",
        "fill mean":    "fill_missing_mean",
        "fill median":  "fill_missing_median",
        "fill mode":    "fill_missing_mode",
        "duplicate":    "remove_duplicates",
        "summary":      "dataset_summary",
        "statistic":    "describe_statistics",
        "describe":     "describe_statistics",
        "train":        "auto_train_models",
        "model":        "auto_train_models",
        "classif":      "auto_train_models",
        "regress":      "auto_train_models",
        "task":         "detect_task_type",
    }

    def __init__(self, tools: List[BaseTool], llm, agent_name: str):
        self.tools = {t.name: t for t in tools}
        self.llm = llm
        self.agent_name = agent_name

    def _get_rule_tool(self, query: str) -> str | None:
        """Return tool name from keyword rules, or None if no match."""
        q = query.lower()
        # Explain / interpret → no rule override, let LLM or fallback decide
        if any(w in q for w in ["explain", "interpret", "meaning", "what does"]):
            return None
        for keyword, tool_name in self.RULE_TOOL_MAP.items():
            if keyword in q:
                if tool_name in self.tools:
                    return tool_name
        return None

    def _get_llm_tool(self, query: str) -> str | None:
        """Ask LLM to pick one tool name. Clean up messy phi3 output."""
        tool_names = list(self.tools.keys())
        try:
            raw = self.llm.invoke(f"""Available tools: {tool_names}

User query: {query}

Return ONLY the exact tool name from the list above. Nothing else.""")

            if not raw or raw.startswith("[ERROR]"):
                return None

            raw = raw.strip().lower().strip('"').strip("'")

            # Strip common phi3 preamble
            for prefix in ["i would choose:", "tool:", "answer:", "use:", "the tool is:"]:
                if raw.startswith(prefix):
                    raw = raw[len(prefix):].strip()

            # Take first line / first word only
            raw = raw.split("\n")[0].split(".")[0].strip()

            # Exact match first
            if raw in self.tools:
                return raw

            # Partial match
            for name in self.tools:
                if name in raw or raw in name:
                    return name

            return None

        except Exception as e:
            logger.warning(f"LLM tool selection failed: {e}")
            return None

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        query = inputs.get("input", "")
        logger.info(f"[{self.agent_name}] SimpleAgent.invoke | query='{query[:60]}'")

        llm_tool  = self._get_llm_tool(query)
        rule_tool = self._get_rule_tool(query)

        logger.info(f"[{self.agent_name}] LLM chose: {llm_tool} | Rule chose: {rule_tool}")

        # Decision logic
        if llm_tool and rule_tool:
            # Both answered — if they agree, great; if conflict, prefer rule
            # (rules are more reliable for explicit chart keywords)
            final_tool = llm_tool if llm_tool == rule_tool else rule_tool
        elif rule_tool:
            final_tool = rule_tool
        elif llm_tool:
            final_tool = llm_tool
        else:
            # Neither answered — use first tool in list
            final_tool = list(self.tools.keys())[0]

        logger.info(f"[{self.agent_name}] Final tool: {final_tool}")

        tool = self.tools.get(final_tool)
        if tool:
            try:
                output = tool.invoke(query)
                return {"output": output}
            except Exception as e:
                logger.error(f"Tool {final_tool} failed: {e}")
                return {"output": f"Tool error ({final_tool}): {e}"}

        return {"output": f"Tool not found: {final_tool}"}


# ─────────────────────────────────────────────────────────────────────────────
# Builder — returns SimpleAgent (Ollama up) or FallbackAgentExecutor (offline)
# ─────────────────────────────────────────────────────────────────────────────

def _build_agent_executor(tools: List[BaseTool], agent_name: str):
    """
    Check Ollama health.
    - Online  → SimpleAgent (LLM picks tool + rule validation)
    - Offline → FallbackAgentExecutor (pure keyword matching, no LLM)
    """
    from utils.ollama import check_ollama_health

    if not check_ollama_health():
        logger.warning(f"[{agent_name}] Ollama offline — using FallbackAgentExecutor")
        return FallbackAgentExecutor(tools, agent_name)

    try:
        llm = get_llm()
        logger.info(f"[{agent_name}] Ollama online — using SimpleAgent")
        return SimpleAgent(tools, llm, agent_name)
    except Exception as e:
        logger.error(f"[{agent_name}] Failed to create SimpleAgent ({e}) — using fallback")
        return FallbackAgentExecutor(tools, agent_name)


# ─────────────────────────────────────────────────────────────────────────────
# Specialized agent factories
# ─────────────────────────────────────────────────────────────────────────────

def build_analysis_agent():
    """Stats, summaries, correlation text."""
    from tools.analysis_tool import dataset_summary, describe_statistics, correlation_summary
    return _build_agent_executor(
        [dataset_summary, describe_statistics, correlation_summary],
        "analysis_agent",
    )


def build_visualization_agent():
    """All chart types — heatmap, histogram, scatter, violin, boxplot, pairplot, distributions."""
    from tools.visualization_tool import (
        correlation_heatmap,
        feature_distributions,
        pairplot_visualization,
        boxplot_visualization,
        scatter_plot,
        histogram_plot,
        violin_plot,
    )
    return _build_agent_executor(
        [
            correlation_heatmap,
            feature_distributions,
            pairplot_visualization,
            boxplot_visualization,
            scatter_plot,
            histogram_plot,
            violin_plot,
        ],
        "visualization_agent",
    )


def build_cleaning_agent():
    """Missing values, duplicates, data quality."""
    from tools.cleaning_tool import (
        cleaning_report,
        remove_duplicates,
        fill_missing_mean,
        fill_missing_median,
        fill_missing_mode,
        drop_missing_rows,
    )
    return _build_agent_executor(
        [
            cleaning_report,
            remove_duplicates,
            fill_missing_mean,
            fill_missing_median,
            fill_missing_mode,
            drop_missing_rows,
        ],
        "cleaning_agent",
    )


def build_ml_agent():
    """Task detection, model training, metrics."""
    from tools.ml_tool import detect_task_type, auto_train_models
    return _build_agent_executor(
        [detect_task_type, auto_train_models],
        "ml_agent",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

AGENT_REGISTRY: Dict[str, Any] = {
    "analysis_agent":      build_analysis_agent,
    "visualization_agent": build_visualization_agent,
    "cleaning_agent":      build_cleaning_agent,
    "ml_agent":            build_ml_agent,
}


def get_agent(agent_name: str):
    if agent_name not in AGENT_REGISTRY:
        raise ValueError(f"Unknown agent '{agent_name}'. Available: {list(AGENT_REGISTRY.keys())}")
    logger.info(f"Instantiating: {agent_name}")
    return AGENT_REGISTRY[agent_name]()
