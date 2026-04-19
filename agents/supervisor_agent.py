"""
Supervisor Agent
================
Uses Ollama LLM (or keyword fallback) to:
  1. Read the user query + dataset schema + full session memory
  2. Decide which specialized agent to call
  3. Detect EXPLAIN / INSIGHT / FOLLOW-UP intents
  4. Return final formatted response
"""

import logging
from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END

from utils.ollama import generate as ollama_generate, check_ollama_health
from utils.schema import schema_to_text
from agents.react_agents import get_agent, AGENT_REGISTRY

logger = logging.getLogger(__name__)


# ── State ──────────────────────────────────────────────────────────────────────
class SupervisorState(TypedDict):
    user_query: str
    schema_text: str
    memory_context: str        # full session history string
    last_output: str           # previous tool output (for explain queries)
    last_ml_result: dict       # for business insight queries
    selected_agent: Optional[str]
    agent_input: str
    agent_output: str
    final_response: str
    intent: str                # "run_tool" | "explain" | "business_insight"
    error: Optional[str]
    used_fallback_routing: bool


# ── Intent detection (no LLM needed) ──────────────────────────────────────────
EXPLAIN_KEYWORDS = [
    "explain", "what does this mean", "interpret", "tell me more",
    "elaborate", "describe this", "what is this", "clarify",
    "what does that mean", "meaning of", "what do you mean",
    "break it down", "in simple terms", "why is", "how does this work",
]

INSIGHT_KEYWORDS = [
    "business insight", "business decision", "what should i do",
    "recommend", "recommendation", "advice", "suggest", "action",
    "improve", "next step", "decision", "deploy", "production ready",
    "should i use", "which model should", "is this good", "good enough",
    "what does accuracy mean", "interpret result", "business impact",
]


def detect_intent(query: str) -> str:
    q = query.lower()
    if any(k in q for k in EXPLAIN_KEYWORDS):
        return "explain"
    if any(k in q for k in INSIGHT_KEYWORDS):
        return "business_insight"
    return "run_tool"


# ── Keyword routing (fallback when Ollama offline) ─────────────────────────────
KEYWORD_ROUTING = {
    "analysis_agent": [
        "summary", "summarize", "describe", "statistics", "statistic",
        "overview", "explore", "column", "shape", "rows", "dtype",
        "tell me about", "what is in", "top correlat",
    ],
    "visualization_agent": [
        "heatmap", "heat map", "plot", "chart", "graph", "visual",
        "distribution", "histogram", "hist", "boxplot", "box plot",
        "pairplot", "pair plot", "scatter", "violin", "draw",
        "figure", "show", "correlat",
    ],
    "cleaning_agent": [
        "clean", "cleaning", "duplicate", "missing value", "null value",
        "nan", "fill missing", "fill null", "drop missing", "drop null",
        "remove duplicate", "impute", "quality", "fix data",
        "handle missing", "data quality", "missing", "replace null",
        "replace missing", "fill with mean", "fill with median",
    ],
    "ml_agent": [
        "train", "model", "predict", "machine learning", " ml ",
        "automl", "classif", "regress", "accuracy", "metric",
        "feature importance", "random forest", "logistic", "linear model",
        "performance", "evaluate", "task type", "suited for",
    ],
}


def _keyword_route(query: str, schema_text: str = ""):
    q = query.lower()

    # 🔥 If query mentions column-like words → force analysis
    if any(word in schema_text.lower() for word in q.split()):
        return "analysis_agent", query

    scores = {agent: 0 for agent in KEYWORD_ROUTING}

    for agent, keywords in KEYWORD_ROUTING.items():
        for kw in keywords:
            if kw in q:
                scores[agent] += 1

    best = max(scores, key=scores.get)

    if scores[best] == 0:
        best = "analysis_agent"

    return best, query


# ── LLM supervisor prompt ──────────────────────────────────────────────────────
SUPERVISOR_PROMPT = """You are a supervisor AI orchestrating a data analysis system.

Available agents:
- analysis_agent: summaries, statistics, correlation analysis, data exploration
- visualization_agent: heatmaps, distribution plots, boxplots, pairplots
- cleaning_agent: missing values, duplicate removal, data quality, fill nulls
- ml_agent: model training, classification, regression, feature importance

Dataset schema:
{schema}

Session history (what has happened so far):
{memory}

User query: {query}

Choose the best single agent and write a precise task for it.
Respond ONLY in this exact format:
AGENT: <agent_name>
INPUT: <task description>"""


def _build_prompt(query: str, schema: str, memory: str) -> str:
    return SUPERVISOR_PROMPT.format(schema=schema, memory=memory, query=query)


def _parse_llm_response(response: str) -> tuple:
    agent_name = None
    agent_input = None
    for line in response.strip().splitlines():
        line = line.strip()
        if line.upper().startswith("AGENT:"):
            raw = line.split(":", 1)[1].strip().lower()
            for known in AGENT_REGISTRY:
                if known in raw or raw in known:
                    agent_name = known
                    break
            if not agent_name:
                agent_name = raw
        elif line.upper().startswith("INPUT:"):
            agent_input = line.split(":", 1)[1].strip()
    return agent_name, agent_input or ""


# ── LangGraph nodes ────────────────────────────────────────────────────────────

def detect_intent_node(state: SupervisorState) -> SupervisorState:
    """Detect whether the user wants to run a tool, get an explanation, or get business advice."""
    intent = detect_intent(state["user_query"])
    logger.info(f"Intent detected: {intent}")
    return {**state, "intent": intent}


def handle_explain(state: SupervisorState) -> SupervisorState:
    """
    User said 'explain that' or 'what does this mean'.
    We already have the previous output in memory — use Ollama to explain it.
    """
    last_output = state.get("last_output", "")
    query = state["user_query"]

    if not last_output:
        return {
            **state,
            "agent_output": "There is no previous output to explain yet. Please run an analysis first.",
            "selected_agent": "memory",
        }

    # Use Ollama if available to explain the output in plain language
    if check_ollama_health():
        prompt = f"""A data analysis tool produced the following output:

{last_output}

The user is asking: "{query}"

Explain the above output in clear, simple, non-technical language.
Focus on what the numbers mean, what patterns are important, and what a business user should understand.
Be concise and direct. Do not repeat the raw numbers — interpret them."""

        explanation = ollama_generate(prompt)
        if explanation.startswith("[ERROR]"):
            explanation = _fallback_explain(last_output)
    else:
        explanation = _fallback_explain(last_output)

    return {
        **state,
        "agent_output": explanation,
        "selected_agent": "explain_memory",
        "used_fallback_routing": not check_ollama_health(),
    }


def _fallback_explain(output: str) -> str:
    """Rule-based explanation when Ollama is offline."""
    lines = ["Here is an interpretation of the previous output:\n"]

    if "Accuracy" in output or "accuracy" in output:
        lines.append("The model accuracy tells you what percentage of predictions were correct.")
        lines.append("Higher is better — above 0.85 is generally considered good for most applications.")
    if "RMSE" in output or "R²" in output:
        lines.append("RMSE (Root Mean Squared Error) shows average prediction error — lower is better.")
        lines.append("R² shows how much of the variation is explained — closer to 1.0 is better.")
    if "correlat" in output.lower():
        lines.append("Correlation values range from -1 to +1.")
        lines.append("Values close to +1 mean two features increase together.")
        lines.append("Values close to -1 mean one increases as the other decreases.")
        lines.append("Values near 0 mean the features are not related.")
    if "missing" in output.lower():
        lines.append("Missing values can bias model training and should be filled or dropped.")
    if "feature" in output.lower() and "import" in output.lower():
        lines.append("Feature importance shows which columns most influence the prediction.")
        lines.append("Higher importance = more impact on the model's decisions.")
    if len(lines) == 1:
        lines.append("The previous analysis completed successfully.")
        lines.append("Ask a more specific question like 'what does the accuracy mean?' for a targeted explanation.")

    return "\n".join(lines)


def handle_business_insight(state: SupervisorState) -> SupervisorState:
    """
    User asked for a business decision or recommendation.
    Combines ML results + session history to give actionable advice.
    """
    ml_result = state.get("last_ml_result", {})
    last_output = state.get("last_output", "")
    query = state["user_query"]
    memory = state.get("memory_context", "")

    if check_ollama_health():
        prompt = f"""You are a data science consultant giving business advice.

Session history:
{memory}

Last analysis output:
{last_output}

ML results (if any): {ml_result}

User question: "{query}"

Give a clear, concise business recommendation based on the data analysis done so far.
Focus on:
1. What the results mean for the business
2. What action should be taken
3. Any risks or limitations to be aware of
Keep it under 200 words. Use plain language."""

        insight = ollama_generate(prompt)
        if insight.startswith("[ERROR]"):
            insight = _fallback_business_insight(ml_result, last_output)
    else:
        insight = _fallback_business_insight(ml_result, last_output)

    return {
        **state,
        "agent_output": insight,
        "selected_agent": "business_insight",
        "used_fallback_routing": not check_ollama_health(),
    }


def _fallback_business_insight(ml_result: dict, last_output: str) -> str:
    """Rule-based business advice when Ollama is offline."""
    lines = ["Business Insight Report\n"]

    if ml_result:
        task = ml_result.get("task", "")
        best_model = ml_result.get("best_model", "Unknown")
        lines.append(f"Recommended model: {best_model}")

        if task == "classification":
            acc = ml_result.get("best_accuracy", 0)
            if acc >= 0.95:
                lines.append(f"Accuracy: {acc:.1%} — Excellent. This model is production-ready.")
                lines.append("Recommendation: Deploy with confidence. Monitor for data drift monthly.")
            elif acc >= 0.80:
                lines.append(f"Accuracy: {acc:.1%} — Good. Acceptable for most business cases.")
                lines.append("Recommendation: Deploy with human oversight. Review edge cases.")
            elif acc >= 0.65:
                lines.append(f"Accuracy: {acc:.1%} — Fair. Needs improvement before production.")
                lines.append("Recommendation: Collect more labeled data. Try feature engineering.")
            else:
                lines.append(f"Accuracy: {acc:.1%} — Poor. Not suitable for production.")
                lines.append("Recommendation: Review data quality. Consider different features.")

        elif task == "regression":
            r2 = ml_result.get("best_r2", 0)
            if r2 >= 0.85:
                lines.append(f"R²: {r2:.3f} — Strong predictor. Model explains {r2:.1%} of variance.")
                lines.append("Recommendation: Ready for business forecasting use.")
            elif r2 >= 0.60:
                lines.append(f"R²: {r2:.3f} — Moderate predictor.")
                lines.append("Recommendation: Useful directionally, but add more features.")
            else:
                lines.append(f"R²: {r2:.3f} — Weak predictor.")
                lines.append("Recommendation: Reconsider features. The chosen columns may not predict the target well.")

        feats = ml_result.get("top_features", [])
        if feats:
            lines.append(f"\nKey business drivers: {', '.join(feats[:3])}")
            lines.append("Focus business decisions on these factors for maximum impact.")
    else:
        lines.append("No ML results found yet. Please train models first.")
        lines.append("Ask: 'Train ML models' to get model performance metrics.")
        if "missing" in last_output.lower():
            lines.append("\nNote: Missing values were detected. Clean data before training for best results.")

    return "\n".join(lines)


def supervisor_decide(state: SupervisorState) -> SupervisorState:
    """Call LLM or keyword routing to pick which agent handles the query."""
    query = state["user_query"]
    logger.info(f"Supervisor deciding for: {query[:80]}")

    ollama_up = check_ollama_health()
    dataset_name = state.get("dataset_name", "unknown")

    # For non-Iris datasets, prefer keyword routing to avoid LLM issues
    if dataset_name != "iris":
        agent_name, agent_input = _keyword_route(query)
        return {
            **state,
            "selected_agent": agent_name,
            "agent_input": agent_input,
            "error": None,
            "used_fallback_routing": True,
        }

    if ollama_up:
        prompt = _build_prompt(query, state["schema_text"], state["memory_context"])
        raw = ollama_generate(prompt)
        logger.debug(f"LLM response: {raw[:200]}")

        if not raw.startswith("[ERROR]"):
            agent_name, agent_input = _parse_llm_response(raw)
            if agent_name and agent_name in AGENT_REGISTRY:
                return {
                    **state,
                    "selected_agent": agent_name,
                    "agent_input": agent_input or query,
                    "error": None,
                    "used_fallback_routing": False,
                }

    # Keyword fallback
    agent_name, agent_input = _keyword_route(query)
    return {
        **state,
        "selected_agent": agent_name,
        "agent_input": agent_input,
        "error": None,
        "used_fallback_routing": True,
    }


def run_selected_agent(state: SupervisorState) -> SupervisorState:
    """Run the chosen specialized agent."""
    agent_name = state.get("selected_agent")
    agent_input = state.get("agent_input", state["user_query"])

    if not agent_name:
        return {**state, "agent_output": "No agent selected. Please try again."}

    try:
        agent = get_agent(agent_name)
        logger.info(f"Running {agent_name} | input: {agent_input[:80]}")
        result = agent.invoke({"input": agent_input})
        output = result.get("output", str(result))
        if not output or not output.strip():
            output = f"Agent {agent_name} ran but returned no output. Try rephrasing."
    except Exception as e:
        output = f"Agent {agent_name} error: {e}"
        logger.error(f"Agent {agent_name} failed: {e}", exc_info=True)

    return {**state, "agent_output": output}


def format_final_response(state: SupervisorState) -> SupervisorState:
    """Wrap output with agent label and fallback notice."""
    agent_name = state.get("selected_agent", "unknown")
    output = state.get("agent_output", "No output.")
    fallback = state.get("used_fallback_routing", False)
    intent = state.get("intent", "run_tool")

    label_map = {
        "explain_memory":   "Context Explanation",
        "business_insight": "Business Insight",
    }
    label = label_map.get(agent_name, agent_name.replace("_", " ").title())

    note = ""
    if fallback and intent == "run_tool":
        note = "\n\n> ⚡ *Ollama offline — keyword routing used. Start `ollama serve` for LLM mode.*"

    response = f"**[{label}]**\n\n{output}{note}"
    return {**state, "final_response": response}


# ── Route function for LangGraph ───────────────────────────────────────────────
def route_by_intent(state: SupervisorState) -> str:
    intent = state.get("intent", "run_tool")
    if intent == "explain":
        return "handle_explain"
    if intent == "business_insight":
        return "handle_business_insight"
    return "supervisor_decide"


# ── Build graph ────────────────────────────────────────────────────────────────
def build_supervisor_graph():
    workflow = StateGraph(SupervisorState)

    workflow.add_node("detect_intent_node", detect_intent_node)
    workflow.add_node("handle_explain", handle_explain)
    workflow.add_node("handle_business_insight", handle_business_insight)
    workflow.add_node("supervisor_decide", supervisor_decide)
    workflow.add_node("run_selected_agent", run_selected_agent)
    workflow.add_node("format_final_response", format_final_response)

    workflow.set_entry_point("detect_intent_node")

    workflow.add_conditional_edges(
        "detect_intent_node",
        route_by_intent,
        {
            "handle_explain": "handle_explain",
            "handle_business_insight": "handle_business_insight",
            "supervisor_decide": "supervisor_decide",
        },
    )

    workflow.add_edge("handle_explain", "format_final_response")
    workflow.add_edge("handle_business_insight", "format_final_response")
    workflow.add_edge("supervisor_decide", "run_selected_agent")
    workflow.add_edge("run_selected_agent", "format_final_response")
    workflow.add_edge("format_final_response", END)

    return workflow.compile()


# ── Public interface ───────────────────────────────────────────────────────────
class SupervisorAgent:
    def __init__(self, schema: Dict[str, Any], dataset_name: str = "unknown"):
        self.schema = schema
        self._dataset_name = dataset_name
        self.schema_text = self._build_rich_schema(schema)   # ✅ FIXED
        self.graph = build_supervisor_graph()
        logger.info("SupervisorAgent initialized with robust schema.")

    # ✅ Robust schema builder (handles all formats)
    def _build_rich_schema(self, schema: Dict[str, Any]) -> str:
        if not schema:
            return "No dataset loaded."

        columns = []
        column_types = {}

        # ✅ Case 1: proper dict format
        if isinstance(schema.get("columns"), dict):
            column_types = schema.get("columns", {})
            columns = list(column_types.keys())

        # ✅ Case 2: list of columns
        elif isinstance(schema.get("columns"), list):
            columns = schema.get("columns", [])

        # ✅ Case 3: fallback keys
        elif "column_names" in schema:
            columns = schema.get("column_names", [])

        # ✅ Last fallback: nothing usable
        else:
            columns = []

        # ✅ Shape handling
        shape = schema.get("shape", None)
        if isinstance(shape, (list, tuple)) and len(shape) == 2:
            rows, cols = shape
        else:
            rows, cols = "Unknown", len(columns)

        # 🔥 Truncate columns to first 10 for LLM prompt brevity
        if len(columns) > 10:
            columns_display = columns[:10] + [f"... and {len(columns)-10} more"]
        else:
            columns_display = columns

        return f"""
DATASET DETAILS:

Columns: {', '.join(str(col) for col in columns_display)}

Total Rows: {rows}
Total Columns: {cols}

IMPORTANT RULES:
- Always use ONLY these column names
- Do NOT assume dataset structure
- Match user query words with column names
"""

    def invoke(
        self,
        query: str,
        memory_context: str = "",
        last_output: str = "",
        last_ml_result: dict = None,
    ) -> Dict[str, Any]:

        # ✅ Safety check
        if not self.schema:
            return {
                "final_response": "⚠️ No dataset loaded. Please upload a dataset first.",
                "selected_agent": None,
                "agent_output": "",
                "intent": "run_tool",
                "error": "No schema",
                "used_fallback_routing": False,
            }

        # Direct execution without LangGraph
        state = {
            "user_query": query,
            "schema_text": self.schema_text,
            "memory_context": memory_context,
            "last_output": last_output,
            "last_ml_result": last_ml_result or {},
            "dataset_name": self._dataset_name,
        }

        # Detect intent
        state["intent"] = detect_intent(query)

        if state["intent"] == "explain":
            result = handle_explain(state)
        elif state["intent"] == "business_insight":
            result = handle_business_insight(state)
        else:
            # Run tool
            state = supervisor_decide(state)
            if state.get("selected_agent"):
                state = run_selected_agent(state)
            else:
                state["agent_output"] = "No agent selected."
            result = format_final_response(state)

        return {
            "final_response": result.get("final_response", ""),
            "selected_agent": result.get("selected_agent"),
            "agent_output": result.get("agent_output", ""),
            "intent": result.get("intent", "run_tool"),
            "error": result.get("error"),
            "used_fallback_routing": result.get("used_fallback_routing", False),
        }