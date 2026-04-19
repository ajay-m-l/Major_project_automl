"""
Conversational Multi-Agent AutoML System — Main Streamlit App
=============================================================
Full session memory: every query + output is remembered.
Supports: explain follow-ups, business insight queries, multi-turn conversations.
"""

import logging
import sys
import os
import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.executor import agent_system
from utils.schema import get_dataset_intelligence
from utils.ollama import check_ollama_health, list_models

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("automl_system.log")],
)
logger = logging.getLogger(__name__)

# ── Session state ──────────────────────────────────────────────────────────────
def _init():
    defaults = {
        "chat_history": [],    # list of {role, content, figure, agent, intent}
        "dataset_loaded": False,
        "current_df": None,
        "dataset_name": "unknown",
        "pending_query": None,
        "last_selected_agent": None,
        "last_routing_method": None,
        "last_routing_input": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()


def normalize_column_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all column labels to strings and make duplicates unique."""
    df = df.copy()
    cols = [str(col) for col in df.columns.tolist()]
    if len(set(cols)) != len(cols):
        seen = {}
        unique_cols = []
        for col in cols:
            if col not in seen:
                seen[col] = 0
                unique_cols.append(col)
            else:
                seen[col] += 1
                unique_cols.append(f"{col}_{seen[col]}")
        cols = unique_cols
    df.columns = cols
    return df


def on_dataset_updated(new_df):
    st.session_state["current_df"] = new_df


# Ensure agent system is ready after reruns by reloading persisted dataset if needed
if st.session_state.get("dataset_loaded") and st.session_state.get("current_df") is not None:
    if not agent_system.is_ready() or agent_system.get_dataset() is None:
        agent_system.load_dataset(
            st.session_state["current_df"],
            name=st.session_state.get("dataset_name", "dataset"),
            update_callback=on_dataset_updated,
        )


# Re-set dataset in tools if loaded (Streamlit resets globals on rerun)
if st.session_state.get("dataset_loaded") and st.session_state.get("current_df") is not None:
    df = st.session_state["current_df"]
    import tools.analysis_tool as analysis_tool
    import tools.visualization_tool as visualization_tool
    import tools.cleaning_tool as cleaning_tool
    import tools.ml_tool as ml_tool
    analysis_tool.set_dataset(df)
    visualization_tool.set_dataset(df)
    cleaning_tool.set_dataset(df, update_callback=on_dataset_updated)
    ml_tool.set_dataset(df)
    ml_tool.set_memory(agent_system.memory)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 Multi-Agent AutoML")

    # Ollama status
    ollama_ok = check_ollama_health()
    if ollama_ok:
        models = list_models()
        st.success(f"✅ Ollama online — {', '.join(models[:2]) if models else 'ready'}")
    else:
        st.warning("⚡ Ollama offline — keyword routing active")
        with st.expander("Enable full LLM mode"):
            st.code("ollama pull phi3\nollama serve", language="bash")

    if st.session_state.get("dataset_loaded"):
        if agent_system.is_ready():
            st.success(f"✅ Agent ready for '{st.session_state['dataset_name']}'")
        else:
            st.warning("⚠️ Dataset loaded but agent is not ready yet. Reload the dataset if needed.")

        try:
            loaded_shape = agent_system.get_dataset().shape
        except Exception:
            loaded_shape = None

        st.divider()
        st.markdown("### 🧪 Debug State")
        st.markdown(f"- Agent ready: `{agent_system.is_ready()}`")
        st.markdown(f"- Dataset shape: `{loaded_shape}`")
        st.markdown(f"- Chat history length: `{len(st.session_state['chat_history'])}`")
        st.markdown(f"- Pending query: `{st.session_state.get('pending_query')}`")
        st.markdown(f"- Last selected agent: `{st.session_state.get('last_selected_agent')}`")
        st.markdown(f"- Last routing method: `{st.session_state.get('last_routing_method')}`")
        if st.session_state.get("last_routing_input"):
            st.markdown("**Last routing input snippet:**")
            st.code(st.session_state["last_routing_input"], language="text")

        if st.session_state['chat_history']:
            last = st.session_state['chat_history'][-1]
            st.markdown(f"- Last history role: `{last.get('role')}`")
            st.markdown(f"- Last history content: `{str(last.get('content'))[:120]}`")

    st.divider()
    st.markdown("### 📂 Dataset")
    source = st.radio("Source", ["Load Iris Dataset", "Upload CSV"], key="data_source", label_visibility="collapsed")

    df_to_load = None
    name_to_use = "dataset"

    if source == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV", type=["csv"], key="csv_upload")
        if uploaded:
            uploaded_name = uploaded.name.replace(".csv", "")
            should_load_csv = (
                not st.session_state["dataset_loaded"]
                or st.session_state["dataset_name"] != uploaded_name
            )

            if should_load_csv:
                try:
                    # Try multiple encodings and separators for robustness
                    df_to_load = None
                    for encoding in ['utf-8', 'latin-1', 'cp1252']:
                        for sep in [',', ';', '\t']:
                            try:
                                uploaded.seek(0)
                                df_to_load = pd.read_csv(
                                    uploaded,
                                    encoding=encoding,
                                    sep=sep,
                                    on_bad_lines='skip',
                                    low_memory=False,
                                )
                                break
                            except (UnicodeDecodeError, pd.errors.ParserError):
                                continue
                        if df_to_load is not None:
                            break

                    if df_to_load is None:
                        raise ValueError("Could not parse CSV with common encodings/separators")

                    df_to_load = normalize_column_labels(df_to_load)
                    name_to_use = uploaded_name

                    # Validate the loaded dataset
                    if df_to_load.empty:
                        raise ValueError("CSV is empty or has no valid rows")
                    if df_to_load.shape[1] == 0:
                        raise ValueError("CSV has no columns")

                    st.success(f"✅ {uploaded.name} loaded ({df_to_load.shape[0]} rows, {df_to_load.shape[1]} columns)")
                    logger.info(f"Uploaded CSV loaded: {uploaded.name} ({df_to_load.shape[0]} rows, {df_to_load.shape[1]} columns)")
                except Exception as e:
                    st.error(f"Error loading CSV: {e}. Please ensure it's a valid CSV file.")
            else:
                st.success(f"✅ {uploaded.name} already loaded")
    else:
        if st.button("📥 Load Iris Dataset", use_container_width=True):
            from sklearn.datasets import load_iris
            iris = load_iris(as_frame=True)
            df_to_load = iris.frame
            df_to_load["target_name"] = iris.target_names[iris.target]
            df_to_load = normalize_column_labels(df_to_load)
            name_to_use = "iris"
            st.success("✅ Iris dataset loaded")

    if df_to_load is not None:
        # Clear memory when a new dataset is loaded
        agent_system.clear_memory()
        st.session_state["chat_history"] = []
        agent_system.load_dataset(df_to_load, name=name_to_use, update_callback=on_dataset_updated)
        st.session_state["current_df"] = df_to_load
        st.session_state["dataset_name"] = name_to_use
        st.session_state["dataset_loaded"] = True

    # ── Dataset Intelligence Panel ─────────────────────────────────────────────
    if st.session_state["dataset_loaded"] and st.session_state["current_df"] is not None:
        df = st.session_state["current_df"]
        intel = get_dataset_intelligence(df)
        st.divider()
        st.markdown(f"### 📊 {st.session_state['dataset_name'].title()}")
        c1, c2 = st.columns(2)
        c1.metric("Rows", intel["shape"][0])
        c2.metric("Columns", intel["shape"][1])
        c1.metric("Duplicates", intel["duplicates"])
        c2.metric("Missing Cells", sum(intel["missing"].values()))

        with st.expander("📋 Columns"):
            for col in df.columns:
                missing = intel["missing"].get(col, 0)
                pct = intel["missing_pct"].get(col, 0.0)
                icon = "⚠️" if missing > 0 else "✅"
                st.markdown(
                    f"<div class='stat-card'>{icon} <b>{col}</b><br>"
                    f"<small>{str(intel['dtypes'][col])} | missing: {missing} ({pct}%)</small></div>",
                    unsafe_allow_html=True,
                )

        with st.expander("📈 Statistics"):
            num = df.select_dtypes(include="number")
            if not num.empty:
                st.dataframe(num.describe().round(3), use_container_width=True)

        with st.expander("🔍 Preview"):
            st.dataframe(df.head(8), use_container_width=True)

    # ── Session Memory Panel ───────────────────────────────────────────────────
    if st.session_state["dataset_loaded"]:
        st.divider()
        mem_summary = agent_system.get_memory_summary()
        st.markdown("### 🧠 Session Memory")
        st.markdown(f"**{mem_summary['total_turns']} turns** remembered")
        if mem_summary["agents_used"]:
            st.markdown(f"Agents used: {', '.join(mem_summary['agents_used'])}")
        if mem_summary["has_ml_results"]:
            st.markdown("✅ ML results stored — ask for business insights")

        last_agent = st.session_state.get("last_selected_agent")
        last_route = st.session_state.get("last_routing_method")
        if last_agent:
            st.markdown("---")
            st.markdown("### 🧪 Routing diagnostics")
            st.markdown(f"**Last selected agent:** `{last_agent}`")
            st.markdown(f"**Routing method:** `{last_route}`")
            if st.session_state.get("last_routing_input"):
                st.markdown("**Last routing input snippet:**")
                st.code(st.session_state["last_routing_input"], language="text")

    st.divider()
    st.markdown("### Agents")
    for name, desc in [
        ("🔍 Analysis", "Stats, summaries, correlation"),
        ("📊 Visualization", "Heatmaps, charts, plots"),
        ("🧹 Cleaning", "Missing values, duplicates"),
        ("🤖 ML / AutoML", "Train models, metrics"),
        ("💡 Explain", "Explain previous output"),
        ("📈 Insights", "Business recommendations"),
    ]:
        st.markdown(f"**{name}** — {desc}")

    st.divider()
    col1, col2 = st.columns(2)
    if col1.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state["chat_history"] = []
        agent_system.clear_memory()
        st.rerun()
    if col2.button("🔄 New Session", use_container_width=True):
        st.session_state["chat_history"] = []
        st.session_state["dataset_loaded"] = False
        st.session_state["current_df"] = None
        agent_system.clear_memory()
        st.rerun()


# ── Main UI ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1 style="margin:0;font-size:1.7rem;">🤖 Conversational Multi-Agent AutoML System</h1>
    <p style="margin:0.3rem 0 0;opacity:0.9;font-size:0.88rem;">
        Full session memory · Intent detection · Explain follow-ups · Business insights
    </p>
</div>
""", unsafe_allow_html=True)

# Render chat history
for msg in st.session_state["chat_history"]:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            agent = msg.get("agent", "")
            intent = msg.get("intent", "run_tool")
            badge_map = {
                "analysis_agent":      ("Analysis Agent",  "badge-analysis"),
                "visualization_agent": ("Visualization",   "badge-viz"),
                "cleaning_agent":      ("Cleaning Agent",  "badge-cleaning"),
                "ml_agent":            ("ML / AutoML",     "badge-ml"),
                "explain_memory":      ("Explaining",      "badge-explain"),
                "business_insight":    ("Business Insight","badge-insight"),
            }
            label, cls = badge_map.get(agent, (agent, "badge-analysis"))
            turns = agent_system.memory.summary()["total_turns"]
            st.markdown(
                f"<span class='agent-badge {cls}'>{label}</span>"
                f"<span class='memory-pill'>turn {msg.get('turn', '?')} · {turns} in memory</span>",
                unsafe_allow_html=True,
            )
        st.markdown(msg["content"])
        if msg.get("figure") is not None:
            try:
                st.pyplot(msg["figure"])
            except Exception:
                pass

# Quick-start suggestions
if st.session_state["dataset_loaded"] and not st.session_state["chat_history"]:
    st.markdown("#### 💡 Try these — in order — to see full session memory in action:")
    flow1 = [
        "Show me the data quality report",
        "Fill missing values with the column mean",
        "Show me a correlation heatmap",
        "Explain what that correlation means",
        "Train ML models",
        "What should I do to improve the business?",
    ]
    flow2 = [
        "Summarize the dataset",
        "Show feature distributions",
        "Remove duplicate rows",
        "Which ML task is this suited for?",
        "Train models and show accuracy",
        "Which model should I deploy?",
    ]
    st.markdown("**Pipeline A — Data quality → Clean → Visualize → ML → Insights:**")
    cols = st.columns(3)
    for i, s in enumerate(flow1):
        if cols[i % 3].button(s, key=f"a_{i}", use_container_width=True):
            st.session_state["pending_query"] = s
            st.rerun()

    st.markdown("**Pipeline B — Explore → Clean → ML → Decision:**")
    cols2 = st.columns(3)
    for i, s in enumerate(flow2):
        if cols2[i % 3].button(s, key=f"b_{i}", use_container_width=True):
            st.session_state["pending_query"] = s
            st.rerun()

# Handle suggestion click
if st.session_state.get("pending_query"):
    query = st.session_state.pop("pending_query")
    st.session_state["chat_history"].append({"role": "user", "content": query})
    st.rerun()

# ── Process last unprocessed user message ──────────────────────────────────────
# Check if the last message is user and has no assistant reply yet
history = st.session_state["chat_history"]
needs_response = (
    history
    and history[-1]["role"] == "user"
    and (len(history) < 2 or history[-2]["role"] != "assistant" or history[-1] != history[-2])
)

logger.info(f"need_response={needs_response} history_len={len(history)} last_role={history[-1]['role'] if history else None}")
if history:
    logger.info(f"last_history={history[-1]}")

if needs_response:
    query = history[-1]["content"]
    turn_number = len([m for m in history if m["role"] == "assistant"]) + 1

    with st.chat_message("assistant"):
        status = st.empty()
        ollama_status = "🧠 LLM routing" if check_ollama_health() else "⚡ Keyword routing"
        status.info(f"{ollama_status} — processing your query...")

        try:
            logger.info(f"Processing user query: {query}")
            result = agent_system.run(query)
            logger.info(f"Run result keys: {list(result.keys()) if isinstance(result, dict) else type(result)}")
            logger.info(f"Run result selected_agent={result.get('selected_agent')} intent={result.get('intent')} error={result.get('error')}")
        except Exception as e:
            result = {
                "final_response": f"System error: {e}",
                "selected_agent": None,
                "figure": None,
                "intent": "run_tool",
                "used_fallback_routing": True,
                "agent_output": "",
            }
            logger.error(f"run error: {e}", exc_info=True)

        status.empty()

        agent_used = result.get("selected_agent") or "unknown"
        intent = result.get("intent", "run_tool")

        badge_map = {
            "analysis_agent":      ("Analysis Agent",  "badge-analysis"),
            "visualization_agent": ("Visualization",   "badge-viz"),
            "cleaning_agent":      ("Cleaning Agent",  "badge-cleaning"),
            "ml_agent":            ("ML / AutoML",     "badge-ml"),
            "explain_memory":      ("Explaining",      "badge-explain"),
            "business_insight":    ("Business Insight","badge-insight"),
        }
        label, cls = badge_map.get(agent_used, (agent_used, "badge-analysis"))
        fallback_tag = " ⚡" if result.get("used_fallback_routing") else ""
        mem_turns = agent_system.memory.summary()["total_turns"]

        st.markdown(
            f"<span class='agent-badge {cls}'>{label}{fallback_tag}</span>"
            f"<span class='memory-pill'>turn {turn_number} · {mem_turns} in memory</span>",
            unsafe_allow_html=True,
        )

        response_text = result.get("final_response") or "⚠️ No response. Please try again."
        st.markdown(response_text)

        figure = result.get("figure")
        if figure is not None:
            try:
                st.pyplot(figure)
                plt.close(figure)
            except Exception:
                pass

        if result.get("error"):
            st.error(f"Error: {result['error']}")

    # Persist fail-safe diagnostics for sidebar display
    st.session_state["last_selected_agent"] = agent_used
    st.session_state["last_routing_method"] = "LLM" if not result.get("used_fallback_routing") else "Keyword fallback"
    st.session_state["last_routing_input"] = result.get("agent_output", "")[:300]

    # Save assistant response to chat history
    st.session_state["chat_history"].append({
        "role": "assistant",
        "content": response_text,
        "figure": figure,
        "agent": agent_used,
        "intent": intent,
        "turn": turn_number,
    })

    # Refresh dataset display if cleaning modified it
    updated = agent_system.get_dataset()
    if updated is not None:
        st.session_state["current_df"] = updated

    logger.info(f"Turn {turn_number} done | agent={agent_used} | intent={intent} | query='{query[:60]}'")

# ── Chat input ─────────────────────────────────────────────────────────────────
query = st.chat_input(
    "Ask anything — try 'explain that', 'what should I do?', 'fill missing values', 'train models'..."
)
if query:
    logger.info(f"Chat input returned query: {query}")
    if not st.session_state["dataset_loaded"]:
        st.error("⚠️ Load a dataset first using the sidebar.")
        st.stop()
    st.session_state["chat_history"].append({"role": "user", "content": query})
    st.rerun()

# ── Empty state ────────────────────────────────────────────────────────────────
if not st.session_state["dataset_loaded"]:
    st.markdown("""
    <div style="text-align:center;padding:3rem;color:#64748b;">
        <h2>👈 Load a dataset to start</h2>
        <p>Upload any CSV or load the built-in Iris dataset from the sidebar.</p>
        <br>
        <p style="font-size:0.9rem;">
            <b>This system remembers your entire conversation.</b><br>
            Ask "show correlation", then "explain that", then "train models",<br>
            then "what should I do for the business?" — it all connects.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<hr style="margin-top:2rem;border:none;border-top:1px solid #e2e8f0;">
<p style="text-align:center;color:#94a3b8;font-size:0.78rem;">
Multi-Agent AutoML · Session Memory · Intent Detection · Ollama phi3 · LangGraph · Streamlit
</p>
""", unsafe_allow_html=True)
