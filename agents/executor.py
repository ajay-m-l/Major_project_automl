"""
AgentSystem — Central coordinator.
Now integrates SessionMemory so every query/response is remembered
and injected into the next supervisor call.
"""

import logging
import pandas as pd
from typing import Any, Dict, List, Optional

from utils.schema import extract_schema
from utils.memory import SessionMemory
from agents.supervisor_agent import SupervisorAgent

import tools.analysis_tool as analysis_tool
import tools.visualization_tool as visualization_tool
import tools.cleaning_tool as cleaning_tool
import tools.ml_tool as ml_tool

logger = logging.getLogger(__name__)


class AgentSystem:
    def __init__(self):
        self._supervisor: Optional[SupervisorAgent] = None
        self._dataset: Optional[pd.DataFrame] = None
        self._schema: Optional[Dict[str, Any]] = None
        self._dataset_name: str = "unknown"
        self.memory = SessionMemory()
        logger.info("AgentSystem initialized.")

    def load_dataset(self, df: pd.DataFrame, name: str = "dataset", update_callback=None):
        df = df.copy()
        df.columns = [str(col) for col in df.columns.tolist()]
        self._dataset = df
        self._schema = extract_schema(df)
        self._dataset_name = name

        # Register with all tool modules
        analysis_tool.set_dataset(df)
        visualization_tool.set_dataset(df)
        cleaning_tool.set_dataset(df, update_callback=update_callback)
        ml_tool.set_dataset(df)
        ml_tool.set_memory(self.memory)   # so ML tool can push results back

        # Update memory with dataset info
        self.memory.set_dataset_info(name, df.shape)

        # Rebuild supervisor with fresh schema
        self._supervisor = SupervisorAgent(schema=self._schema, dataset_name=self._dataset_name)
        logger.info(f"AgentSystem: dataset '{name}' loaded {df.shape}")

    def run(self, query: str) -> Dict[str, Any]:
        if self._supervisor is None:
            return {
                "final_response": "⚠️ No dataset loaded. Please upload a CSV or load the Iris dataset first.",
                "selected_agent": None,
                "agent_output": "",
                "figure": None,
                "error": "No dataset",
                "intent": "run_tool",
            }

        # Build context from session memory
        memory_context = self.memory.get_context_for_llm(max_turns=6)
        last_output = self.memory.get_last_output()
        last_ml_result = self.memory.get_last_ml_result()

        result = self._supervisor.invoke(
            query=query,
            memory_context=memory_context,
            last_output=last_output,
            last_ml_result=last_ml_result,
        )

        # Attach figure if visualization ran
        result["figure"] = visualization_tool.get_last_figure()

        # Determine if a figure was generated
        figure_generated = result["figure"] is not None

        # Save this turn to memory
        self.memory.add_turn(
            user_query=query,
            agent_used=result.get("selected_agent") or "unknown",
            tool_output=result.get("agent_output", ""),
            figure_generated=figure_generated,
            used_fallback=result.get("used_fallback_routing", False),
        )

        # Re-register cleaned dataset if cleaning tool modified it
        cleaned = cleaning_tool.get_dataset()
        if cleaned is not None and self._dataset is not None:
            if cleaned.shape != self._dataset.shape or not cleaned.equals(self._dataset):
                logger.info("Dataset modified by cleaning agent — re-registering.")
                self._dataset = cleaned
                self._schema = extract_schema(cleaned)
                analysis_tool.set_dataset(cleaned)
                visualization_tool.set_dataset(cleaned)
                ml_tool.set_dataset(cleaned)
                self.memory.set_dataset_info(self._dataset_name, cleaned.shape)
                # Rebuild supervisor with updated schema
                self._supervisor = SupervisorAgent(schema=self._schema, dataset_name=self._dataset_name)

        return result

    def get_schema(self):
        return self._schema

    def get_dataset(self):
        # Always return the latest (possibly cleaned) dataset
        cleaned = cleaning_tool.get_dataset()
        if cleaned is not None:
            return cleaned
        return self._dataset

    def is_ready(self) -> bool:
        return self._supervisor is not None

    def clear_memory(self):
        self.memory.clear()
        logger.info("Session memory cleared.")

    def get_memory_summary(self) -> Dict:
        return self.memory.summary()


# Singleton
agent_system = AgentSystem()
