"""
Session Memory Manager
======================
Stores the full conversation history per session including:
  - user queries
  - which agent was used
  - the actual tool output (text result)
  - any figure that was generated
  - dataset state snapshots (name, shape)

This enables true context-aware follow-up questions like:
  "explain the correlation map"   → system knows the previous output was a heatmap
  "now fill missing values"       → system knows the dataset has nulls from earlier report
  "which model should I use?"     → system knows the ML results from previous turn
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """One complete exchange: user query → agent response."""
    turn_id: int
    timestamp: str
    user_query: str
    agent_used: str
    tool_output: str          # raw text output from the tool
    figure_generated: bool    # whether a chart was produced
    dataset_name: str         # which dataset was active
    dataset_shape: tuple      # (rows, cols) at time of query
    used_fallback: bool       # whether Ollama was offline


class SessionMemory:
    """
    Manages full conversation memory for one session.
    Injected into every supervisor prompt so the LLM has full context.
    """

    def __init__(self):
        self.turns: List[ConversationTurn] = []
        self._dataset_name: str = "unknown"
        self._dataset_shape: tuple = (0, 0)
        self._last_ml_result: Dict[str, Any] = {}
        self._last_tool_output: str = ""
        self._last_agent: str = ""
        logger.info("SessionMemory initialized")

    def set_dataset_info(self, name: str, shape: tuple):
        self._dataset_name = name
        self._dataset_shape = shape

    def add_turn(
        self,
        user_query: str,
        agent_used: str,
        tool_output: str,
        figure_generated: bool = False,
        used_fallback: bool = False,
    ):
        turn = ConversationTurn(
            turn_id=len(self.turns) + 1,
            timestamp=datetime.now().strftime("%H:%M:%S"),
            user_query=user_query,
            agent_used=agent_used,
            tool_output=tool_output,
            figure_generated=figure_generated,
            dataset_name=self._dataset_name,
            dataset_shape=self._dataset_shape,
            used_fallback=used_fallback,
        )
        self.turns.append(turn)
        self._last_tool_output = tool_output
        self._last_agent = agent_used
        logger.info(f"Memory: turn {turn.turn_id} saved | agent={agent_used} | query='{user_query[:50]}'")

    def store_ml_result(self, result_dict: Dict[str, Any]):
        """Store ML training results for business insight follow-ups."""
        self._last_ml_result = result_dict
        logger.info(f"Memory: ML result stored — {result_dict}")

    def get_last_output(self) -> str:
        return self._last_tool_output

    def get_last_agent(self) -> str:
        return self._last_agent

    def get_last_ml_result(self) -> Dict[str, Any]:
        return self._last_ml_result

    def get_context_for_llm(self, max_turns: int = 6) -> str:
        """
        Build a compact context string to inject into the supervisor prompt.
        Gives the LLM full awareness of what happened earlier in the session.
        """
        if not self.turns:
            return "No previous conversation in this session."

        recent = self.turns[-max_turns:]
        lines = [f"Session so far ({len(self.turns)} total turns):"]

        for t in recent:
            lines.append(f"\n[Turn {t.turn_id} | {t.timestamp}]")
            lines.append(f"  User asked : {t.user_query}")
            lines.append(f"  Agent used : {t.agent_used}")
            if t.figure_generated:
                lines.append(f"  Output     : [chart/figure generated]")
            else:
                # Include first 300 chars of previous output so LLM can explain it
                snippet = t.tool_output[:300].replace("\n", " ")
                lines.append(f"  Output     : {snippet}{'...' if len(t.tool_output) > 300 else ''}")
            lines.append(f"  Dataset    : {t.dataset_name} {t.dataset_shape}")

        if self._last_ml_result:
            lines.append(f"\nLast ML result: {self._last_ml_result}")

        return "\n".join(lines)

    def build_explanation_context(self) -> str:
        """
        When user asks 'explain that' or 'what does this mean',
        return the full last output for the LLM to explain.
        """
        if not self._last_tool_output:
            return "No previous output to explain."
        return self._last_tool_output

    def clear(self):
        self.turns.clear()
        self._last_ml_result = {}
        self._last_tool_output = ""
        self._last_agent = ""
        logger.info("SessionMemory cleared")

    def summary(self) -> Dict[str, Any]:
        return {
            "total_turns": len(self.turns),
            "agents_used": list({t.agent_used for t in self.turns}),
            "dataset": self._dataset_name,
            "has_ml_results": bool(self._last_ml_result),
            "last_agent": self._last_agent,
        }
