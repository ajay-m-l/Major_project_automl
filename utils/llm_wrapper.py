"""
LangChain-compatible LLM wrapper for Ollama.
Allows Ollama to be used as a drop-in LLM inside LangChain agents and chains.
"""

import logging
from typing import Any, List, Optional, Iterator

from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from pydantic import Field

from utils.ollama import generate as ollama_generate, DEFAULT_MODEL

logger = logging.getLogger(__name__)


class OllamaLLM(LLM):
    """
    LangChain LLM wrapper that routes calls to a local Ollama instance.

    Uses phi3:latest by default (lightweight, suitable for low-RAM systems).
    """

    model: str = Field(default=DEFAULT_MODEL, description="Ollama model name")
    temperature: float = Field(default=0.1, description="Sampling temperature")
    max_tokens: int = Field(default=1024, description="Max tokens to generate")

    @property
    def _llm_type(self) -> str:
        return "ollama"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Call Ollama with the given prompt and return the text response.
        """
        logger.debug(f"OllamaLLM._call called, prompt_length={len(prompt)}")

        # Build the full prompt with optional stop injection
        full_prompt = prompt
        if stop:
            # Append stop instruction in prompt (Ollama doesn't natively support stop tokens via API)
            stop_hint = f"\n[Stop generating if you encounter any of these: {', '.join(stop)}]"
            full_prompt = prompt + stop_hint

        response = ollama_generate(full_prompt, model=self.model)

        # Trim response at first stop token if provided
        if stop:
            for token in stop:
                if token in response:
                    response = response[: response.index(token)]

        return response.strip()

    @property
    def _identifying_params(self) -> dict:
        return {"model": self.model, "temperature": self.temperature}


def get_llm(model: str = DEFAULT_MODEL) -> OllamaLLM:
    """
    Factory function to create an OllamaLLM instance.

    Args:
        model: Ollama model name (default: phi3:latest)

    Returns:
        Configured OllamaLLM instance
    """
    logger.info(f"Creating OllamaLLM with model={model}")
    return OllamaLLM(model=model)
