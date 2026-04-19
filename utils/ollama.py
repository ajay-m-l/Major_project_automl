"""
Ollama API wrapper — handles raw HTTP calls to local Ollama server.
Provides a clean interface for generating completions from a local LLM.
"""

import requests
import json
import logging

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "phi3"


def generate(prompt: str, model: str = DEFAULT_MODEL, stream: bool = False) -> str:
    """
    Send a prompt to the Ollama /api/generate endpoint and return the response text.

    Args:
        prompt: The text prompt to send.
        model: Ollama model name (default: phi3).
        stream: Whether to stream the response (default: False).

    Returns:
        Generated text as a string, or an error message on failure.
    """
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": stream,
    }

    try:
        logger.debug(f"Calling Ollama model={model}, prompt_length={len(prompt)}")
        response = requests.post(url, json=payload, timeout=15)
        response.raise_for_status()

        if stream:
            # Collect streamed chunks
            full_text = ""
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    full_text += chunk.get("response", "")
                    if chunk.get("done", False):
                        break
            return full_text.strip()
        else:
            data = response.json()
            return data.get("response", "").strip()

    except requests.exceptions.ConnectionError:
        logger.error("Cannot connect to Ollama. Is 'ollama serve' running?")
        return "[ERROR] Cannot connect to Ollama. Please run: ollama serve"

    except requests.exceptions.Timeout:
        logger.error("Ollama request timed out.")
        return "[ERROR] Ollama request timed out. Try a shorter prompt or faster model."

    except requests.exceptions.HTTPError as e:
        logger.error(f"Ollama HTTP error: {e}")
        return f"[ERROR] Ollama returned HTTP error: {e}"

    except Exception as e:
        logger.error(f"Unexpected Ollama error: {e}")
        return f"[ERROR] Unexpected error calling Ollama: {e}"


def check_ollama_health() -> bool:
    """Check if Ollama server is reachable."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def list_models() -> list:
    """Return list of locally available Ollama models."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        data = response.json()
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []
