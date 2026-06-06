"""
Ollama API wrapper — handles raw HTTP calls to local Ollama server.
Provides a clean interface for generating completions from a local LLM.
"""

# import requests
# import json
# import logging

# logger = logging.getLogger(__name__)

# OLLAMA_BASE_URL = "http://127.0.0.1:11434"
# DEFAULT_MODEL = "phi3"


# def generate(prompt: str, model: str = DEFAULT_MODEL, stream: bool = False) -> str:
#     """
#     Send a prompt to the Ollama /api/generate endpoint and return the response text.

#     Args:
#         prompt: The text prompt to send.
#         model: Ollama model name (default: phi3).
#         stream: Whether to stream the response (default: False).

#     Returns:
#         Generated text as a string, or an error message on failure.
#     """
#     url = f"{OLLAMA_BASE_URL}/api/generate"
#     payload = {
#         "model": model,
#         "prompt": prompt,
#         "stream": stream,
#     }

#     try:
#         logger.debug(f"Calling Ollama model={model}, prompt_length={len(prompt)}")
#         response = requests.post(url, json=payload, timeout=180)
#         response.raise_for_status()

#         if stream:
#             # Collect streamed chunks
#             full_text = ""
#             for line in response.iter_lines():
#                 if line:
#                     chunk = json.loads(line)
#                     full_text += chunk.get("response", "")
#                     if chunk.get("done", False):
#                         break
#             return full_text.strip()
#         else:
#             data = response.json()
#             return data.get("response", "").strip()

#     except requests.exceptions.ConnectionError:
#         logger.error("Cannot connect to Ollama.")
#         return "[ERROR] CONNECTION_FAILED"

#     except requests.exceptions.Timeout:
#         logger.warning("Ollama request timed out. Retrying once...")

#         try:
#             # 🔁 RETRY with higher timeout
#             response = requests.post(url, json=payload, timeout=180)
#             response.raise_for_status()

#             if stream:
#                 full_text = ""
#                 for line in response.iter_lines():
#                     if line:
#                         chunk = json.loads(line)
#                         full_text += chunk.get("response", "")
#                         if chunk.get("done", False):
#                             break
#                 return full_text.strip()
#             else:
#                 data = response.json()
#                 result = data.get("response", "").strip()

#                 if result:
#                     return result

#                 logger.warning("Empty response after retry")
#                 return "[ERROR] EMPTY_RESPONSE"

#         except Exception as retry_error:
#             logger.error(f"Retry failed: {retry_error}")
#             return "[ERROR] TIMEOUT_AFTER_RETRY"

#     except requests.exceptions.HTTPError as e:
#         logger.error(f"Ollama HTTP error: {e}")
#         return "[ERROR] HTTP_ERROR"

#     except Exception as e:
#         logger.error(f"Unexpected Ollama error: {e}")
#         return "[ERROR] UNKNOWN_ERROR"


# def check_ollama_health() -> bool:
#     """Check if Ollama server is reachable."""
#     try:
#         response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=180)
#         logger.debug(f"Ollama health check status={response.status_code}")
#         return response.status_code == 200
#     except Exception as e:
#         logger.error(f"Ollama health check failed: {e}")
#         return False


# def list_models() -> list:
#     """Return list of locally available Ollama models."""
#     try:
#         response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=180)
#         data = response.json()
#         return [m["name"] for m in data.get("models", [])]
#     except Exception:
#         return []








import requests
import json
import logging

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "llama3"


def generate(prompt: str, model: str = DEFAULT_MODEL, stream: bool = False) -> str:
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": stream,
    }

    try:
        logger.info(f"Calling Ollama... (prompt_len={len(prompt)})")

        response = requests.post(url, json=payload, timeout=180)
        response.raise_for_status()

        if stream:
            full_text = ""
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    full_text += chunk.get("response", "")
                    if chunk.get("done", False):
                        break
            result = full_text.strip()
        else:
            data = response.json()
            result = data.get("response", "").strip()

        if not result:
            logger.warning("Empty response from Ollama")
            return "[ERROR] EMPTY_RESPONSE"

        logger.info("Ollama response received")
        return result

    except requests.exceptions.Timeout:
        logger.warning(" Timeout — retrying once...")

        try:
            response = requests.post(url, json=payload, timeout=180)
            response.raise_for_status()
            data = response.json()
            result = data.get("response", "").strip()

            if result:
                logger.info(" Ollama response received after retry")
                return result

            return "[ERROR] EMPTY_RESPONSE"

        except requests.exceptions.Timeout:
            logger.error("❌ Timeout again after retry")
            return "[ERROR] TIMEOUT_AFTER_RETRY"

        except Exception as e:
            logger.error(f"Retry failed: {e}")
            return "[ERROR] RETRY_FAILED"

    except requests.exceptions.ConnectionError:
        logger.error("❌ Cannot connect to Ollama")
        return "[ERROR] CONNECTION_FAILED"

    except requests.exceptions.HTTPError as e:
        logger.error(f"Ollama HTTP error: {e}")
        return "[ERROR] HTTP_ERROR"

    except Exception as e:
        logger.error(f"Unexpected Ollama error: {e}")
        return "[ERROR] UNKNOWN_ERROR"


def check_ollama_health() -> bool:
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return response.status_code == 200
    except Exception:
        return False
    
def list_models() -> list:
    """Return list of locally available Ollama models."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        data = response.json()
        return [m["name"] for m in data.get("models", [])]
    except Exception as e:
        logger.error(f"Error fetching model list: {e}")
        return []