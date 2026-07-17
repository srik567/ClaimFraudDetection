"""
Thin Ollama HTTP client used by the LLM reviewer.

Talks to the local Ollama server (default http://localhost:11434) and
requests JSON-mode responses so callers can parse structured output.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:latest")
DEFAULT_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))


class OllamaClient:
    """Minimal chat client for the Ollama /api/chat endpoint."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def is_available(self) -> bool:
        """Return True if the Ollama server responds to /api/tags."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def chat(
        self,
        system: str,
        user: str,
        *,
        json_mode: bool = True,
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Send a chat request and return the assistant message content.

        Raises:
            requests.RequestException: on network / HTTP errors.
            RuntimeError: if the response payload is missing message content.
        """
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": options or {"temperature": 0.2, "num_predict": 512},
        }
        if json_mode:
            payload["format"] = "json"

        url = f"{self.base_url}/api/chat"
        logger.debug("Ollama chat → model=%s url=%s", self.model, url)
        resp = requests.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        content = (data.get("message") or {}).get("content")
        if not content:
            raise RuntimeError(f"Empty Ollama response: {data!r}")
        return content
