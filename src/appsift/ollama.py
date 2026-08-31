# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT
"""Minimal Ollama HTTP client. Standard library only, no third-party dependencies."""
from __future__ import annotations

import json
import urllib.request
from typing import Any


class Ollama:
    def __init__(self, host: str, timeout: int = 2400) -> None:
        self.host = host
        self.timeout = timeout

    def _get(self, path: str, timeout: int | None = None) -> dict:
        with urllib.request.urlopen(self.host + path, timeout=timeout or self.timeout) as r:
            return json.loads(r.read())

    def loaded_context(self, model: str) -> int | None:
        """The context window this model is actually loaded with, or None if it
        is not currently loaded.

        Not something a request configures here: opencode talks to Ollama over
        the OpenAI-compatible endpoint, which has no num_ctx field at all, so
        the server's own default governs every model alike. This is that value,
        read back rather than assumed, so a reader on a smaller machine can tell
        whether these results were measured at a window their own setup would
        never reach.
        """
        try:
            data = self._get("/api/ps", timeout=30)
        except Exception:
            return None
        for m in data.get("models", []):
            if m.get("name") == model or m.get("model") == model:
                return m.get("context_length")
        return None

    def _post(self, path: str, payload: dict[str, Any], timeout: int | None = None) -> dict:
        req = urllib.request.Request(
            self.host + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
            return json.loads(r.read())

    def unload(self, model: str) -> None:
        try:
            self._post("/api/chat", {"model": model, "messages": [], "keep_alive": 0}, 120)
        except Exception:
            pass
