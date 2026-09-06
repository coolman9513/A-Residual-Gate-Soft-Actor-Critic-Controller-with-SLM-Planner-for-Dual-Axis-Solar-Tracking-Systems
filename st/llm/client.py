"""OpenAI-compatible client for Qwen served by LM Studio."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class QwenClient:
    """Small wrapper around LM Studio's OpenAI-compatible chat endpoint.

    Parameters
    ----------
    base_url:
        LM Studio server URL. The default matches the usual local server.
    api_key:
        Placeholder key; LM Studio does not validate it.
    model:
        Model id exposed by LM Studio.
    timeout:
        Request timeout in seconds.
    """

    base_url: str = "http://localhost:1234/v1"
    api_key: str = "lm-studio"
    model: str = "qwen3-1.7b"
    timeout: float = 120.0

    def _message_text(self, message: Any) -> str:
        """Return text from OpenAI/LM Studio message objects.

        Some Qwen3/LM Studio configurations may place generated text in
        provider-specific fields instead of ``message.content``. This helper
        checks common alternatives before declaring the response empty.
        """

        candidates = []
        for attr in ["content", "reasoning_content", "reasoning", "text"]:
            if hasattr(message, attr):
                candidates.append(getattr(message, attr))

        model_extra = getattr(message, "model_extra", None)
        if isinstance(model_extra, dict):
            for key in ["content", "reasoning_content", "reasoning", "text"]:
                candidates.append(model_extra.get(key))

        if isinstance(message, dict):
            for key in ["content", "reasoning_content", "reasoning", "text"]:
                candidates.append(message.get(key))

        for value in candidates:
            if isinstance(value, str) and value.strip():
                return value.strip()
            if isinstance(value, list):
                parts = []
                for item in value:
                    if isinstance(item, dict):
                        parts.append(str(item.get("text", "")))
                    else:
                        parts.append(str(item))
                joined = "".join(parts).strip()
                if joined:
                    return joined

        return ""

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 512,
    ) -> str:
        """Return raw assistant text from Qwen.

        The parser/validator is intentionally kept outside this class so the
        fallback logic can be tested without requiring LM Studio to be running.
        """

        from openai import OpenAI

        client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout)
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return self._message_text(response.choices[0].message)
