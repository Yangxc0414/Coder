"""LLM client — wraps OpenAI-compatible API for chat completion."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from openai import OpenAI


@dataclass
class LLMResponse:
    """Unified wrapper for model response."""

    content: str | None
    tool_calls: list[dict] | None
    finish_reason: str | None
    usage: dict[str, int] | None


class LLMClient:
    """OpenAI-compatible LLM client.

    Supports any OpenAI-compatible endpoint:
    - OpenAI (default)
    - DeepSeek (OPENAI_BASE_URL=https://api.deepseek.com/v1)
    - OpenRouter, local vLLM/Ollama, etc.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.model = model
        client_kwargs: dict[str, Any] = {
            "api_key": api_key or os.getenv("OPENAI_API_KEY", ""),
        }
        if base_url:
            client_kwargs["base_url"] = base_url
        elif env_url := os.getenv("OPENAI_BASE_URL"):
            client_kwargs["base_url"] = env_url
        self._client = OpenAI(**client_kwargs)

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Send a chat completion request and return a structured response."""
        params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"

        response = self._client.chat.completions.create(**params)
        choice = response.choices[0]

        tc_list = None
        if choice.message.tool_calls:
            tc_list = [
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                }
                for tc in choice.message.tool_calls
            ]

        return LLMResponse(
            content=choice.message.content,
            tool_calls=tc_list,
            finish_reason=choice.finish_reason,
            usage=(
                {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                }
                if response.usage
                else None
            ),
        )
