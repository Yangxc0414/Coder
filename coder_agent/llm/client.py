"""LLM client — wraps OpenAI-compatible API for chat completion."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI


def _user_config() -> dict:
    """读取 ~/.coder_config.json（模型/API 配置），不存在时返回空 dict。"""
    try:
        return json.loads((Path.home() / ".coder_config.json")
                          .read_text(encoding="utf-8"))
    except Exception:
        return {}


_USER_CFG = _user_config()


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
        # 配置优先级：显式参数 > ~/.coder_config.json > 环境变量
        self.model = model or _USER_CFG.get("model") or "agnes-2.5-flash"
        self.base_url = (base_url or _USER_CFG.get("base_url")
                         or os.getenv("OPENAI_BASE_URL"))
        resolved_key = (api_key or _USER_CFG.get("api_key")
                        or os.getenv("OPENAI_API_KEY", ""))
        client_kwargs: dict[str, Any] = {
            "api_key": resolved_key,
        }
        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        # 使用 SDK 默认 transport（自定义 SSL context 会导致部分端点 401）
        import httpx as _httpx
        self._client = OpenAI(**client_kwargs,
                              http_client=_httpx.Client(timeout=60.0))

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
        on_token: Any = None,
    ) -> LLMResponse:
        """Send a chat completion request and return a structured response.

        Args:
            on_token: Optional callback receiving each content token delta as
                it arrives (streaming mode). When provided, the request is
                streamed and the callback fires per chunk; the returned
                LLMResponse still contains the full accumulated content.
                Tool-call deltas are aggregated internally.
        """
        if on_token is None:
            return self._chat_once(messages, tools, max_tokens)
        # 流式模式：逐块回调内容，聚合 tool_calls
        params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"
        try:
            stream = self._client.chat.completions.create(**params)
            content_parts: list[str] = []
            tool_calls: dict[int, dict[str, str]] = {}
            finish_reason: str | None = None
            usage: dict[str, int] | None = None
            for chunk in stream:
                if not chunk.choices:
                    if getattr(chunk, "usage", None):
                        usage = {
                            "prompt_tokens": chunk.usage.prompt_tokens,
                            "completion_tokens": chunk.usage.completion_tokens,
                        }
                    continue
                choice = chunk.choices[0]
                if choice.finish_reason:
                    finish_reason = choice.finish_reason
                delta = choice.delta
                if delta and delta.content:
                    content_parts.append(delta.content)
                    on_token(delta.content)
                if delta and delta.tool_calls:
                    for tc in delta.tool_calls:
                        entry = tool_calls.setdefault(
                            tc.index, {"id": "", "name": "", "arguments": ""})
                        if tc.id:
                            entry["id"] = tc.id
                        if tc.function and tc.function.name:
                            entry["name"] = tc.function.name
                        if tc.function and tc.function.arguments:
                            entry["arguments"] += tc.function.arguments
            tc_list = None
            if tool_calls:
                tc_list = [
                    {"id": v["id"], "name": v["name"],
                     "arguments": v["arguments"]}
                    for _, v in sorted(tool_calls.items())
                ]
            return LLMResponse(
                content="".join(content_parts) or None,
                tool_calls=tc_list,
                finish_reason=finish_reason,
                usage=usage,
            )
        except Exception:
            # 流式失败时降级为非流式（保证可用性）
            return self._chat_once(messages, tools, max_tokens)

    def _chat_once(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        max_tokens: int,
    ) -> LLMResponse:
        """非流式单次请求。"""
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
