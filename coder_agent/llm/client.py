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

# Try to use legacy SSL context for older OpenSSL versions
# (Agnes API may require TLS 1.3 which old OpenSSL doesn't support)
try:
    import ssl as _ssl
    # 优先：默认 context + 不校验证书（兼容多数 OpenAI 兼容端点）
    _ssl_ctx = _ssl.create_default_context()
    _ssl_ctx.check_hostname = False
    _ssl_ctx.verify_mode = _ssl.CERT_NONE
    _SSL_CONTEXT = _ssl_ctx
except Exception:
    try:
        import ssl as _ssl
        _ssl_ctx = _ssl._create_legacy_context()
        _SSL_CONTEXT = _ssl_ctx
    except Exception:
        _SSL_CONTEXT = None


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

        # Patch SSL if needed for older OpenSSL
        if _SSL_CONTEXT is not None:
            import httpx as _httpx
            _ssl_transport = _httpx.HTTPTransport(verify=_SSL_CONTEXT)
            _ssl_hx_client = _httpx.Client(transport=_ssl_transport, timeout=60.0)
            self._client = OpenAI(**client_kwargs, http_client=_ssl_hx_client)
        else:
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
