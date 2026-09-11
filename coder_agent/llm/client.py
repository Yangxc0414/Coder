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
        self.model = model or _USER_CFG.get("model") or "agnes-3.0-flash"
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

    def list_models(self) -> list[str]:
        """列出当前端点可用的模型（复用同一配置解析与客户端）。

        Web /api/models 与 CLI --list-models 共用，避免各自手写
        base_url/api_key 解析和 urllib 请求。
        """
        try:
            models = self._client.models.list()
            return sorted(str(m.id) for m in models.data if m.id)
        except Exception:
            pass
        # SDK transport 受阻时的兜底：标准库 urllib 走不同的 TLS 路径，
        # 健康检查据此能区分"网络被干扰"与"端点真的不可用"。
        try:
            return self._list_models_urllib()
        except Exception:
            return []

    def _list_models_urllib(self) -> list[str]:
        """标准库版模型列表（SDK 传输层异常时的回退路径）。"""
        import urllib.request

        base = (self.base_url or "https://api.openai.com/v1").rstrip("/")
        key = self._client.api_key or ""
        req = urllib.request.Request(
            base + "/models", headers={"Authorization": f"Bearer {key}"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8", "replace"))
        return sorted(str(m["id"]) for m in data.get("data", []) if m.get("id"))

    def _chat_once_urllib(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        max_tokens: int,
    ) -> LLMResponse:
        """标准库版 chat（SDK 传输层被网络代理/TLS 沙箱拦截时的回退路径）。

        沙箱环境下 httpx 的 TLS 握手被中间代理破坏（EOF violation），
        urllib 走不同的 TLS 路径可通。回退实现覆盖非流式语义：
        工具调用/usage/finish_reason 与 SDK 路径对齐，on_token 流式在
        回退模式下退化为非流式（流式失败本来也走 _chat_once）。
        """
        import urllib.request

        base = (self.base_url or "https://api.openai.com/v1").rstrip("/")
        key = self._client.api_key or ""
        params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"
        req = urllib.request.Request(
            base + "/chat/completions",
            data=json.dumps(params).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8", "replace"))
        choice = data["choices"][0]
        msg = choice.get("message") or {}
        tc_list = None
        if msg.get("tool_calls"):
            tc_list = [
                {"id": tc.get("id", ""),
                 "name": (tc.get("function") or {}).get("name", ""),
                 "arguments": (tc.get("function") or {}).get("arguments", "")}
                for tc in msg["tool_calls"]
            ]
        usage = data.get("usage")
        return LLMResponse(
            content=msg.get("content"),
            tool_calls=tc_list,
            finish_reason=choice.get("finish_reason"),
            usage=(
                {"prompt_tokens": usage.get("prompt_tokens", 0),
                 "completion_tokens": usage.get("completion_tokens", 0)}
                if usage else None
            ),
        )

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
                # usage 可能出现在任何 chunk（部分 API 在带 choices 的末块）
                if getattr(chunk, "usage", None) is not None:
                    usage = {
                        "prompt_tokens": chunk.usage.prompt_tokens,
                        "completion_tokens": chunk.usage.completion_tokens,
                    }
                if not chunk.choices:
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

        try:
            response = self._client.chat.completions.create(**params)
        except Exception as sdk_err:
            # SDK 传输层异常（沙箱代理 TLS 拦截等）→ urllib 回退。
            # 仅对"连接/传输"类错误回退；API 层面的 4xx/5xx 直接上抛
            # （urllib 重放同样会拿到相同错误，不掩盖真实故障）。
            # openai.APIConnectionError / httpx.ConnectError 都算传输层故障。
            _is_transport = False
            try:
                from openai import APIConnectionError as _OAIC
                _is_transport = isinstance(sdk_err, _OAIC)
            except Exception:
                pass
            if not _is_transport:
                import httpx
                _is_transport = isinstance(sdk_err, (
                    httpx.ConnectError, httpx.ConnectTimeout,
                    httpx.ReadTimeout, httpx.RemoteProtocolError,
                    httpx.ProxyError,
                ))
            if _is_transport:
                try:
                    return self._chat_once_urllib(messages, tools, max_tokens)
                except Exception:
                    raise sdk_err
            raise
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
