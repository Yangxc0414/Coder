"""Token counting utility using tiktoken.

Provides token estimation for messages and individual texts,
used by ContextManager to decide when to compress the conversation.

Accuracy note: counts are ESTIMATES for budgeting, not billing. For very
large texts we fall back to a chars/4 heuristic because tiktoken's Rust
encoder degrades badly on this machine (measured: 20K chars ≈ 0.6s,
200K chars ≈ 30s+); one oversized tool output must never stall the loop.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    import tiktoken
except ImportError:  # pragma: no cover
    tiktoken = None


# Mapping from model name patterns to tiktoken encodings.
# We try common encodings and fall back to cl100k_base (GPT-4 family).
_MODEL_ENCODINGS: dict[str, str] = {
    "gpt-4": "cl100k_base",
    "gpt-3.5": "p50k_base",
    "gpt-3": "p50k_base",
    "text-davinci": "p50k_edit",
    "ada": "r50k_base",
    "babbage": "r50k_base",
}

# Texts at or above this length use the fast heuristic instead of tiktoken.
_LARGE_TEXT_CHARS = 20_000

# Encoding cache — one lookup per model per process.
_ENCODING_CACHE: dict[str, "tiktoken.Encoding"] = {}
# 失败降级缓存：避免每次调用都重试网络（断网环境下 tiktoken 首次加载 BPE 会失败）
_FALLBACK_ACTIVE: dict[str, bool] = {}


def _heuristic_tokens(text: str) -> int:
    """~4 chars/token 估算（代码密集内容偏保守，预算场景可接受）。"""
    return len(text) // 4 if text else 0


def _encoding_for_model(model: str) -> "tiktoken.Encoding | None":
    """Get the tiktoken encoding for a model name (cached).

    Returns None when tiktoken is unavailable or the BPE download failed
    (offline environments) — callers then use the char heuristic.
    """
    if tiktoken is None:
        return None
    cached = _ENCODING_CACHE.get(model)
    if cached is not None:
        return cached
    if _FALLBACK_ACTIVE.get(model):
        return None
    model_lower = model.lower()
    enc_name = "cl100k_base"
    for pattern, name in _MODEL_ENCODINGS.items():
        if pattern in model_lower:
            enc_name = name
            break
    try:
        enc = tiktoken.get_encoding(enc_name)
    except Exception as e:
        logger.debug("tiktoken unavailable for %s (%s); using char heuristic",
                     model, type(e).__name__)
        _FALLBACK_ACTIVE[model] = True
        return None
    _ENCODING_CACHE[model] = enc
    return enc


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Count tokens in a single text string."""
    if len(text) >= _LARGE_TEXT_CHARS:
        # ~4 chars per token is a safe over/under-estimate for budgeting
        return len(text) // 4
    enc = _encoding_for_model(model)
    if enc is None:
        return _heuristic_tokens(text)
    return len(enc.encode(text))


def count_message_tokens(message: dict, model: str = "gpt-4o") -> int:
    """Estimate token count for a single message dict.

    Accounts for role prefix and tool_call structure per OpenAI formatting.
    """
    # Base tokens per message (role + content overhead)
    tokens = 4  # roughly ~4 tokens for message wrapper
    content = message.get("content") or ""
    # Fast path first — avoids calling the encoder on oversized content
    tokens += count_tokens(content, model)
    if len(content) >= _LARGE_TEXT_CHARS:
        return tokens
    tokens += count_tokens(message.get("role", ""), model)
    # tool_calls also consume tokens
    for tc in message.get("tool_calls") or []:
        fn = tc.get("function", {})
        tokens += count_tokens(fn.get("name", ""), model)
        tokens += count_tokens(fn.get("arguments", ""), model)
    return tokens


def count_messages_tokens(messages: list[dict], model: str = "gpt-4o") -> int:
    """Estimate total token count for a list of messages."""
    if not messages:
        return 0
    total = 0
    for msg in messages:
        total += count_message_tokens(msg, model)
    # Add tokens for the message list wrapper
    total += 6
    return total
