"""Token counting utility using tiktoken.

Provides token estimation for messages and individual texts,
used by ContextManager to decide when to compress the conversation.
"""

from __future__ import annotations

import tiktoken


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


def _encoding_for_model(model: str) -> tiktoken.Encoding:
    """Get the tiktoken encoding for a model name."""
    model_lower = model.lower()
    for pattern, enc_name in _MODEL_ENCODINGS.items():
        if pattern in model_lower:
            return tiktoken.get_encoding(enc_name)
    # Default to cl100k_base (GPT-4 / most OpenAI-compatible models)
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Count tokens in a single text string."""
    enc = _encoding_for_model(model)
    return len(enc.encode(text))


def count_message_tokens(message: dict, model: str = "gpt-4o") -> int:
    """Estimate token count for a single message dict.

    Accounts for role prefix and tool_call structure per OpenAI formatting.
    """
    enc = _encoding_for_model(model)
    # Base tokens per message (role + content overhead)
    tokens = 4  # roughly ~4 tokens for message wrapper
    tokens += len(enc.encode(message.get("role", "")))
    content = message.get("content") or ""
    tokens += len(enc.encode(content))
    # tool_calls also consume tokens
    for tc in message.get("tool_calls") or []:
        fn = tc.get("function", {})
        tokens += len(enc.encode(fn.get("name", "")))
        tokens += len(enc.encode(fn.get("arguments", "")))
    return tokens


def count_messages_tokens(messages: list[dict], model: str = "gpt-4o") -> int:
    """Estimate total token count for a list of messages."""
    total = 0
    for msg in messages:
        total += count_message_tokens(msg, model)
    # Add tokens for the message list wrapper
    total += 6
    return total
