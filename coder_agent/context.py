"""Context Manager — controls conversation length to stay within token budget.

Three-layer compression strategy:
1. Always keep the system prompt intact
2. Keep the most recent N rounds (full messages)
3. Compress older messages into short summaries

Inspired by:
- Aider's Repo Map (controls context window usage)
- smolagents' write_memory_to_messages(summary_mode=True)
"""

from __future__ import annotations

import logging
from typing import Any

from .llm.tokenizer import count_message_tokens, count_messages_tokens

logger = logging.getLogger(__name__)

# Default token budget for the conversation (leaving room for tools/schema)
DEFAULT_MAX_TOKENS = 8000
# Number of recent rounds to keep in full detail
DEFAULT_KEEP_ROUNDS = 6
# Max characters for a compressed summary
DEFAULT_SUMMARY_MAX_CHARS = 150

# 模型上下文窗口映射（模式匹配，单位 tokens）——压缩阈值按窗口比例计算
MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "agnes-2.5-pro": 262144,   # 256K
    "agnes-2.0-pro": 262144,
    "agnes": 131072,           # agnes 系列默认 128K
    "deepseek": 65536,
    "gpt-4o": 131072,
    "gpt-4-turbo": 131072,
    "gpt-4": 8192,
    "gpt-3.5": 16385,
    "claude": 200000,
    "gemini": 1048576,
}
DEFAULT_CONTEXT_WINDOW = 131072
# 对话预算占窗口的比例（其余留给系统提示 + 工具 schema）
DEFAULT_CONTEXT_RATIO = 0.8


def resolve_context_window(model: str) -> int:
    """按模型名解析上下文窗口大小（模式匹配 + 默认 128K）。"""
    m = (model or "").lower()
    for pattern, win in MODEL_CONTEXT_WINDOWS.items():
        if pattern in m:
            return win
    return DEFAULT_CONTEXT_WINDOW


class ContextManager:
    """Manages LLM conversation context to prevent token overflow.

    Each time before calling the LLM, call `build_messages()` to get a
    token-bounded version of the full message history.

    Budget is model-aware: when max_tokens is not given explicitly it
    defaults to ``context_window * DEFAULT_CONTEXT_RATIO`` (e.g. 80% of
    the model's context window), so different models compress at
    different thresholds automatically.
    """

    def __init__(
        self,
        max_tokens: int | None = None,
        keep_rounds: int = DEFAULT_KEEP_ROUNDS,
        summary_max_chars: int = DEFAULT_SUMMARY_MAX_CHARS,
        model: str = "gpt-4o",
        context_window: int | None = None,
    ) -> None:
        self.model = model
        self.context_window = (context_window
                               or resolve_context_window(model))
        self.max_tokens = (max_tokens if max_tokens is not None
                           else int(self.context_window * DEFAULT_CONTEXT_RATIO))
        self.keep_rounds = keep_rounds
        self.summary_max_chars = summary_max_chars
        # 最近一次 build_messages 的压缩统计（UI 展示用；None=尚未调用/无压缩）
        self.last_compression: dict | None = None

    def build_messages(
        self,
        system: str,
        all_messages: list[dict],
    ) -> list[dict]:
        """Build a token-bounded message list.

        Strategy:
        1. Always include the system prompt
        2. From the end, keep `keep_rounds` full rounds (user+assistant+tool)
        3. Compress older messages into summaries
        4. If still over budget, further truncate summaries
        """
        # Calculate token budget for conversation (leave room for tools schema)
        tool_schema_tokens = self._estimate_tool_schema_tokens(all_messages)
        budget = max(1000, self.max_tokens - tool_schema_tokens)

        # Separate system from conversation
        conv_messages = [m for m in all_messages if m.get("role") != "system"]
        # Cap oversized messages on ALL paths — including the no-compression
        # early return, where a single huge tool output would otherwise pass
        # through uncapped (584K-token incident).
        conv_messages = self._cap_oversized_messages(conv_messages, budget)
        # System is always kept
        result = [{"role": "system", "content": system}]

        if not conv_messages:
            return result

        # Split into rounds: each round = [user/assistant, (tool results)]
        rounds = self._split_into_rounds(conv_messages)

        total_rounds = len(rounds)
        keep_from_end = min(self.keep_rounds * 2, total_rounds)  # 2 messages per round
        compress_count = total_rounds - keep_from_end

        if compress_count <= 0:
            # No compression needed, return all messages
            self.last_compression = None
            result.extend(conv_messages)
            return result

        # Compress old rounds, keep recent rounds full
        compressed = []
        for i, round_msgs in enumerate(rounds[:compress_count]):
            summary = self._compress_round(round_msgs)
            compressed.append(summary)

        recent = rounds[compress_count:]
        recent_msgs = []
        for r in recent:
            recent_msgs.extend(r)

        # Build final list: system + compressed summaries + recent full messages
        result.append({
            "role": "user",
            "content": f"[Earlier conversation summarized ({compress_count} rounds ago)]\n"
                        + "\n---\n".join(compressed),
        })
        result.extend(recent_msgs)

        # Final check: if still over budget, truncate the summary portion
        current_tokens = count_messages_tokens(result, self.model)
        if current_tokens > budget:
            result = self._truncate_to_budget(result, budget)

        logger.debug(
            "ContextManager: %d total rounds, keeping %d full, compressing %d, "
            "estimated %d tokens (budget: %d)",
            total_rounds, keep_from_end // 2, compress_count,
            current_tokens, budget,
        )
        # 记录最近一次压缩统计（UI 展示三层压缩逻辑用）
        self.last_compression = {
            "total_rounds": total_rounds,
            "kept_full": keep_from_end // 2,
            "compressed": compress_count,
            "tokens_est": current_tokens,
            "budget": budget,
        }
        return result

    def _split_into_rounds(self, messages: list[dict]) -> list[list[dict]]:
        """Split messages into rounds. Each round starts with a user/assistant
        message and includes any following tool messages."""
        rounds: list[list[dict]] = []
        current_round: list[dict] = []

        for msg in messages:
            role = msg.get("role")
            if role in ("user", "assistant"):
                if current_round:
                    rounds.append(current_round)
                current_round = [msg]
            elif role == "tool" and current_round:
                current_round.append(msg)
            else:
                # Stray messages go into current round or start a new one
                if current_round:
                    current_round.append(msg)
                else:
                    current_round = [msg]

        if current_round:
            rounds.append(current_round)

        return rounds

    def _compress_round(self, round_msgs: list[dict]) -> str:
        """Compress a round of messages into a short summary string."""
        parts = []
        for msg in round_msgs:
            role = msg.get("role", "")
            content = msg.get("content") or ""
            tc = msg.get("tool_calls")

            if role == "user":
                # Truncate user content
                preview = content[:self.summary_max_chars]
                parts.append(f"User: {preview}{'...' if len(content) > self.summary_max_chars else ''}")
            elif role == "assistant":
                if tc:
                    tool_names = [t.get("function", {}).get("name", "?") for t in tc]
                    parts.append(f"Assistant called: {', '.join(tool_names)}")
                else:
                    preview = content[:self.summary_max_chars]
                    parts.append(f"Assistant: {preview}{'...' if len(content) > self.summary_max_chars else ''}")
            elif role == "tool":
                tool_id = msg.get("tool_call_id", "")
                output = content[:self.summary_max_chars]
                parts.append(f"Tool({tool_id}): {output}{'...' if len(content) > self.summary_max_chars else ''}")

        return "\n".join(parts)

    def _cap_oversized_messages(
        self, messages: list[dict], budget: int
    ) -> list[dict]:
        """Cap any single message that would dominate the token budget.

        Even inside the "recent rounds kept full" window, one oversized tool
        output (e.g. a huge file read) must not exceed the budget — otherwise
        _truncate_to_budget's fallback would discard the entire history.
        Returns copies; the caller's message list is not mutated.
        """
        per_msg_cap = int(budget * 0.6)  # tokens a single message may occupy
        capped = []
        for msg in messages:
            if count_message_tokens(msg, self.model) <= per_msg_cap:
                capped.append(msg)
                continue
            content = msg.get("content") or ""
            if not content:
                # e.g. assistant tool_calls-only message — keep structure intact
                capped.append(msg)
                continue
            # ~3 chars/token is conservative for code-heavy content
            keep_chars = per_msg_cap * 3
            new_msg = dict(msg)
            new_msg["content"] = (
                content[:keep_chars]
                + f"\n[... truncated by ContextManager: {len(content)} -> "
                f"{keep_chars} chars to fit the context budget; "
                "use search_text for targeted lookup]"
            )
            capped.append(new_msg)
        return capped

    def _estimate_tool_schema_tokens(self, messages: list[dict]) -> int:
        """Estimate tokens consumed by tool schemas sent with each API call.

        Tool schemas are included in every request, so we estimate based on
        the actual tool definitions rather than message count.
        """
        # Each tool definition averages ~150-250 tokens
        # We estimate from the number of tool result messages (proxy for tools used)
        return 1500  # conservative estimate for 4-5 tools

    def _truncate_to_budget(self, messages: list[dict], max_tokens: int) -> list[dict]:
        """Aggressively truncate messages to fit within token budget."""
        # Remove oldest summary content first
        if len(messages) <= 2:
            return messages

        # Keep only the last N messages that fit
        for keep in range(len(messages) - 1, 1, -1):
            subset = messages[:2] + messages[keep:]  # keep system + last `keep` msgs
            tokens = count_messages_tokens(subset, self.model)
            if tokens <= max_tokens:
                return subset

        return messages[:2]  # fallback: system + last message only
