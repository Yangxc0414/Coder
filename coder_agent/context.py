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

from .llm.tokenizer import count_messages_tokens

logger = logging.getLogger(__name__)

# Default token budget for the conversation (leaving room for tools/schema)
DEFAULT_MAX_TOKENS = 8000
# Number of recent rounds to keep in full detail
DEFAULT_KEEP_ROUNDS = 6
# Max characters for a compressed summary
DEFAULT_SUMMARY_MAX_CHARS = 150


class ContextManager:
    """Manages LLM conversation context to prevent token overflow.

    Each time before calling the LLM, call `build_messages()` to get a
    token-bounded version of the full message history.
    """

    def __init__(
        self,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        keep_rounds: int = DEFAULT_KEEP_ROUNDS,
        summary_max_chars: int = DEFAULT_SUMMARY_MAX_CHARS,
        model: str = "gpt-4o",
    ) -> None:
        self.max_tokens = max_tokens
        self.keep_rounds = keep_rounds
        self.summary_max_chars = summary_max_chars
        self.model = model

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

    def _estimate_tool_schema_tokens(self, messages: list[dict]) -> int:
        """Rough estimate of tokens consumed by tool schemas in API calls."""
        # Tool schemas are sent with every request, estimate ~500-1000 tokens
        # based on number of tools
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        # Rough: each tool def ~200 tokens + tool results vary
        return min(2000, len(tool_msgs) * 100 + 500)

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
