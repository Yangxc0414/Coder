"""Context Inspector — shows current context status and token usage."""

from __future__ import annotations

from .llm.tokenizer import count_messages_tokens


class ContextInspector:
    """Inspect current context state without modifying it.

    Used by the CLI to show the user:
    - Current token count
    - Budget limit
    - Whether compression is active
    - How many rounds are kept vs compressed
    """

    def __init__(
        self,
        max_tokens: int = 8000,
        keep_rounds: int = 6,
        model: str = "gpt-4o",
    ) -> None:
        self.max_tokens = max_tokens
        self.keep_rounds = keep_rounds
        self.model = model

    def inspect(self, messages: list[dict]) -> dict:
        """Return context status information."""
        total_tokens = count_messages_tokens(messages, self.model)
        budget = max(1000, self.max_tokens - 1500)  # ~1500 for tool schemas
        utilization = (total_tokens / self.max_tokens * 100) if self.max_tokens > 0 else 0
        needs_compress = total_tokens > budget

        # Count rounds
        rounds = self._count_rounds(messages)

        return {
            "total_messages": len([m for m in messages if m.get("role") != "system"]),
            "total_tokens": total_tokens,
            "budget": budget,
            "utilization_pct": round(utilization, 1),
            "needs_compression": needs_compress,
            "rounds_count": rounds,
            "keep_rounds": self.keep_rounds,
            "max_tokens": self.max_tokens,
        }

    def print_status(self, messages: list[dict]) -> None:
        """Print a human-readable status report."""
        info = self.inspect(messages)
        print("\n┌─ Context Status ──────────────────────────────┐")
        print(f"│ Messages:    {info['total_messages']:>4}                    │")
        print(f"│ Tokens:      {info['total_tokens']:>5} / {info['max_tokens']}           │")
        print(f"│ Utilization: {info['utilization_pct']:>5.1f}%                   │")
        print(f"│ Rounds:      {info['rounds_count']:>4} (keeping {info['keep_rounds']} full)   │")
        status = "⚠️  NEEDS COMPRESSION" if info["needs_compression"] else "✅ OK"
        print(f"│ Status:      {status:<23} │")
        print("└────────────────────────────────────────────────┘")

    def _count_rounds(self, messages: list[dict]) -> int:
        """Count conversation rounds (user/assistant pairs)."""
        rounds = 0
        for msg in messages:
            if msg.get("role") in ("user", "assistant"):
                rounds += 1
        return rounds // 2
