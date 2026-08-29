"""Overflow offloading — cap tool output in context without losing data.

Pattern borrowed from opencode's truncate.ts and OneCode's ToolResultStorage:
when a tool's output exceeds the context-safe cap, the FULL content is
written to a file inside the workspace and the tool returns a preview plus
a pointer. The model can retrieve specific parts later via read_file /
search_text on that file — context stays bounded, data is not destroyed.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

# Directory (inside the workspace) where offloaded outputs live
OFFLOAD_DIR = ".coder_truncs"


def offload_overflow(
    workspace: Path,
    full_content: str,
    source: str,
    preview_chars: int,
) -> str:
    """Save *full_content* to a file, return a preview + retrieval pointer.

    Args:
        workspace: Tool workspace root (the offload dir lives here so the
            agent can read it back with its own read_file tool).
        full_content: The complete tool output that exceeded the cap.
        source: Short label for the filename (e.g. "read_file", "search_text").
        preview_chars: How many characters of the original output to keep
            inline in the returned message.

    Returns:
        preview + truncation marker + the offload file path.
    """
    offload_dir = workspace / OFFLOAD_DIR
    offload_dir.mkdir(parents=True, exist_ok=True)

    digest = hashlib.sha1(full_content.encode("utf-8", errors="replace")).hexdigest()[:8]
    stamp = time.strftime("%H%M%S")
    name = f"{source}_{stamp}_{digest}.txt"
    offload_path = offload_dir / name
    offload_path.write_text(full_content, encoding="utf-8")

    rel_path = f"{OFFLOAD_DIR}/{name}"
    preview = full_content[:preview_chars]
    return (
        f"{preview}\n"
        f"[... output truncated for context ({len(full_content)} chars total). "
        f"FULL content saved to: {rel_path} — use search_text on it to find "
        f"the specific part you need]"
    )
