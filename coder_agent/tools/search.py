"""Search tool — find text patterns in workspace files.

Inspired by Aider's grep-like search and SWE-agent's file exploration tools.
Supports both simple substring search and regex patterns.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from .base import Tool, ToolResult


class SearchTextTool(Tool):
    name = "search_text"
    description = (
        "Search for a text pattern in files within the workspace. "
        "Supports simple substring search and regex. "
        "Returns matching lines with file:line_number: content format."
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Text or regex pattern to search for",
            },
            "path": {
                "type": "string",
                "description": "Directory to search in (default: workspace root)",
                "default": ".",
            },
            "file_pattern": {
                "type": "string",
                "description": "Glob pattern for file names (default: '*')",
                "default": "*",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return (default: 30)",
                "default": 30,
            },
        },
        "required": ["pattern"],
    }

    MAX_OUTPUT_CHARS = 6000

    def __init__(self, workspace: Path) -> None:
        self.workspace = Path(workspace).resolve()

    def execute(self, args: dict[str, str]) -> ToolResult:
        pattern = args["pattern"]
        search_path = args.get("path", ".")
        file_pattern = args.get("file_pattern", "*")
        max_results = int(args.get("max_results", 30))

        try:
            base = self._resolve_path(search_path)
        except PermissionError as e:
            return ToolResult(error=str(e))

        # Compile regex (fall back to substring match if invalid)
        use_regex = False
        try:
            regex = re.compile(pattern)
            use_regex = True
        except re.error:
            regex = re.compile(re.escape(pattern))  # treat as literal

        # Find matching files
        search_root = base
        matches = []
        files_scanned = 0

        for filepath in search_root.rglob(file_pattern):
            if not filepath.is_file():
                continue
            if any(part.startswith(".coder_") for part in filepath.parts):
                continue  # 跳过自身运行产物（转存/日志/记忆文件）
            files_scanned += 1
            if files_scanned > 500:  # safety limit
                break
            try:
                content = filepath.read_text(encoding="utf-8", errors="ignore")
            except (OSError, UnicodeDecodeError):
                continue

            rel_path = filepath.relative_to(self.workspace).as_posix()

            for line_no, line in enumerate(content.splitlines(), 1):
                if use_regex:
                    if regex.search(line):
                        matches.append(f"{rel_path}:{line_no}:{line.rstrip()[:500]}")
                else:
                    if pattern in line:
                        matches.append(f"{rel_path}:{line_no}:{line.rstrip()[:500]}")

                if len(matches) >= max_results:
                    break
            if len(matches) >= max_results:
                break

        if not matches:
            return ToolResult(
                output=f"No matches found for '{pattern}' "
                       f"(scanned {files_scanned} files in {search_path})"
            )

        output = "\n".join(matches)
        if len(output) > self.MAX_OUTPUT_CHARS:
            # Keep all matches retrievable — full list goes to an offload file
            from .overflow import offload_overflow

            full = f"Found {len(matches)} matches in {files_scanned} files:\n{output}"
            output = offload_overflow(
                self.workspace, full,
                source="search_text", preview_chars=self.MAX_OUTPUT_CHARS,
            )
            return ToolResult(output=output)

        return ToolResult(
            output=f"Found {len(matches)} matches in {files_scanned} files:\n{output}"
        )

    def _resolve_path(self, path_str: str) -> Path:
        raw = Path(path_str)
        resolved = (self.workspace / raw).resolve()
        try:
            import os as _os
            common = _os.path.commonpath([str(self.workspace), str(resolved)])
        except ValueError:
            raise PermissionError(f"Path escapes workspace: {path_str}")
        if common != str(self.workspace):
            raise PermissionError(f"Path escapes workspace: {path_str}")
        return resolved
