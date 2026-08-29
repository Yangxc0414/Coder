"""Filesystem tools: read_file, write_file, list_files."""

from __future__ import annotations

from pathlib import Path

from .base import Tool, ToolResult


class ReadFileTool(Tool):
    name = "read_file"
    description = (
        "Read the contents of a text file from the workspace. "
        "Returns the file content as a string (large files are truncated "
        "with a marker — read again in slices via search_text if needed)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path, relative to workspace root",
            }
        },
        "required": ["path"],
    }

    # ~40K chars ≈ 10K tokens — keeps one read from dominating the context
    MAX_CHARS = 40_000

    def __init__(self, workspace: Path) -> None:
        self.workspace = Path(workspace).resolve()

    def execute(self, args: dict[str, str]) -> ToolResult:
        try:
            resolved = self._resolve_path(args["path"])
            if not resolved.exists():
                return ToolResult(error=f"File not found: {resolved}")
            content = resolved.read_text(encoding="utf-8")
            if len(content) > self.MAX_CHARS:
                # Cap context usage but keep the data retrievable:
                # full content goes to an offload file the agent can read.
                from .overflow import offload_overflow

                output = offload_overflow(
                    self.workspace, content,
                    source="read_file", preview_chars=self.MAX_CHARS,
                )
                return ToolResult(output=output)
            return ToolResult(output=content)
        except Exception as e:
            return ToolResult(error=f"read_file failed: {e}")

    def _resolve_path(self, path_str: str) -> Path:
        raw = Path(path_str)
        resolved = (self.workspace / raw).resolve()
        if not str(resolved).startswith(str(self.workspace)):
            raise PermissionError(
                f"Path escapes workspace: {path_str}"
            )
        return resolved


class WriteFileTool(Tool):
    name = "write_file"
    description = (
        "Write content to a file in the workspace. "
        "Creates the file if it doesn't exist, overwrites if it does. "
        "Creates parent directories as needed."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path, relative to workspace root",
            },
            "content": {
                "type": "string",
                "description": "Content to write",
            },
        },
        "required": ["path", "content"],
    }

    def __init__(self, workspace: Path, dry_run: bool = False) -> None:
        self.workspace = Path(workspace).resolve()
        self.dry_run = dry_run

    def execute(self, args: dict[str, str]) -> ToolResult:
        try:
            resolved = self._resolve_path(args["path"])
            resolved.parent.mkdir(parents=True, exist_ok=True)
            if self.dry_run:
                # In dry-run mode, don't actually write — just report what would happen
                return ToolResult(
                    output=f"[DRY RUN] Would write {len(args['content'])} chars to {resolved}"
                )
            resolved.write_text(args["content"], encoding="utf-8")
            return ToolResult(
                output=f"Written {len(args['content'])} characters to {resolved.name}"
            )
        except Exception as e:
            return ToolResult(error=f"write_file failed: {e}")

    def _resolve_path(self, path_str: str) -> Path:
        raw = Path(path_str)
        resolved = (self.workspace / raw).resolve()
        if not str(resolved).startswith(str(self.workspace)):
            raise PermissionError(
                f"Path escapes workspace: {path_str}"
            )
        return resolved


class ListFilesTool(Tool):
    name = "list_files"
    description = (
        "List files and directories under a path, recursively. "
        "Paths are relative to the workspace root. "
        "Output is capped — narrow the path if the listing is truncated."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Subdirectory to list (default: root)",
                "default": ".",
            }
        },
        "required": [],
    }

    # A workspace with vendored/3rd-party trees can hold tens of thousands
    # of files; uncapped output once blew a 584K-token context in real use.
    MAX_ENTRIES = 300

    def __init__(self, workspace: Path) -> None:
        self.workspace = Path(workspace).resolve()

    def execute(self, args: dict[str, str]) -> ToolResult:
        try:
            target = self._resolve_path(args.get("path", "."))
            entries = sorted(
                str(p.relative_to(self.workspace))
                for p in target.rglob("*")
            )
            if not entries:
                return ToolResult(output="(directory is empty)")
            if len(entries) <= self.MAX_ENTRIES:
                return ToolResult(output="\n".join(entries))
            # Large tree: a recursive first-300 slice buries the directories
            # the user actually wants (real incident: root listing showed 300
            # .pytest_cache/OneCode files while the target dir never appeared).
            # Give a shallow map with per-directory sizes instead.
            lines = []
            for p in sorted(target.iterdir()):
                rel = p.relative_to(self.workspace).as_posix()
                if p.is_dir():
                    try:
                        n = sum(1 for _ in p.rglob("*"))
                    except OSError:
                        n = "?"
                    lines.append(f"{rel}/  ({n} entries)")
                else:
                    lines.append(rel)
            return ToolResult(
                output="\n".join(lines)
                + f"\n[... shallow overview ({len(entries)} files total) — "
                "call list_files on a subdirectory to expand it]"
            )
        except Exception as e:
            return ToolResult(error=f"list_files failed: {e}")

    def _resolve_path(self, path_str: str) -> Path:
        raw = Path(path_str)
        resolved = (self.workspace / raw).resolve()
        if not str(resolved).startswith(str(self.workspace)):
            raise PermissionError(
                f"Path escapes workspace: {path_str}"
            )
        return resolved
