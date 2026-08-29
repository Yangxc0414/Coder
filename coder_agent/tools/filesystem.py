"""Filesystem tools: read_file, write_file, list_files."""

from __future__ import annotations

from pathlib import Path

from .base import Tool, ToolResult


class ReadFileTool(Tool):
    name = "read_file"
    description = (
        "Read the contents of a text file from the workspace. "
        "Returns the file content as a string (large files are truncated "
        "with an offload pointer — use search_text for targeted lookup). "
        "Re-reading an UNCHANGED file returns a short summary instead of "
        "the full content — do not re-read files you have not modified."
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
    # Real-run finding: the model re-read unchanged files 17-22 times each
    # (57 reads where ~6 sufficed, burning ~40% of all steps). The cache
    # short-circuits that: same path + same (mtime_ns, size) → summary.
    CACHE_LIMIT = 64
    UNCHANGED_PREVIEW = 200

    def __init__(self, workspace: Path) -> None:
        self.workspace = Path(workspace).resolve()
        # path -> (mtime_ns, size, preview_of_last_content)
        self._cache: dict[str, tuple[int, int, str]] = {}

    def execute(self, args: dict[str, str]) -> ToolResult:
        try:
            resolved = self._resolve_path(args["path"])
            if not resolved.exists():
                self._cache.pop(str(resolved), None)
                return ToolResult(error=f"File not found: {resolved}")

            stat = resolved.stat()
            signature = (stat.st_mtime_ns, stat.st_size)
            key = str(resolved)

            cached = self._cache.get(key)
            if cached and (cached[0], cached[1]) == signature:
                return ToolResult(
                    output=(
                        f"[unchanged] 文件内容与上次读取时完全相同（期间没有任何修改），"
                        f"不再重复发送全文。前 {self.UNCHANGED_PREVIEW} 字符预览:\n"
                        f"{cached[2]}\n"
                        f"[如需定位特定片段，请用 search_text 在该文件内查找]"
                    )
                )

            content = resolved.read_text(encoding="utf-8")
            self._cache[key] = (stat.st_mtime_ns, stat.st_size, content[: self.UNCHANGED_PREVIEW])
            if len(self._cache) > self.CACHE_LIMIT:
                # 简单淘汰：丢掉最早的一半，避免长会话无界增长
                for k in list(self._cache)[: self.CACHE_LIMIT // 2]:
                    self._cache.pop(k, None)

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
                # 不把自己的运行产物当工作区内容展示
                if ".coder_" not in p.parts
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
                if p.name.startswith(".coder_"):
                    continue
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
