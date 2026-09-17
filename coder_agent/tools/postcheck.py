"""Post-edit verification — a cheap deterministic check right after a write.

SOTA pattern (aider `linter.py` feedback loop; opencode `tool/edit.ts`
LSP-diagnostics-into-tool-result): after the agent edits a file, run a
fast, deterministic validator and append the verdict to the tool result,
so the model fixes syntax errors *in the same step* instead of discovering
them much later (or never) at the LLM-verifier stage.

This is deliberately cheap and dependency-free:
- ``.py``  -> ``ast.parse`` (syntax only, no import side effects)
- ``.json``/``.jsonc`` -> ``json.loads``
- everything else -> no check (returns None)

It layers *below* the semantic Verifier (pytest / git-diff): this one is a
fast gate that catches "I just wrote broken Python", the Verifier is the
slow gate that checks "did the task actually get done".
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Optional


_PY_SUFFIXES = (".py",)
_JSON_SUFFIXES = (".json",)


def _check_python(source: str, path: str) -> str:
    try:
        ast.parse(source, filename=path)
    except SyntaxError as e:
        return (
            f"Python syntax check FAILED (ast.parse):\n"
            f"  line {e.lineno}: {e.msg}\n"
            f"Fix the syntax before proceeding — re-write the file or "
            f"run_command 'python -c \"import ast; ast.parse(open(r{path!r}).read())\"' to see the error."
        )
    return "Python syntax check OK (ast.parse)."


def _check_json(source: str, path: str) -> str:
    try:
        json.loads(source)
    except json.JSONDecodeError as e:
        return (
            f"JSON check FAILED (json.loads):\n"
            f"  line {e.lineno} col {e.colno}: {e.msg}\n"
            f"Fix the JSON before proceeding."
        )
    return "JSON check OK (json.loads)."


def post_edit_check(path: Path, content: str) -> Optional[str]:
    """Run a cheap deterministic validator on freshly-written content.

    Returns a human-readable verdict string, or ``None`` when the file type
    has no cheap validator (in which case the caller should not append
    anything).
    """
    name = path.name.lower()
    if name.endswith(_PY_SUFFIXES):
        return _check_python(content, str(path))
    if name.endswith(_JSON_SUFFIXES):
        return _check_json(content, str(path))
    return None


__all__ = ["post_edit_check"]
