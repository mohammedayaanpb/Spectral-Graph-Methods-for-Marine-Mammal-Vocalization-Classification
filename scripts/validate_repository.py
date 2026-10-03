"""Validate source and notebooks without executing experiments or downloading data.

Run from any working directory: python scripts/validate_repository.py
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import re
import sys

from IPython.core.inputtransformer2 import TransformerManager
import nbformat


ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRECTORIES = {
    ".git", ".local-backup", ".venv", "venv", "__pycache__",
    ".ipynb_checkpoints", ".pytest_cache", "node_modules", "dist", "build", "Mss",
}
HOME_PATH = re.compile(
    r"(?:[A-Za-z]:[/\\]+(?:Users|Documents and Settings)[/\\]+[^/\\\s\"']+"
    r"|/(?:home|Users)/[^/\s\"']+)",
    re.IGNORECASE,
)


def source_files():
    for directory, subdirectories, filenames in os.walk(ROOT):
        subdirectories[:] = sorted(set(subdirectories) - SKIP_DIRECTORIES)
        for filename in sorted(filenames):
            path = Path(directory) / filename
            if path.suffix in {".py", ".ipynb"}:
                yield path


def main() -> int:
    errors = []
    counts = {".py": 0, ".ipynb": 0}
    transformer = TransformerManager()
    for path in source_files():
        relative = path.relative_to(ROOT)
        counts[path.suffix] += 1
        try:
            text = path.read_text(encoding="utf-8-sig")
            if path.suffix == ".py":
                ast.parse(text, filename=str(relative))
                if HOME_PATH.search(text):
                    errors.append(f"{relative}: hardcoded user-home path")
                continue
            notebook = json.loads(text)
            nbformat.validate(notebook)
            for number, cell in enumerate(notebook["cells"], start=1):
                if cell["cell_type"] != "code":
                    continue
                source = cell["source"]
                if isinstance(source, list):
                    source = "".join(source)
                location = f"{relative}, cell {number}"
                if HOME_PATH.search(source):
                    errors.append(f"{location}: hardcoded user-home path")
                try:
                    # IPython transforms shell commands and magics to Python.
                    compile(transformer.transform_cell(source), location, "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
                except (SyntaxError, ValueError) as exc:
                    errors.append(f"{location}: {exc}")
        except Exception as exc:
            errors.append(f"{relative}: {type(exc).__name__}: {exc}")
    if errors:
        print("Repository validation failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print(f"Validated {counts['.py']} Python files and {counts['.ipynb']} notebooks.")
    print("Checked syntax, notebook schemas, and portable user-home paths; experiments were not executed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
