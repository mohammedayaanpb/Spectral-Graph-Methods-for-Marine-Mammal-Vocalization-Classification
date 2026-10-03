"""Per-run output folders for V2 notebooks.

Each notebook execution creates a fresh `runs/runN/` directory under its
level folder, with `figures/` and `results/` subfolders. Existing runs
are never modified or overwritten.

Cross-notebook dependencies (e.g. Fix 6 reads the masks Fix 1 wrote) are
resolved via `find_latest_result(level_dir, filename)`, which walks the
runs from highest N down and returns the first match.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path


def _existing_run_numbers(runs_root: Path) -> list[int]:
    if not runs_root.exists():
        return []
    out = []
    for p in runs_root.glob("run*"):
        if not p.is_dir():
            continue
        suffix = p.name[3:]
        if suffix.isdigit():
            out.append(int(suffix))
    return sorted(out)


def make_run_dir(level_dir: Path, notebook_label: str = "") -> Path:
    """Create a fresh `runs/runN/` under `level_dir` and return its path.

    Also creates `figures/` and `results/` subfolders and writes a tiny
    README.md recording timestamp and notebook label. The returned path
    is the run folder itself; callers typically work with
    `run_dir / 'figures'` and `run_dir / 'results'`.
    """
    runs_root = level_dir / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    nums = _existing_run_numbers(runs_root)
    next_n = (nums[-1] + 1) if nums else 1
    run_dir = runs_root / f"run{next_n}"
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    (run_dir / "results").mkdir(parents=True, exist_ok=True)
    label = notebook_label or "(unspecified)"
    (run_dir / "README.md").write_text(
        f"# {level_dir.name} / run{next_n}\n\n"
        f"- Created: {_dt.datetime.now().isoformat(timespec='seconds')}\n"
        f"- Notebook: {label}\n"
        f"- Edit the notebook between runs to compare configurations.\n",
        encoding="utf-8",
    )
    return run_dir


def find_latest_result(level_dir: Path, filename: str) -> Path | None:
    """Return the most recent run's `results/<filename>`, or None.

    Falls back to `level_dir / filename` if no `runs/` folder exists yet
    (legacy layout from the initial V2 commits).
    """
    runs_root = level_dir / "runs"
    if runs_root.exists():
        for n in reversed(_existing_run_numbers(runs_root)):
            candidate = runs_root / f"run{n}" / "results" / filename
            if candidate.exists():
                return candidate
    legacy = level_dir / filename
    if legacy.exists():
        return legacy
    return None
