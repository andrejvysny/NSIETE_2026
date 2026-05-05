"""Utility for surgical cell-level edits to Jupyter notebooks.

Used by scripts/wire_nb_*.py. Not part of the runtime project.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_nb(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def save_nb(path: Path, nb: dict) -> None:
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
        f.write("\n")


def set_cell_source(nb: dict, idx: int, source: str) -> None:
    """Replace cell source. Splits to lines preserving newlines for nb format."""
    lines = source.splitlines(keepends=True)
    nb["cells"][idx]["source"] = lines
    # Reset outputs and execution count for code cells we modified
    if nb["cells"][idx]["cell_type"] == "code":
        nb["cells"][idx]["outputs"] = []
        nb["cells"][idx]["execution_count"] = None


def insert_cell(nb: dict, idx: int, cell_type: str, source: str) -> None:
    """Insert a new cell at position idx."""
    lines = source.splitlines(keepends=True)
    cell = {"cell_type": cell_type, "metadata": {}, "source": lines}
    if cell_type == "code":
        cell["outputs"] = []
        cell["execution_count"] = None
    nb["cells"].insert(idx, cell)
