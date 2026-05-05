"""Execute a Jupyter notebook in place AND stream cell output in real time.

`jupyter nbconvert --execute` waits until each cell finishes before flushing
its stdout to the terminal, which makes long training cells look stuck.
This wrapper drives the notebook via nbclient and forwards the kernel's
iopub stream messages to stdout / stderr immediately, so tqdm progress bars
and per-epoch print() lines appear as they happen.

Usage:
    uv run python scripts/run_notebook.py notebooks/07_scratch_vit.ipynb
    uv run python scripts/run_notebook.py notebooks/07_scratch_vit.ipynb --no-save

The notebook is updated in place by default (so executed outputs are saved
back to disk for diff / report). Pass --no-save to leave the notebook file
untouched (useful for smoke runs).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell  # noqa: F401  (kept for nbformat install hint)


class StreamingNotebookClient(NotebookClient):
    """NotebookClient that prints stream + error output to the terminal live."""

    def output(self, outs, msg, display_id, cell_index):  # type: ignore[override]
        result = super().output(outs, msg, display_id, cell_index)
        try:
            msg_type = msg["header"]["msg_type"]
            content = msg["content"]
            if msg_type == "stream":
                stream_name = content.get("name", "stdout")
                text = content.get("text", "")
                target = sys.stderr if stream_name == "stderr" else sys.stdout
                target.write(text)
                target.flush()
            elif msg_type == "error":
                tb = "\n".join(content.get("traceback", []))
                sys.stderr.write(tb + "\n")
                sys.stderr.flush()
        except Exception as exc:  # never let logging crash the run
            sys.stderr.write(f"[run_notebook] streaming hook error: {exc}\n")
        return result

    def on_cell_executed(self, *, cell, cell_index: int, execute_reply):  # type: ignore[override]
        super().on_cell_executed(cell=cell, cell_index=cell_index, execute_reply=execute_reply)
        elapsed = time.time() - self._cell_start
        sys.stdout.write(f"\n[cell {cell_index}] done in {elapsed:.1f}s\n")
        sys.stdout.flush()

    def on_cell_start(self, *, cell, cell_index: int):  # type: ignore[override]
        super().on_cell_start(cell=cell, cell_index=cell_index)
        self._cell_start = time.time()
        ctype = cell.get("cell_type", "?")
        head = "".join(cell.get("source", []))[:80].replace("\n", " ")
        sys.stdout.write(f"\n[cell {cell_index}] ({ctype}) >>> {head}\n")
        sys.stdout.flush()


def _normalize_ids(nb: Any) -> None:
    """Add missing cell ids to silence nbformat validation warnings."""
    try:
        from nbformat.validator import normalize
        changes, nb_norm = normalize(nb)
        # normalize returns (n_changes, normalized_nb) on recent nbformat;
        # mutate in place by replacing cells if normalization happened.
        if changes:
            nb["cells"] = nb_norm["cells"]
    except Exception:
        # Fall back to manually generating ids.
        import uuid
        for cell in nb.get("cells", []):
            if "id" not in cell:
                cell["id"] = uuid.uuid4().hex[:8]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("notebook", type=Path)
    ap.add_argument(
        "--no-save",
        action="store_true",
        help="Don't write executed outputs back to the .ipynb file.",
    )
    ap.add_argument(
        "--kernel",
        default="python3",
        help="Jupyter kernel name to use (default: python3).",
    )
    ap.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Per-cell timeout in seconds (default: no timeout).",
    )
    args = ap.parse_args()

    nb_path = args.notebook
    if not nb_path.exists():
        sys.stderr.write(f"Notebook not found: {nb_path}\n")
        return 2

    nb = nbformat.read(nb_path, as_version=4)
    _normalize_ids(nb)

    sys.stdout.write(f"[run_notebook] executing {nb_path} (kernel={args.kernel})\n")
    sys.stdout.flush()

    start = time.time()
    client = StreamingNotebookClient(
        nb,
        timeout=args.timeout,
        kernel_name=args.kernel,
        allow_errors=False,
        record_timing=True,
        resources={"metadata": {"path": str(nb_path.parent)}},
    )

    try:
        client.execute()
    except Exception as exc:
        sys.stderr.write(f"\n[run_notebook] FAILED after {time.time() - start:.1f}s: {exc}\n")
        if not args.no_save:
            try:
                nbformat.write(nb, nb_path)
                sys.stderr.write(f"[run_notebook] partial outputs saved to {nb_path}\n")
            except Exception as werr:
                sys.stderr.write(f"[run_notebook] could not save partial outputs: {werr}\n")
        return 1

    elapsed = time.time() - start
    sys.stdout.write(f"\n[run_notebook] all cells executed OK in {elapsed:.1f}s\n")
    sys.stdout.flush()

    if not args.no_save:
        nbformat.write(nb, nb_path)
        sys.stdout.write(f"[run_notebook] outputs saved to {nb_path}\n")
        sys.stdout.flush()

    return 0


if __name__ == "__main__":
    sys.exit(main())
