"""Execute the lesson notebooks top to bottom and save their outputs.

Usage (from the repository root):

    uv run python tools/run_notebooks.py            # all notebooks
    uv run python tools/run_notebooks.py 01_text_mining/01_mini_project_sms_spam/01_Text_Mining_Basics_SMS_Spam.ipynb

Cells that are meant to fail (to show an error on purpose) must carry the
cell tag ``raises-exception``. Any other error makes the script exit with 1.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

if sys.platform == "win32":  # the default Proactor loop hangs talking to the kernel on Windows
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = Path(__file__).resolve().parent.parent


def run(path: Path) -> bool:
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=1200,
        startup_timeout=90,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
    )
    try:
        client.execute()
    except CellExecutionError as exc:
        print(f"FAIL  {path.relative_to(ROOT)}\n{exc}")
        return False
    nbformat.write(nb, path)
    print(f"OK    {path.relative_to(ROOT)}")
    return True


def main() -> int:
    targets = [Path(p).resolve() for p in sys.argv[1:]] or sorted(ROOT.glob("0*_*/0*_*/*.ipynb"))
    results = [run(p) for p in targets]
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
